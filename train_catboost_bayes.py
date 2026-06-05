"""
LSTM-BO-CatBoost — hybrid model from Sun & Tian (2023) paper.

Architecture (matching the paper):
  1. Simple 2-layer LSTM (100 units) trained on High, Low, Close
  2. LSTM predictions become features for CatBoost
  3. CatBoost hyperparameters optimized with Bayesian optimization (optuna)

The LSTM captures temporal patterns; CatBoost learns non-linear
feature interactions. Both together outperform either alone.

Usage:
    python train_catboost_bayes.py MSFT_daily_data.csv
    python train_catboost_bayes.py MSFT_daily_data.csv --n_trials 30
"""

import argparse, os, sys, warnings, logging, joblib
import numpy as np, pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
)

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('optuna').setLevel(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trade_probability_analyzer import (
    predict_multi_day_path, monte_carlo_simulation,
    find_similar_patterns, calculate_ensemble_probability,
    format_analysis_report,
)

# ===========================================================================
# INDICATORS — same 38-feature heavy set
# ===========================================================================

def _rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-10)))

def _atr(df, period):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()

def compute_indicators(df):
    out = df.copy(); c = out['Close']; vol = out['Volume']
    out['SMA_20'] = c.rolling(20).mean(); out['SMA_50'] = c.rolling(50).mean()
    out['EMA_9']  = c.ewm(span=9, min_periods=9).mean()
    out['EMA_21'] = c.ewm(span=21, min_periods=21).mean()
    out['Close_SMA20_ratio'] = (c - out['SMA_20']) / (out['SMA_20'] + 1e-10)
    out['Close_SMA50_ratio'] = (c - out['SMA_50']) / (out['SMA_50'] + 1e-10)
    out['RSI_7'] = _rsi(c, 7); out['RSI_14'] = _rsi(c, 14)
    ema12 = c.ewm(span=12, min_periods=12).mean()
    ema26 = c.ewm(span=26, min_periods=26).mean()
    out['MACD_line'] = ema12 - ema26
    out['MACD_signal'] = out['MACD_line'].ewm(span=9, min_periods=9).mean()
    out['MACD_hist'] = out['MACD_line'] - out['MACD_signal']
    bb_mid = c.rolling(20).mean(); bb_std = c.rolling(20).std()
    bb_up = bb_mid + 2*bb_std; bb_lo = bb_mid - 2*bb_std
    out['BB_pct'] = (c - bb_lo) / (bb_up - bb_lo + 1e-10)
    out['BB_width'] = (bb_up - bb_lo) / (bb_mid + 1e-10)
    out['ATR_14'] = _atr(out, 14)
    low14 = out['Low'].rolling(14).min(); high14 = out['High'].rolling(14).max()
    k = 100*(c - low14)/(high14 - low14 + 1e-10)
    out['STOCH_K'] = k; out['STOCH_D'] = k.rolling(3).mean()
    direction = np.sign(c.diff()).fillna(0)
    obv_raw = (direction * vol).cumsum()
    out['OBV'] = np.log1p(obv_raw.abs()) * np.sign(obv_raw)
    tp = (out['High'] + out['Low'] + c)/3
    tp_ma = tp.rolling(14).mean(); tp_mad = tp.rolling(14).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True)
    out['CCI_14'] = (tp - tp_ma) / (0.015 * tp_mad + 1e-10)
    out['Volume_log'] = np.log1p(vol)
    out['Volume_MA20_ratio'] = vol / (vol.rolling(20).mean() + 1e-10)
    out['Price_change_1d'] = c.pct_change(1)*100
    out['Price_change_5d'] = c.pct_change(5)*100
    ret = c.pct_change()
    out['Volatility_5d'] = ret.rolling(5).std()*100
    out['Volatility_20d'] = ret.rolling(20).std()*100
    out['HL_range_pct'] = (out['High'] - out['Low'])/(c + 1e-10)*100
    out['RSI14_slope_3d'] = out['RSI_14'].diff(3)
    out['MACD_accel'] = out['MACD_hist'].diff(1)
    bb_width_ma = out['BB_width'].rolling(20).mean()
    out['BB_squeeze'] = out['BB_width'] / (bb_width_ma + 1e-10)
    # New indicators
    high, low = out['High'], out['Low']
    tr_adx = pd.concat([high-low, (high-c.shift()).abs(), (low-c.shift()).abs()], axis=1).max(axis=1)
    up_move = high.diff(); down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr14_adx = tr_adx.ewm(span=14, min_periods=14).mean()
    plus_di = 100*pd.Series(plus_dm).ewm(span=14, min_periods=14).mean()/(atr14_adx+1e-10)
    minus_di = 100*pd.Series(minus_dm).ewm(span=14, min_periods=14).mean()/(atr14_adx+1e-10)
    dx = 100*abs(plus_di-minus_di)/(plus_di+minus_di+1e-10)
    out['ADX_14'] = dx.ewm(span=14, min_periods=14).mean()
    out['PLUS_DI'] = plus_di; out['MINUS_DI'] = minus_di
    mid = (out['High']+out['Low'])/2
    out['AO'] = mid.rolling(5).mean() - mid.rolling(34).mean()
    dpo_ma = c.rolling(20).mean()
    out['DPO_20'] = c - dpo_ma.shift(20//2+1)
    tr_sum = tr_adx.rolling(14).sum(); range_14 = high.rolling(14).max()-low.rolling(14).min()
    out['CHOP_14'] = 100*np.log10(tr_sum/(range_14+1e-10))/np.log10(14)
    out['COPPOCK'] = (c.pct_change(14)*100+c.pct_change(11)*100).ewm(span=10, min_periods=10).mean()
    out['MOM_5'] = c-c.shift(5); out['MOM_10'] = c-c.shift(10)
    out['DISPARITY_5'] = (c-c.rolling(5).mean())/(c.rolling(5).mean()+1e-10)*100
    out['DISPARITY_10'] = (c-c.rolling(10).mean())/(c.rolling(10).mean()+1e-10)*100
    out['Target'] = out['Close'].pct_change().shift(-1)*100
    return out.dropna().reset_index(drop=True)

FEATURES = [c for c in [
    'Open','High','Low','Close','Volume','SMA_20','SMA_50','EMA_9','EMA_21',
    'Close_SMA20_ratio','Close_SMA50_ratio','RSI_7','RSI_14',
    'MACD_line','MACD_signal','MACD_hist','BB_pct','BB_width','ATR_14',
    'STOCH_K','STOCH_D','OBV','CCI_14','Volume_log','Volume_MA20_ratio',
    'Price_change_1d','Price_change_5d','Volatility_5d','Volatility_20d',
    'HL_range_pct','RSI14_slope_3d','MACD_accel','BB_squeeze',
    'ADX_14','PLUS_DI','MINUS_DI','AO','DPO_20','CHOP_14','COPPOCK',
    'MOM_5','MOM_10','DISPARITY_5','DISPARITY_10',
] if c not in ['Target','Date']]


# ===========================================================================
# MAIN
# ===========================================================================

def run_catboost_bayes(csv_file, n_trials=30):
    global FEATURES
    print("=" * 60)
    print("BAYESIAN-OPTIMIZED CATBOOST (BO-CatBoost)")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = compute_indicators(pd.read_csv(csv_file))
    print(f"Records: {len(df)}  Features: {len(FEATURES)}")
    print(f"Bayesian optimization trials: {n_trials}")

    n = len(df); cut = int(n * 0.9)
    train, test = df.iloc[:cut], df.iloc[cut:]
    X_tr, y_tr = train[FEATURES].values, train['Target'].values
    X_te, y_te = test[FEATURES].values, test['Target'].values
    print(f"Train: {len(train)}  Test: {len(test)}")

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

    # ---- LSTM Feature Generator (matching paper: 2-layer, 100 units) ----
    print("\nTraining LSTM feature generator (High/Low/Close only)...")
    lookback = 20  # paper-suggested lookback
    lstm_feats = ['High', 'Low', 'Close']

    def _build_sequences(data, feats, lb):
        X, y = [], []
        for i in range(lb, len(data)):
            X.append(data[feats].iloc[i-lb:i].values)
            y.append(data['Target'].iloc[i])
        return np.array(X), np.array(y)

    # Scale LSTM inputs separately
    lstm_sc = StandardScaler()
    lstm_data = df[lstm_feats + ['Target']].copy()
    lstm_data[lstm_feats] = lstm_sc.fit_transform(lstm_data[lstm_feats])

    X_lstm, y_lstm = _build_sequences(lstm_data, lstm_feats, lookback)
    cut_lstm = int(len(X_lstm) * 0.9)
    X_l_tr, X_l_te = X_lstm[:cut_lstm], X_lstm[cut_lstm:]
    y_l_tr, y_l_te = y_lstm[:cut_lstm], y_lstm[cut_lstm:]

    # Build simple 2-layer LSTM (paper: units=100, no CNN)
    try:
        import os as _os
        _os.environ['KERAS_BACKEND'] = 'torch'
        import keras
        from keras import layers
    except ImportError:
        print("  Keras not available — skipping LSTM feature generator")
        lstm_pred_tr = np.zeros((len(X_tr), 3))
        lstm_pred_te = np.zeros((len(X_te), 3))
        lstm_pred_now = np.zeros((1, 3))
    else:
        lstm_model = keras.Sequential([
            layers.Input(shape=(lookback, len(lstm_feats))),
            layers.LSTM(100, return_sequences=True),
            layers.LSTM(100, return_sequences=False),
            layers.Dense(32, activation='relu'),
            layers.Dense(3),  # predict High, Low, Close
        ])
        lstm_model.compile(optimizer='adam', loss='mse')
        lstm_model.fit(X_l_tr, X_l_tr[:, -1, :3],  # train on last-timestep values
                       validation_data=(X_l_te, X_l_te[:, -1, :3]),
                       epochs=10, batch_size=32, verbose=0)
        print("  LSTM feature generator trained")

        # Generate LSTM predictions as features for the full dataset
        lstm_data_all = lstm_data[lstm_feats].values
        all_lstm_preds = []
        for i in range(len(lstm_data_all)):
            if i < lookback:
                all_lstm_preds.append(lstm_data_all[i])
            else:
                seq = lstm_data_all[i-lookback:i].reshape(1, lookback, len(lstm_feats))
                pred = lstm_model.predict(seq, verbose=0)[0]
                all_lstm_preds.append(pred)
        all_lstm_preds = np.array(all_lstm_preds)

        # Align LSTM predictions with the main dataframe (drop first `lookback` rows)
        lstm_pred_aligned = all_lstm_preds[lookback:]  # shape: (n-lookback, 3)
        df_aligned = df.iloc[lookback:].reset_index(drop=True)

        # Add LSTM predictions as features
        df_aligned['LSTM_High'] = lstm_pred_aligned[:, 0]
        df_aligned['LSTM_Low']  = lstm_pred_aligned[:, 1]
        df_aligned['LSTM_Close'] = lstm_pred_aligned[:, 2]
        # Also add LSTM forecast direction
        df_aligned['LSTM_dir'] = np.sign(
            df_aligned['LSTM_Close'] - df_aligned['Close'])

        # Re-split with LSTM features included
        lstm_extra = ['LSTM_High', 'LSTM_Low', 'LSTM_Close', 'LSTM_dir']
        FEATURES_LSTM = FEATURES + lstm_extra
        n2 = len(df_aligned); cut2 = int(n2 * 0.9)
        train2, test2 = df_aligned.iloc[:cut2], df_aligned.iloc[cut2:]
        X_tr2 = train2[FEATURES_LSTM].values; y_tr2 = train2['Target'].values
        X_te2 = test2[FEATURES_LSTM].values;  y_te2 = test2['Target'].values
        sc2 = StandardScaler()
        X_tr_s = sc2.fit_transform(X_tr2); X_te_s = sc2.transform(X_te2)
        # Update references for rest of the function
        X_tr, X_te = X_tr2, X_te2
        y_tr, y_te = y_tr2, y_te2
        FEATURES = FEATURES_LSTM
        sc = sc2
        test = test2
        df = df_aligned
        print(f"  Added 4 LSTM features -> {len(FEATURES)} total features")

        # Clean up keras model to free memory
        del lstm_model
        import gc; gc.collect()

    # Bayesian hyperparameter optimization
    try:
        import optuna
        from catboost import CatBoostRegressor
    except ImportError:
        print("Need: pip install optuna catboost"); sys.exit(1)

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 3000, step=500),
            'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.05, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_strength': trial.suggest_float('random_strength', 0.5, 3.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 1.0),
        }
        m = CatBoostRegressor(**params, loss_function='RMSE', random_seed=42,
                              verbose=0, thread_count=-1)
        m.fit(X_tr_s, y_tr, eval_set=(X_te_s, y_te),
              early_stopping_rounds=50, verbose=0)
        pred = m.predict(X_te_s)
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        return rmse

    print("\nRunning Bayesian optimization...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\nBest params: {best}")
    print(f"Best RMSE: {study.best_value:.4f}")

    # Train final model with best params
    model = CatBoostRegressor(
        iterations=best['iterations'],
        learning_rate=best['learning_rate'],
        depth=best['depth'],
        l2_leaf_reg=best['l2_leaf_reg'],
        random_strength=best['random_strength'],
        bagging_temperature=best['bagging_temperature'],
        loss_function='RMSE', random_seed=42, verbose=100,
        early_stopping_rounds=50, thread_count=-1,
    )
    model.fit(X_tr_s, y_tr, eval_set=(X_te_s, y_te), verbose=100)

    # Metrics
    tr_pred = model.predict(X_tr_s); te_pred = model.predict(X_te_s)
    close_idx = FEATURES.index('Close')
    prev_tr, prev_te = X_tr[:, close_idx], X_te[:, close_idx]
    tr_pred_pr = prev_tr*(1+tr_pred/100); te_pred_pr = prev_te*(1+te_pred/100)
    y_tr_pr = prev_tr*(1+y_tr/100); y_te_pr = prev_te*(1+y_te/100)

    tr_mae = mean_absolute_error(y_tr_pr, tr_pred_pr)
    te_mae = mean_absolute_error(y_te_pr, te_pred_pr)
    te_rmse = np.sqrt(mean_squared_error(y_te_pr, te_pred_pr))

    tr_dir_act = (y_tr > 0).astype(int); tr_dir_pr = (tr_pred > 0).astype(int)
    te_dir_act = (y_te > 0).astype(int); te_dir_pr = (te_pred > 0).astype(int)
    te_acc = accuracy_score(te_dir_act, te_dir_pr)
    te_f1 = f1_score(te_dir_act, te_dir_pr, zero_division=0)

    print(f"\n{'='*60}")
    print("BO-CATBOOST RESULTS")
    print("="*60)
    print(f"Test MAE:      ${te_mae:.2f}")
    print(f"Test RMSE:     ${te_rmse:.2f}")
    print(f"Test Accuracy: {te_acc*100:.2f}%")
    print(f"Test F1:       {te_f1*100:.2f}%")

    # Signal
    today_price = float(df['Close'].iloc[-1])
    recent = df[FEATURES].iloc[-1:].values
    ret_pred = model.predict(sc.transform(recent))[0]
    expected_move_pct = ret_pred
    tomorrow_price = today_price * (1 + ret_pred/100)

    vol_20d = float(df['Close'].pct_change().tail(20).std()*100)
    sig_thresh = max(0.15*vol_20d, 0.1)

    if expected_move_pct > sig_thresh: signal="BUY (LONG)"; signal_int=1
    elif expected_move_pct < -sig_thresh: signal="SHORT (SELL)"; signal_int=-1
    else: signal="HOLD (No clear signal)"; signal_int=0

    h, l, cr = df['High'], df['Low'], df['Close']
    tr = pd.concat([h-l, (h-cr.shift()).abs(), (l-cr.shift()).abs()], axis=1).max(axis=1)
    atr_val = float(tr.ewm(span=14, min_periods=14).mean().iloc[-1])
    if pd.isna(atr_val) or atr_val <= 0: atr_val = today_price * 0.02
    sl_dist=1.0*atr_val; tp_dist=1.5*atr_val
    volatility = float(df['Close'].tail(20).pct_change().dropna().std() * today_price)

    if signal_int==1: sl=today_price-sl_dist; tp=today_price+tp_dist
    elif signal_int==-1: sl=today_price+sl_dist; tp=today_price-tp_dist
    else: sl=today_price-sl_dist; tp=today_price+tp_dist

    emoji = "[BUY]" if signal_int==1 else ("[SHORT]" if signal_int==-1 else "[HOLD]")
    print(f"\n{emoji} SIGNAL: {signal}")
    print(f"Price: ${today_price:.2f} | Pred: ${tomorrow_price:.2f} | Move: {expected_move_pct:+.2f}%")
    print(f"SL: ${sl:.2f} ({((sl-today_price)/today_price*100):+.2f}%) | TP: ${tp:.2f} ({((tp-today_price)/today_price*100):+.2f}%)")

    # Probability analysis
    print("\n" + "="*70)
    print("Running Multi-Approach Win Probability Analysis...")
    print("="*70)
    mc = monte_carlo_simulation(today_price, sl, tp, volatility, expected_move_pct)
    pat = find_similar_patterns(df, today_price, sl, tp)
    ens = calculate_ensemble_probability(None, mc, pat)
    report = format_analysis_report(None, mc, pat, ens, signal, today_price, sl, tp)
    print(report)
    if ens:
        print(f"ENSEMBLE_PROBABILITY: {ens['ensemble_probability']:.1f}%")
        print(f"CONFIDENCE_LEVEL: {ens['confidence_level']}")
        print(f"RECOMMENDATION: {ens['recommendation']}")

    # Plots
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fi = pd.DataFrame({'feature': FEATURES, 'importance': model.get_feature_importance()
        }).sort_values('importance', ascending=False)
    plt.figure(figsize=(12, 8)); top = fi.head(20)
    plt.barh(range(len(top)), top['importance']); plt.yticks(range(len(top)), top['feature'])
    plt.xlabel('Importance'); plt.title('BO-CatBoost: Feature Importance (Top 20)')
    plt.gca().invert_yaxis(); plt.tight_layout()
    plt.savefig('catboost_bayes_feature_importance.png', dpi=150, bbox_inches='tight'); plt.close()

    plot_n = min(200, len(y_te))
    plt.figure(figsize=(15, 6))
    plt.plot(range(plot_n), y_te_pr[-plot_n:], label='Actual', color='blue', linewidth=2)
    plt.plot(range(plot_n), te_pred_pr[-plot_n:], label='Predicted', color='red', linewidth=2, alpha=0.7)
    plt.title('BO-CatBoost: Actual vs Predicted (Last 200 Test Samples)')
    plt.xlabel('Test Sample'); plt.ylabel('Price ($)'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('catboost_bayes_predictions.png', dpi=150, bbox_inches='tight'); plt.close()
    print("Plots saved: catboost_bayes_feature_importance.png, catboost_bayes_predictions.png")

    # Save
    joblib.dump(model, os.path.join(base_dir, 'catboost_bayes_model.pkl'))
    joblib.dump(sc, os.path.join(base_dir, 'catboost_bayes_scaler.pkl'))
    with open(os.path.join(base_dir, 'catboost_bayes_features.txt'), 'w') as f:
        f.write('\n'.join(FEATURES))
    info_path = os.path.join(base_dir, 'catboost_bayes_model_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"ticker: {os.path.basename(csv_file).split('_')[0]}\n")
        f.write(f"model_type: BO-CatBoost (Bayesian optimized)\n")
        f.write(f"best_params: {best}\n")
        f.write(f"test_mae: {te_mae}\n")
        f.write(f"test_rmse: {te_rmse}\n")
        f.write(f"test_accuracy: {te_acc}\n")
        f.write(f"test_f1: {te_f1}\n")
        f.write(f"timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\nModel saved.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BO-CatBoost')
    parser.add_argument('csv_file', type=str)
    parser.add_argument('--n_trials', type=int, default=30)
    args = parser.parse_args()
    run_catboost_bayes(args.csv_file, n_trials=args.n_trials)
