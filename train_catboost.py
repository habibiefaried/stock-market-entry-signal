"""
CatBoost Model — third GBM alongside XGBoost and LightGBM.

CatBoost (Categorical Boosting) uses ordered boosting and symmetric trees.
It often outperforms XGBoost/LightGBM on financial data with default parameters
and handles missing values natively without imputation.

Uses the same 38-feature set as the heavy models.

Usage:
    python train_catboost.py MSFT_daily_data_20260520.csv
"""

import argparse, os, sys, warnings, logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trade_probability_analyzer import (
    predict_multi_day_path,
    monte_carlo_simulation,
    find_similar_patterns,
    calculate_ensemble_probability,
    format_analysis_report,
)

# ---------------------------------------------------------------------------
# INDICATORS (same 38-feature set as heavy models)
# ---------------------------------------------------------------------------

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
    out = df.copy()
    c = out['Close']
    vol = out['Volume']

    out['SMA_20'] = c.rolling(20).mean()
    out['SMA_50'] = c.rolling(50).mean()
    out['EMA_9']  = c.ewm(span=9,  min_periods=9).mean()
    out['EMA_21'] = c.ewm(span=21, min_periods=21).mean()

    out['Close_SMA20_ratio'] = (c - out['SMA_20']) / (out['SMA_20'] + 1e-10)
    out['Close_SMA50_ratio'] = (c - out['SMA_50']) / (out['SMA_50'] + 1e-10)

    out['RSI_7']  = _rsi(c, 7)
    out['RSI_14'] = _rsi(c, 14)

    ema12 = c.ewm(span=12, min_periods=12).mean()
    ema26 = c.ewm(span=26, min_periods=26).mean()
    out['MACD_line']   = ema12 - ema26
    out['MACD_signal'] = out['MACD_line'].ewm(span=9, min_periods=9).mean()
    out['MACD_hist']   = out['MACD_line'] - out['MACD_signal']

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_lo  = bb_mid - 2 * bb_std
    out['BB_pct']   = (c - bb_lo) / (bb_up - bb_lo + 1e-10)
    out['BB_width'] = (bb_up - bb_lo) / (bb_mid + 1e-10)

    out['ATR_14'] = _atr(out, 14)

    low14  = out['Low'].rolling(14).min()
    high14 = out['High'].rolling(14).max()
    k = 100 * (c - low14) / (high14 - low14 + 1e-10)
    out['STOCH_K'] = k
    out['STOCH_D'] = k.rolling(3).mean()

    direction = np.sign(c.diff()).fillna(0)
    obv_raw = (direction * vol).cumsum()
    out['OBV'] = np.log1p(obv_raw.abs()) * np.sign(obv_raw)

    tp = (out['High'] + out['Low'] + c) / 3
    tp_ma = tp.rolling(14).mean()
    tp_mad = tp.rolling(14).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    out['CCI_14'] = (tp - tp_ma) / (0.015 * tp_mad + 1e-10)

    out['Volume_log']        = np.log1p(vol)
    out['Volume_MA20_ratio'] = vol / (vol.rolling(20).mean() + 1e-10)

    out['Price_change_1d'] = c.pct_change(1) * 100
    out['Price_change_5d'] = c.pct_change(5) * 100

    ret = c.pct_change()
    out['Volatility_5d']  = ret.rolling(5).std()  * 100
    out['Volatility_20d'] = ret.rolling(20).std() * 100

    out['HL_range_pct'] = (out['High'] - out['Low']) / (c + 1e-10) * 100

    out['RSI14_slope_3d'] = out['RSI_14'].diff(3)
    out['MACD_accel']     = out['MACD_hist'].diff(1)
    bb_width_ma = out['BB_width'].rolling(20).mean()
    out['BB_squeeze'] = out['BB_width'] / (bb_width_ma + 1e-10)

    return out.dropna().reset_index(drop=True)


FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_20', 'SMA_50', 'EMA_9', 'EMA_21',
    'Close_SMA20_ratio', 'Close_SMA50_ratio',
    'RSI_7', 'RSI_14',
    'MACD_line', 'MACD_signal', 'MACD_hist',
    'BB_pct', 'BB_width',
    'ATR_14',
    'STOCH_K', 'STOCH_D',
    'OBV',
    'CCI_14',
    'Volume_log', 'Volume_MA20_ratio',
    'Price_change_1d', 'Price_change_5d',
    'Volatility_5d', 'Volatility_20d',
    'HL_range_pct',
    'RSI14_slope_3d', 'MACD_accel', 'BB_squeeze',
]


def create_lag_features(df, feature_cols):
    df_lagged = df.copy()
    for lag in [1, 3, 5]:
        df_lagged[f'Close_lag_{lag}'] = df['Close'].shift(lag)
    for lag in [1, 5]:
        df_lagged[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
    df_lagged['Target'] = df['Close'].pct_change(3).shift(-3) * 100  # 3-day forward return
    df_lagged = df_lagged.dropna().reset_index(drop=True)
    all_features = [c for c in df_lagged.columns
                    if c not in ['Date', 'Target']]
    return df_lagged, all_features


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_catboost(csv_file, n_estimators=2000, learning_rate=0.01, max_depth=6):
    print("=" * 60)
    print("CATBOOST MODEL")
    print("=" * 60)

    df = pd.read_csv(csv_file)
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    if any(c not in df.columns for c in required):
        raise ValueError("CSV missing OHLCV columns")

    df = compute_indicators(df)
    df_lagged, all_features = create_lag_features(df, FEATURE_COLS)
    print(f"Records: {len(df_lagged)}  Features: {len(all_features)}")

    # 90/10 split
    n = len(df_lagged)
    train_size = int(n * 0.9)
    train = df_lagged.iloc[:train_size]
    test  = df_lagged.iloc[train_size:]

    X_train, y_train = train[all_features].values, train['Target'].values
    X_test,  y_test  = test[all_features].values,  test['Target'].values

    print(f"Train: {len(train)}  Test: {len(test)}")

    # Scale
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Train CatBoost
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        print("CatBoost not installed. Install with: pip install catboost")
        sys.exit(1)

    model = CatBoostRegressor(
        iterations=n_estimators,
        learning_rate=learning_rate,
        depth=max_depth,
        loss_function='RMSE',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        thread_count=-1,
        l2_leaf_reg=3,
        random_strength=1,
        bagging_temperature=0.5,
    )

    model.fit(
        X_train_scaled, y_train,
        eval_set=(X_test_scaled, y_test),
        verbose=100,
    )

    # Predictions
    train_pred = model.predict(X_train_scaled)
    test_pred  = model.predict(X_test_scaled)

    train_mae = np.mean(np.abs(train_pred - y_train))
    train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
    test_mae = np.mean(np.abs(test_pred - y_test))
    test_rmse = np.sqrt(np.mean((test_pred - y_test) ** 2))

    # Direction accuracy
    prev_prices_train = X_train[:, list(all_features).index('Close')]
    prev_prices_test  = X_test[:, list(all_features).index('Close')]
    train_dir_pred = (train_pred > prev_prices_train).astype(int)
    train_dir_actual = (y_train > prev_prices_train).astype(int)
    test_dir_pred = (test_pred > prev_prices_test).astype(int)
    test_dir_actual = (y_test > prev_prices_test).astype(int)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    train_acc = accuracy_score(train_dir_actual, train_dir_pred)
    test_acc = accuracy_score(test_dir_actual, test_dir_pred)
    test_prec = precision_score(test_dir_actual, test_dir_pred, zero_division=0)
    test_rec = recall_score(test_dir_actual, test_dir_pred, zero_division=0)
    test_f1 = f1_score(test_dir_actual, test_dir_pred, zero_division=0)

    print(f"\n{'='*60}")
    print("CATBOOST MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nREGRESSION METRICS (Price Prediction):")
    print(f"Training MAE:  ${train_mae:.2f}")
    print(f"Training RMSE: ${train_rmse:.2f}")
    print(f"Test MAE:      ${test_mae:.2f}")
    print(f"Test RMSE:     ${test_rmse:.2f}")
    print(f"\nCLASSIFICATION METRICS (Direction Prediction):")
    print(f"Training Accuracy:  {train_acc*100:.2f}%")
    print(f"Test Accuracy:      {test_acc*100:.2f}%")
    print(f"Test Precision:     {test_prec*100:.2f}%")
    print(f"Test Recall:        {test_rec*100:.2f}%")
    print(f"Test F1-Score:      {test_f1*100:.2f}%")

    # --- Plots ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Feature importance
    fi = pd.DataFrame({
        'feature': all_features,
        'importance': model.get_feature_importance()
    }).sort_values('importance', ascending=False)
    plt.figure(figsize=(12, 8))
    top = fi.head(20)
    plt.barh(range(len(top)), top['importance'])
    plt.yticks(range(len(top)), top['feature'])
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importances (CatBoost)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('catboost_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Predictions plot
    plot_n = min(200, len(y_test))
    plt.figure(figsize=(15, 6))
    plt.plot(range(plot_n), y_test[-plot_n:], label='Actual', color='blue', linewidth=2)
    plt.plot(range(plot_n), test_pred[-plot_n:], label='Predicted', color='red', linewidth=2, alpha=0.7)
    plt.title('CatBoost: Actual vs Predicted Prices (Last 200 Test Samples)')
    plt.xlabel('Test Sample')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('catboost_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plots saved: catboost_feature_importance.png, catboost_predictions.png")

    # --- Trading signal ---
    today_price = float(df['Close'].iloc[-1])
    recent = df_lagged[all_features].iloc[-1:].values
    recent_scaled = scaler.transform(recent)
    tomorrow_return = model.predict(recent_scaled)[0]
    expected_move_pct = tomorrow_return  # already a percentage
    tomorrow_pred_price = today_price * (1 + tomorrow_return / 100)
    expected_move = tomorrow_pred_price - today_price

    vol_20d_pct = float(df['Close'].pct_change().tail(20).std() * 100)
    sig_threshold = max(0.15 * vol_20d_pct, 0.1)

    if expected_move_pct > sig_threshold:
        signal = "BUY (LONG)"; signal_int = 1
    elif expected_move_pct < -sig_threshold:
        signal = "SHORT (SELL)"; signal_int = -1
    else:
        signal = "HOLD (No clear signal)"; signal_int = 0

    h, l, c_raw = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c_raw.shift()).abs(), (l - c_raw.shift()).abs()], axis=1).max(axis=1)
    atr_val = float(tr.ewm(span=14, min_periods=14).mean().iloc[-1])
    if pd.isna(atr_val) or atr_val <= 0: atr_val = today_price * 0.02
    sl_dist = 1.5 * atr_val; tp_dist = 2.05 * atr_val
    volatility = float(df['Close'].tail(20).pct_change().dropna().std() * today_price)

    if signal_int == 1:
        stop_loss = today_price - sl_dist
        take_profit = today_price + tp_dist
    elif signal_int == -1:
        stop_loss = today_price + sl_dist
        take_profit = today_price - tp_dist
    else:
        stop_loss = today_price - sl_dist
        take_profit = today_price + tp_dist

    confidence = test_acc * 100

    emoji = "[BUY]" if signal_int == 1 else ("[SHORT]" if signal_int == -1 else "[HOLD]")
    print(f"\n{emoji} SIGNAL: {signal}")
    print(f"\nCurrent Price (Today):      ${today_price:.2f}")
    print(f"Predicted Price (3-Day): ${tomorrow_pred_price:.2f}")
    print(f"Expected Move:              ${expected_move:+.2f} ({expected_move_pct:+.2f}%)")
    print(f"\nRisk Management (Stock Price Levels):")
    print(f"  Stop Loss:    ${stop_loss:.2f} ({((stop_loss - today_price) / today_price * 100):+.2f}%)")
    print(f"  Take Profit:  ${take_profit:.2f} ({((take_profit - today_price) / today_price * 100):+.2f}%)")
    print(f"\n5x Leverage Position P&L (for IQ Option auto-close):")
    print(f"  Stop Loss %:   {((stop_loss - today_price) / today_price * 100 * 5):+.1f}%")
    print(f"  Take Profit %: {((take_profit - today_price) / today_price * 100 * 5):+.1f}%")
    print(f"  Risk/Reward:   1.33:1")
    print(f"\nModel Confidence: {confidence:.1f}%")

    # --- Probability analysis ---
    print("\n" + "="*70)
    print("Running Multi-Approach Win Probability Analysis...")
    print("="*70)

    prediction_result = predict_multi_day_path(
        model=model, scaler=scaler, df=df_lagged,
        feature_cols=all_features, current_price=today_price,
        stop_loss=stop_loss, take_profit=take_profit, model_type='gbm',
    )

    mc_result = monte_carlo_simulation(
        current_price=today_price, stop_loss=stop_loss,
        take_profit=take_profit, volatility=volatility,
        predicted_move_pct=expected_move_pct,
    )

    pattern_result = find_similar_patterns(
        df=df, current_price=today_price,
        stop_loss=stop_loss, take_profit=take_profit,
    )

    ensemble = calculate_ensemble_probability(
        prediction_result=prediction_result,
        monte_carlo_result=mc_result,
        pattern_result=pattern_result,
    )

    report = format_analysis_report(
        prediction_result=prediction_result,
        monte_carlo_result=mc_result,
        pattern_result=pattern_result,
        ensemble_result=ensemble,
        signal=signal, current_price=today_price,
        stop_loss=stop_loss, take_profit=take_profit,
    )
    print(report)

    if ensemble:
        print(f"ENSEMBLE_PROBABILITY: {ensemble['ensemble_probability']:.1f}%")
        print(f"CONFIDENCE_LEVEL: {ensemble['confidence_level']}")
        print(f"RECOMMENDATION: {ensemble['recommendation']}")

    # --- Save model ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    joblib.dump(model, os.path.join(base_dir, 'catboost_model.pkl'))
    joblib.dump(scaler, os.path.join(base_dir, 'catboost_scaler.pkl'))
    with open(os.path.join(base_dir, 'catboost_features.txt'), 'w') as f:
        f.write('\n'.join(all_features))
    with open(os.path.join(base_dir, 'catboost_model_info.txt'), 'w') as f:
        f.write(f"ticker: {os.path.basename(csv_file).split('_')[0]}\n")
        f.write(f"model_type: CatBoost\n")
        f.write(f"n_features: {len(all_features)}\n")
        f.write(f"n_estimators: {n_estimators}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"max_depth: {max_depth}\n")
        f.write(f"train_size: {len(train)}\n")
        f.write(f"test_size: {len(test)}\n")
        f.write(f"test_mae: {test_mae}\n")
        f.write(f"test_rmse: {test_rmse}\n")
        f.write(f"test_accuracy: {test_acc}\n")
        f.write(f"test_precision: {test_prec}\n")
        f.write(f"test_recall: {test_rec}\n")
        f.write(f"test_f1: {test_f1}\n")
        f.write(f"timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\nModel saved: catboost_model.pkl, catboost_scaler.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CatBoost Model')
    parser.add_argument('csv_file', type=str, help='Path to CSV file')
    parser.add_argument('--n_estimators',  type=int, default=2000)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--max_depth',     type=int, default=6)
    args = parser.parse_args()
    run_catboost(args.csv_file, args.n_estimators, args.learning_rate, args.max_depth)
