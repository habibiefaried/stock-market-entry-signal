"""
LightGBM Heavy - Rich Feature Set + Deep Ensemble

Differences from train_lightgbm.py (standard):
  Features  : 5 OHLCV -> 70+ (all 52 technical indicators + lag features)
  Trees     : 1,000  -> 3,000
  Learn rate: 0.01   -> 0.005  (slower, more precise convergence)
  num_leaves: 31     -> 63     (deeper trees, more expressive splits)
  Regularisation: adds min_data_in_leaf, bagging_fraction, feature_fraction,
                  lambda_l1, lambda_l2

Why more features help LightGBM:
  LightGBM uses leaf-wise tree growth - it always splits the leaf with the
  highest delta loss. With 70+ features it can find high-order interactions
  like "RSI overbought AND BB squeeze tightening AND OBV diverging -> reversal"
  that raw OHLCV lags simply cannot capture.

Why more trees + lower LR:
  Each tree corrects smaller residuals -> smoother, less overfit function.
  Rule of thumb: halve LR, double n_estimators.
  LightGBM's GOSS (gradient-based one-side sampling) makes it especially
  efficient with large n_estimators.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             accuracy_score, precision_score, recall_score, f1_score)
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import joblib
from datetime import datetime
import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('lightgbm').setLevel(logging.ERROR)

from trade_probability_analyzer import (
    predict_multi_day_path,
    monte_carlo_simulation,
    find_similar_patterns,
    calculate_ensemble_probability,
    format_analysis_report
)


# ============================================================================
# TECHNICAL INDICATOR COMPUTATION  (same 52 as CNN-LSTM / XGBoost-Heavy)
# ============================================================================

def _rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-10)))


def _atr(df, period):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()


def _cci(df, period):
    tp  = (df['High'] + df['Low'] + df['Close']) / 3
    ma  = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * mad + 1e-10)


def compute_technical_indicators(df):
    out = df.copy()
    c   = out['Close']

    # Moving averages: 2 SMA + 2 EMA (short and medium-term only)
    out['SMA_20'] = c.rolling(20).mean()
    out['SMA_50'] = c.rolling(50).mean()
    out['EMA_9']  = c.ewm(span=9,  min_periods=9).mean()
    out['EMA_21'] = c.ewm(span=21, min_periods=21).mean()

    # Normalised distance from MA (dimensionless, avoids price-scale leakage)
    out['Close_SMA20_ratio'] = (c - out['SMA_20']) / (out['SMA_20'] + 1e-10)
    out['Close_SMA50_ratio'] = (c - out['SMA_50']) / (out['SMA_50'] + 1e-10)

    # RSI: 2 periods (short + standard)
    out['RSI_7']  = _rsi(c, 7)
    out['RSI_14'] = _rsi(c, 14)

    # MACD: all 3 components carry distinct info (level, signal, momentum)
    ema12 = c.ewm(span=12, min_periods=12).mean()
    ema26 = c.ewm(span=26, min_periods=26).mean()
    out['MACD_line']   = ema12 - ema26
    out['MACD_signal'] = out['MACD_line'].ewm(span=9, min_periods=9).mean()
    out['MACD_hist']   = out['MACD_line'] - out['MACD_signal']

    # Bollinger Bands: normalised only (raw upper/lower/mid are redundant with Close)
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_lo  = bb_mid - 2 * bb_std
    out['BB_pct']   = (c - bb_lo) / (bb_up - bb_lo + 1e-10)
    out['BB_width'] = (bb_up - bb_lo) / (bb_mid + 1e-10)

    # ATR: single standard period
    out['ATR_14'] = _atr(out, 14)

    # Stochastic: both K and D (momentum + smoothed)
    low_min  = df['Low'].rolling(14).min()
    high_max = df['High'].rolling(14).max()
    k = 100 * (c - low_min) / (high_max - low_min + 1e-10)
    out['STOCH_K'] = k
    out['STOCH_D'] = k.rolling(3).mean()

    # OBV: log-scaled cumulative volume pressure
    direction  = np.sign(c.diff()).fillna(0)
    obv_raw    = (direction * out['Volume']).cumsum()
    out['OBV'] = np.log1p(obv_raw.abs()) * np.sign(obv_raw)

    # CCI: single standard period (WILLR_14 dropped -- same concept as STOCH)
    out['CCI_14'] = _cci(out, 14)

    # Volume features
    vol = out['Volume']
    out['Volume_log']        = np.log1p(vol)
    out['Volume_MA20_ratio'] = vol / (vol.rolling(20).mean() + 1e-10)

    # Price changes: 2 distinct horizons (ROC dropped -- identical to pct_change)
    out['Price_change_1d'] = c.pct_change(1) * 100
    out['Price_change_5d'] = c.pct_change(5) * 100

    # Volatility: 2 horizons (short + medium)
    ret = c.pct_change()
    out['Volatility_5d']  = ret.rolling(5).std()  * 100
    out['Volatility_20d'] = ret.rolling(20).std() * 100

    # Daily range normalised
    out['HL_range_pct'] = (out['High'] - out['Low']) / (c + 1e-10) * 100

    return out


# 30 indicator columns (max 2 per indicator family to limit multicollinearity)
INDICATOR_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_20', 'SMA_50',
    'EMA_9', 'EMA_21',
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
]


# ============================================================================
# DATA LOADING & FEATURE ENGINEERING
# ============================================================================

def load_and_prepare_data(csv_file):
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("Computing technical indicators (reduced set)...")
    df = compute_technical_indicators(df)
    df = df.dropna(subset=INDICATOR_COLS).reset_index(drop=True)

    print(f"Records after indicator computation: {len(df)}")
    return df


def create_rich_features(df, lags=[1, 3, 5]):
    """
    Build the heavy feature set (reduced to ~42 features):
      30 technical indicators (max 2 per family)
    + Close lags x3 (1, 3, 5 days)
    + Volume lags x2 (1, 5 days)
    + RSI14 slope over 3 days (momentum-of-momentum)
    + MACD acceleration (histogram first difference)
    + BB squeeze (BB_width vs its 20-day mean)
    """
    keep = ['Date'] + INDICATOR_COLS if 'Date' in df.columns else INDICATOR_COLS
    out  = df[keep].copy()

    # Close lags: 3 periods (reduced from 5)
    for lag in lags:
        out[f'Close_lag_{lag}'] = out['Close'].shift(lag)

    # Volume lags: 2 periods only
    out['Volume_lag_1'] = out['Volume'].shift(1)
    out['Volume_lag_5'] = out['Volume'].shift(5)

    # Derived momentum features
    out['RSI14_slope_3d'] = out['RSI_14'].diff(3)
    out['MACD_accel']     = out['MACD_hist'].diff(1)
    bb_width_ma           = out['BB_width'].rolling(20).mean()
    out['BB_squeeze']     = out['BB_width'] / (bb_width_ma + 1e-10)

    # Target
    out['Target'] = out['Close'].shift(-1)
    out = out.dropna()

    all_features = [c for c in out.columns if c not in ['Date', 'Target']]
    print(f"Total features (heavy): {len(all_features)}")
    return out, all_features


def split_train_test(df, train_ratio=9/10):
    idx = int(len(df) * train_ratio)
    train_df, test_df = df[:idx], df[idx:]
    print(f"\nData split:")
    print(f"  Train: {len(train_df)} records  ({train_df['Date'].min()} to {train_df['Date'].max()})")
    print(f"  Test:  {len(test_df)} records  ({test_df['Date'].min()} to {test_df['Date'].max()})")
    return train_df, test_df


def calculate_direction_metrics(y_true, y_pred):
    yt = (np.diff(y_true) > 0).astype(int)
    yp = (np.diff(y_pred) > 0).astype(int)
    return (accuracy_score(yt, yp),
            precision_score(yt, yp, zero_division=0),
            recall_score(yt, yp, zero_division=0),
            f1_score(yt, yp, zero_division=0))


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_lightgbm_heavy_model(
    csv_file,
    n_estimators=3000,
    learning_rate=0.005,
    num_leaves=63,
    min_data_in_leaf=20,
    bagging_fraction=0.8,
    feature_fraction=0.7,
    lambda_l1=0.05,
    lambda_l2=1.0,
):
    df = load_and_prepare_data(csv_file)
    df_feat, all_features = create_rich_features(df)
    train_df, test_df     = split_train_test(df_feat)

    X_train = train_df[all_features].values
    y_train = train_df['Target'].values
    X_test  = test_df[all_features].values
    y_test  = test_df['Target'].values

    print(f"\nFeature matrix: X_train={X_train.shape}  X_test={X_test.shape}")

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print("\nBuilding LightGBM-Heavy model...")

    _using_gpu = False

    def _make_model(use_gpu):
        params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_data_in_leaf=min_data_in_leaf,
            bagging_fraction=bagging_fraction,
            bagging_freq=5,
            feature_fraction=feature_fraction,
            reg_alpha=lambda_l1,
            reg_lambda=lambda_l2,
            random_state=42,
            verbose=-1,
        )
        if use_gpu:
            params['device'] = 'gpu'
        return lgb.LGBMRegressor(**params)

    print(f"\nTraining LightGBM-Heavy ({n_estimators} trees, lr={learning_rate})...")
    print("This may take 5-20 minutes on CPU, 2-5 minutes on GPU")
    try:
        model = _make_model(use_gpu=True)
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
            eval_names=['train', 'test'],
            callbacks=[lgb.log_evaluation(period=100)],
        )
        _using_gpu = True
        print("Using GPU acceleration (CUDA)")
    except Exception as e:
        print(f"GPU not available, falling back to CPU: {e}")
        model = _make_model(use_gpu=False)
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
            eval_names=['train', 'test'],
            callbacks=[lgb.log_evaluation(period=100)],
        )

    print("\nMaking predictions...")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred  = model.predict(X_test_scaled)

    train_mae  = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_mae   = mean_absolute_error(y_test, y_test_pred)
    test_rmse  = np.sqrt(mean_squared_error(y_test, y_test_pred))

    train_acc, train_prec, train_rec, train_f1 = calculate_direction_metrics(y_train, y_train_pred)
    test_acc,  test_prec,  test_rec,  test_f1  = calculate_direction_metrics(y_test, y_test_pred)

    print("\n" + "="*60)
    print("LIGHTGBM-HEAVY MODEL EVALUATION RESULTS")
    print("="*60)
    print("\nREGRESSION METRICS (Price Prediction):")
    print(f"  Training MAE:   ${train_mae:.2f}")
    print(f"  Training RMSE:  ${train_rmse:.2f}")
    print(f"  Test MAE:       ${test_mae:.2f}")
    print(f"  Test RMSE:      ${test_rmse:.2f}")
    print("\nCLASSIFICATION METRICS (Direction Prediction Up/Down):")
    print(f"  Training Accuracy:  {train_acc*100:.2f}%")
    print(f"  Training Precision: {train_prec*100:.2f}%")
    print(f"  Training Recall:    {train_rec*100:.2f}%")
    print(f"  Training F1-Score:  {train_f1*100:.2f}%")
    print(f"\n  Test Accuracy:      {test_acc*100:.2f}%")
    print(f"  Test Precision:     {test_prec*100:.2f}%")
    print(f"  Test Recall:        {test_rec*100:.2f}%")
    print(f"  Test F1-Score:      {test_f1*100:.2f}%")
    print("\n" + "="*60)

    fi = pd.DataFrame({
        'feature':    all_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 15 Most Important Features:")
    print(fi.head(15).to_string(index=False))

    plt.figure(figsize=(12, 8))
    top = fi.head(20)
    plt.barh(range(len(top)), top['importance'])
    plt.yticks(range(len(top)), top['feature'])
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importances (LightGBM-Heavy)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('lightgbm_heavy_feature_importance.png')
    print("\nFeature importance plot saved as: lightgbm_heavy_feature_importance.png")

    plt.figure(figsize=(15, 6))
    test_dates = test_df['Date'].values
    plt.plot(test_dates, y_test,      label='Actual Price',    color='blue', linewidth=2)
    plt.plot(test_dates, y_test_pred, label='Predicted Price', color='red',  linewidth=2, alpha=0.7)
    plt.title('LightGBM-Heavy: Actual vs Predicted Prices (Test Set)')
    plt.xlabel('Date'); plt.ylabel('Price')
    plt.legend(); plt.xticks(rotation=45); plt.grid(True); plt.tight_layout()
    plt.savefig('lightgbm_heavy_predictions.png')
    print("Predictions plot saved as: lightgbm_heavy_predictions.png")

    # ---- Trading signal ----
    print("\n" + "="*60)
    print("TRADING SIGNAL FOR NEXT DAY")
    print("="*60)

    today_price     = df['Close'].iloc[-1]
    recent_features = df_feat[all_features].iloc[-1:].values
    recent_scaled   = scaler.transform(recent_features)
    tomorrow_pred   = model.predict(recent_scaled)[0]

    expected_move     = tomorrow_pred - today_price
    expected_move_pct = (expected_move / today_price) * 100

    vol_20d_pct    = df['Volatility_20d'].iloc[-1]
    sig_threshold  = max(0.3 * vol_20d_pct, 0.3)

    if expected_move_pct > sig_threshold:
        signal       = "BUY (LONG)"
        signal_emoji = "[BUY]"
    elif expected_move_pct < -sig_threshold:
        signal       = "SHORT (SELL)"
        signal_emoji = "[SHORT]"
    else:
        signal       = "HOLD (No clear signal)"
        signal_emoji = "[HOLD]"

    atr = float(df['ATR_14'].iloc[-1])
    stop_loss_distance   = 1.0 * atr
    take_profit_distance = 1.5 * atr
    volatility           = df[['Close']].tail(20)['Close'].pct_change().dropna().std() * today_price

    if signal == "BUY (LONG)":
        stop_loss   = today_price - stop_loss_distance
        take_profit = today_price + take_profit_distance
    elif signal == "SHORT (SELL)":
        stop_loss   = today_price + stop_loss_distance
        take_profit = today_price - take_profit_distance
    else:
        stop_loss   = today_price - stop_loss_distance
        take_profit = today_price + take_profit_distance

    confidence = test_acc * 100

    print(f"\n{signal_emoji} SIGNAL: {signal}")
    print(f"\nCurrent Price (Today):      ${today_price:.2f}")
    print(f"Predicted Price (Tomorrow): ${tomorrow_pred:.2f}")
    print(f"Expected Move:              ${expected_move:+.2f} ({expected_move_pct:+.2f}%)")
    print(f"\nRisk Management (Stock Price Levels):")
    print(f"  Stop Loss:    ${stop_loss:.2f} ({((stop_loss - today_price) / today_price * 100):+.2f}%)")
    print(f"  Take Profit:  ${take_profit:.2f} ({((take_profit - today_price) / today_price * 100):+.2f}%)")
    print(f"\n5x Leverage Position P&L (for IQ Option auto-close):")
    print(f"  Stop Loss %:   {((stop_loss - today_price) / today_price * 100 * 5):+.1f}%")
    print(f"  Take Profit %: {((take_profit - today_price) / today_price * 100 * 5):+.1f}%")
    print(f"  Risk/Reward:   1.67:1")
    print(f"\nModel Confidence: {confidence:.1f}% (based on test accuracy)")
    print(f"Recent Volatility: ${volatility:.2f} per day")

    # ---- Probability analysis ----
    print("\n" + "="*70)
    print("Running Multi-Approach Win Probability Analysis...")
    print("="*70)

    print("\n[1/3] Running multi-day sequential prediction...")
    prediction_result = predict_multi_day_path(
        model=model,
        scaler=scaler,
        df=df_feat,
        feature_cols=all_features,
        current_price=today_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        model_type='gbm',
    )

    print("[2/3] Running Monte Carlo simulation...")
    monte_carlo_result = monte_carlo_simulation(
        current_price=today_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        volatility=volatility,
        predicted_move_pct=expected_move_pct,
    )

    print("[3/3] Searching historical patterns...")
    pattern_result = find_similar_patterns(
        df=df,
        current_price=today_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )

    ensemble_result = calculate_ensemble_probability(
        prediction_result=prediction_result,
        monte_carlo_result=monte_carlo_result,
        pattern_result=pattern_result,
    )

    analysis_report = format_analysis_report(
        prediction_result=prediction_result,
        monte_carlo_result=monte_carlo_result,
        pattern_result=pattern_result,
        ensemble_result=ensemble_result,
        signal=signal,
        current_price=today_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )
    print(analysis_report)

    print("\n" + "="*70)
    print("PROBABILITY_ANALYSIS_RESULTS:")
    print(f"ENSEMBLE_PROBABILITY: {ensemble_result['ensemble_probability']:.1f}%" if ensemble_result else "ENSEMBLE_PROBABILITY: N/A")
    print(f"CONFIDENCE_LEVEL: {ensemble_result['confidence_level']}" if ensemble_result else "CONFIDENCE_LEVEL: N/A")
    print(f"RECOMMENDATION: {ensemble_result['recommendation']}" if ensemble_result else "RECOMMENDATION: N/A")
    print("="*70)

    print("\n" + "="*60)
    print("DISCLAIMER:")
    print("This is a statistical prediction, NOT financial advice.")
    print("Past performance does not guarantee future results.")
    print("Always do your own research and manage risk appropriately.")
    print("="*60)

    # Save
    joblib.dump(model,  'lightgbm_heavy_model.pkl')
    joblib.dump(scaler, 'lightgbm_heavy_scaler.pkl')
    with open('lightgbm_heavy_features.txt', 'w') as f:
        f.write('\n'.join(all_features))

    ticker = os.path.basename(csv_file).split('_')[0]
    model_info = {
        'ticker':         ticker,
        'model_type':     f'LightGBM-Heavy ({"GPU" if _using_gpu else "CPU"})',
        'n_features':     len(all_features),
        'n_estimators':   n_estimators,
        'learning_rate':  learning_rate,
        'train_size':     len(X_train),
        'test_size':      len(X_test),
        'test_mae':       test_mae,
        'test_rmse':      test_rmse,
        'test_accuracy':  test_acc,
        'test_precision': test_prec,
        'test_recall':    test_rec,
        'test_f1':        test_f1,
        'timestamp':      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open('lightgbm_heavy_model_info.txt', 'w') as f:
        for k, v in model_info.items():
            f.write(f"{k}: {v}\n")

    print("\nModel saved as: lightgbm_heavy_model.pkl")
    print("Scaler saved as: lightgbm_heavy_scaler.pkl")
    return model, scaler, model_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train LightGBM-Heavy model (70+ features, 3000 trees)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python train_lightgbm_heavy.py MSFT_daily_data_20260520.csv
  python train_lightgbm_heavy.py MSFT_daily_data_20260520.csv --n_estimators 5000 --learning_rate 0.003
        '''
    )
    parser.add_argument('csv_file',            type=str)
    parser.add_argument('--n_estimators',      type=int,   default=3000)
    parser.add_argument('--learning_rate',     type=float, default=0.005)
    parser.add_argument('--num_leaves',        type=int,   default=63)
    parser.add_argument('--min_data_in_leaf',  type=int,   default=20)
    parser.add_argument('--bagging_fraction',  type=float, default=0.8)
    parser.add_argument('--feature_fraction',  type=float, default=0.7)
    parser.add_argument('--lambda_l1',         type=float, default=0.05)
    parser.add_argument('--lambda_l2',         type=float, default=1.0)

    args = parser.parse_args()
    if not os.path.exists(args.csv_file):
        print(f"Error: File {args.csv_file} not found!")
        exit(1)

    train_lightgbm_heavy_model(
        args.csv_file,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_data_in_leaf=args.min_data_in_leaf,
        bagging_fraction=args.bagging_fraction,
        feature_fraction=args.feature_fraction,
        lambda_l1=args.lambda_l1,
        lambda_l2=args.lambda_l2,
    )
