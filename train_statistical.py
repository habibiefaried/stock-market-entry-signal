"""
KNN Statistical Model — non-parametric similarity-based prediction.

Finds the k most similar historical days using normalized indicator features,
then predicts direction by weighted vote of what happened next. Complements
the tree models with a completely different inductive bias (similarity vs splitting).

Runs in seconds, no GPU needed, fully interpretable.

Usage:
    python train_statistical.py MSFT_daily_data_20260520.csv
    python train_statistical.py MSFT_daily_data_20260520.csv --k 75
"""

import argparse, os, sys, warnings, logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trade_probability_analyzer import (
    monte_carlo_simulation,
    find_similar_patterns,
    calculate_ensemble_probability,
    format_analysis_report,
)

# ---------------------------------------------------------------------------
# INDICATORS — same 38-feature set as heavy models
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


FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_20', 'SMA_50', 'EMA_9', 'EMA_21',
    'Close_SMA20_ratio', 'Close_SMA50_ratio',
    'RSI_7', 'RSI_14',
    'MACD_line', 'MACD_signal', 'MACD_hist',
    'BB_pct', 'BB_width',
    'ATR_14',
    'STOCH_K', 'STOCH_D',
    'OBV', 'CCI_14',
    'Volume_log', 'Volume_MA20_ratio',
    'Price_change_1d', 'Price_change_5d',
    'Volatility_5d', 'Volatility_20d',
    'HL_range_pct',
    'RSI14_slope_3d', 'MACD_accel', 'BB_squeeze',
]

# ---------------------------------------------------------------------------
# KNN PREDICTION
# ---------------------------------------------------------------------------

def knn_predict(df, features, k=50, lookback_ratio=0.7, top_n_features=20,
                calibrate=False):
    """
    KNN prediction with feature-weighted distance, temporal decay, dynamic k.
    Set calibrate=True for the final prediction to get calibrated confidence.
    """
    n = len(df)
    search_end = int(n * lookback_ratio)
    if search_end < 50:
        search_end = max(50, n // 2)

    k_eff = max(30, min(k, int(np.sqrt(search_end)), search_end - 2))

    # Feature selection + feature weights
    returns = df['Close'].pct_change().shift(-1) * 100
    if top_n_features and top_n_features < len(features):
        corrs = df[features].iloc[:search_end].corrwith(
            returns.iloc[:search_end]).abs().fillna(0)
        selected = corrs.nlargest(top_n_features).index.tolist()
        feature_weights = corrs[selected].values.copy()
        feature_weights = np.clip(feature_weights, 0.01, None)
        feature_weights /= feature_weights.sum()
    else:
        selected = list(features)
        feature_weights = np.ones(len(selected)) / len(selected)

    # Scale
    scaler = StandardScaler()
    X_all = scaler.fit_transform(df[selected].values)

    # Feature-weighted distance: multiply each dimension by sqrt(weight)
    # This makes high-correlation features dominate the distance calculation
    fw = np.sqrt(feature_weights * len(selected))  # scale so avg weight = 1.0
    X_all_weighted = X_all * fw[np.newaxis, :]
    current = X_all_weighted[-1:]
    X_hist = X_all_weighted[:search_end]

    nn = NearestNeighbors(n_neighbors=k_eff, metric='euclidean')
    nn.fit(X_hist)
    distances, indices = nn.kneighbors(current)

    # Distance threshold: if closest neighbor is too far (>2.5x median), skip
    median_dist = np.median(distances[0])
    if distances[0][0] > median_dist * 2.5 and median_dist > 0:
        return 0, 50.0, 0.0  # novel regime — don't predict

    # Neighbor aggregation with distance + temporal weighting
    neighbor_returns = []
    neighbor_weights = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx + 1 < n:
            next_close = df['Close'].iloc[idx + 1]
            cur_close = df['Close'].iloc[idx]
            ret = (next_close - cur_close) / (cur_close + 1e-10) * 100
            neighbor_returns.append(ret)
            dist_w = 1.0 / (dist + 1e-6)
            recency_bonus = 1.0 + 0.5 * (idx / max(1, search_end))
            neighbor_weights.append(dist_w * recency_bonus)

    if not neighbor_returns:
        return 0, 50.0, 0.0

    weights = np.array(neighbor_weights)
    weights /= weights.sum()
    weighted_return = np.sum(np.array(neighbor_returns) * weights)

    # Direction vote with calibrated confidence
    rets = np.array(neighbor_returns)
    up_w = float(np.sum(weights[rets > 0])) if np.any(rets > 0) else 0.0
    down_w = float(np.sum(weights[rets <= 0])) if np.any(rets <= 0) else 0.0
    total_w = up_w + down_w + 1e-10
    vote_margin = abs(up_w - down_w) / total_w

    signal_int = 1 if up_w > down_w else (-1 if down_w > up_w else 0)
    expected_move = weighted_return  # raw weighted return — confidence handles uncertainty

    # Calibrated confidence: backtest KNN over recent data for empirical winrate
    if calibrate:
        cal_window = min(100, n - search_end - 5, search_end - 20)
        if cal_window > 20:
            wins = 0; trials = 0
            for i in range(n - cal_window - 1, n - 1):
                sub_end = int(i * lookback_ratio)
                if sub_end < k_eff + 5:
                    continue
                sub_scaler = StandardScaler()
                X_sub = sub_scaler.fit_transform(df[selected].iloc[:i].values)
                X_sub_w = X_sub * fw[np.newaxis, :]
                sub_cur = X_sub_w[-1:]; sub_hist = X_sub_w[:sub_end]
                sub_nn = NearestNeighbors(n_neighbors=min(k_eff, sub_end-1), metric='euclidean')
                sub_nn.fit(sub_hist)
                _, sub_idxs = sub_nn.kneighbors(sub_cur)
                sub_ups = sum(1 for idx in sub_idxs[0]
                             if idx + 1 < i and df['Close'].iloc[idx+1] > df['Close'].iloc[idx])
                sub_downs = len(sub_idxs[0]) - sub_ups
                if sub_ups == sub_downs: continue
                pred_dir = 1 if sub_ups > sub_downs else -1
                if i + 1 < n:
                    actual_dir = 1 if df['Close'].iloc[i+1] > df['Close'].iloc[i] else -1
                    trials += 1
                    if pred_dir == actual_dir: wins += 1
            if trials > 10:
                confidence = wins / trials * 60.0 + vote_margin * 40.0
                confidence = max(40.0, min(confidence, 90.0))
            else:
                confidence = 50.0 + vote_margin * 40.0
        else:
            confidence = 50.0 + vote_margin * 40.0
    else:
        confidence = 50.0 + vote_margin * 40.0

    return signal_int, confidence, expected_move


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_statistical(csv_file, k=50):
    print("=" * 60)
    print("KNN STATISTICAL MODEL")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(csv_file)
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    if any(c not in df.columns for c in required):
        raise ValueError("CSV missing OHLCV columns")

    df = compute_indicators(df)
    print(f"Records: {len(df)}  Features: {len(FEATURES)}  k={k}")

    # KNN prediction
    signal_int, confidence, expected_move_pct = knn_predict(df, FEATURES, k=k, calibrate=True)
    today_price = float(df['Close'].iloc[-1])

    if signal_int == 1:
        signal = "BUY (LONG)"
        emoji = "[BUY]"
    elif signal_int == -1:
        signal = "SHORT (SELL)"
        emoji = "[SHORT]"
    else:
        signal = "HOLD (No clear signal)"
        emoji = "[HOLD]"

    # ATR-based TP/SL
    atr_val = float(df['ATR_14'].iloc[-1])
    if pd.isna(atr_val) or atr_val <= 0:
        atr_val = today_price * 0.02

    sl_dist = 1.5 * atr_val
    tp_dist = 2.0 * atr_val
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

    print(f"\n{emoji} SIGNAL: {signal}")
    print(f"\nCurrent Price (Today):      ${today_price:.2f}")
    print(f"Expected Move (KNN):        {expected_move_pct:+.2f}%")
    print(f"k={k} nearest neighbors")
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

    mc_result = monte_carlo_simulation(
        current_price=today_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        volatility=volatility,
        predicted_move_pct=expected_move_pct,
    )

    pattern_result = find_similar_patterns(
        df=df,
        current_price=today_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )

    ensemble = calculate_ensemble_probability(
        prediction_result=None,
        monte_carlo_result=mc_result,
        pattern_result=pattern_result,
    )

    report = format_analysis_report(
        prediction_result=None,
        monte_carlo_result=mc_result,
        pattern_result=pattern_result,
        ensemble_result=ensemble,
        signal=signal,
        current_price=today_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )
    print(report)

    if ensemble:
        print(f"ENSEMBLE_PROBABILITY: {ensemble['ensemble_probability']:.1f}%")
        print(f"CONFIDENCE_LEVEL: {ensemble['confidence_level']}")
        print(f"RECOMMENDATION: {ensemble['recommendation']}")

    # --- Plots ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Walk-back predictions: for last 200 days, predict next-day return using KNN
    n_plot = min(200, len(df) - 60)
    actual_returns = []
    pred_returns = []
    for i in range(len(df) - n_plot, len(df) - 1):
        _, _, ret = knn_predict(df.iloc[:i+1], FEATURES, k=k, top_n_features=20)
        actual = float(df['Close'].pct_change().iloc[i+1] * 100) if i+1 < len(df) else 0
        pred_returns.append(ret)
        actual_returns.append(actual)

    plt.figure(figsize=(15, 6))
    plt.plot(range(n_plot-1), actual_returns, label='Actual Return %', color='blue', linewidth=1.5)
    plt.plot(range(n_plot-1), pred_returns, label='KNN Predicted Return %', color='red', linewidth=1.5, alpha=0.7)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.title(f'KNN Statistical: Predicted vs Actual Returns (Last {n_plot} Days, k={k})')
    plt.xlabel('Days Ago')
    plt.ylabel('Next-Day Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('statistical_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Neighbor distance histogram (for the current prediction)
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    # Use same feature selection as knn_predict
    returns = df['Close'].pct_change().shift(-1) * 100
    corrs = df[FEATURES].corrwith(returns).abs().fillna(0)
    selected_feats = corrs.nlargest(20).index.tolist()
    sc = StandardScaler()
    X_all = sc.fit_transform(df[selected_feats].values)
    search_end = int(len(df) * 0.7)
    nn = NearestNeighbors(n_neighbors=min(k, search_end-1), metric='euclidean')
    nn.fit(X_all[:search_end])
    dists, _ = nn.kneighbors(X_all[-1:])
    plt.figure(figsize=(10, 4))
    plt.bar(range(1, len(dists[0])+1), sorted(dists[0], reverse=True), color='steelblue')
    plt.title(f'KNN Neighbor Distances (Current Day, k={k})')
    plt.xlabel('Neighbor Rank (closest first)')
    plt.ylabel('Euclidean Distance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('statistical_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plots saved: statistical_predictions.png, statistical_feature_importance.png")

    # --- Save signal file for RL agent ---
    ensemble_prob = ensemble['ensemble_probability'] if ensemble else 50.0
    sig_path = os.path.join(base_dir, 'statistical_signal.txt')
    with open(sig_path, 'w') as f:
        f.write(f"signal: {signal_int}\n")
        f.write(f"prob: {confidence/100:.4f}\n")
        f.write(f"ensemble_prob: {ensemble_prob:.1f}\n")
    print(f"\nSignal saved to: {sig_path}")

    # --- Save model info ---
    info_path = os.path.join(base_dir, 'statistical_model_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"ticker: {os.path.basename(csv_file).split('_')[0]}\n")
        f.write(f"model_type: KNN Statistical (k={k})\n")
        f.write(f"n_features: {len(FEATURES)}\n")
        f.write(f"n_records: {len(df)}\n")
        f.write(f"k: {k}\n")
        f.write(f"timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"signal: {signal_int}\n")
        f.write(f"ensemble_prob: {ensemble_prob:.1f}\n")

    print(f"Info saved to: {info_path}")
    print("\n" + "=" * 60)
    print("KNN STATISTICAL MODEL COMPLETE")
    print("=" * 60)

    return signal_int, confidence, ensemble


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN Statistical Model')
    parser.add_argument('csv_file', type=str, help='Path to CSV file')
    parser.add_argument('--k', type=int, default=50, help='Number of nearest neighbors (default: 50)')
    args = parser.parse_args()
    run_statistical(args.csv_file, k=args.k)
