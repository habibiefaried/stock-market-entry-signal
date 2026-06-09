"""
Recount - Live trading prediction using pre-trained models.

Loads all 7 trained models + the RL agent policy, fetches recent market data,
and produces a BUY/SHORT/HOLD decision with TP/SL levels at the current price.

Usage:
    python recount.py --ticker MSFT --current-price 441.31
    python recount.py --ticker MSFT --current-price 441.31 --leverage 5
    python recount.py --ticker AAPL --current-price 312.50 --leverage 5 --months 12

The models must already be trained for this ticker (via `python main.py --ticker X`).
If no trained models are found, or they were trained for a different ticker,
the script exits with an error.
"""

import argparse
import os
import sys
import warnings
import logging
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Import shared utilities from agent_trader
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from agent_trader import (
    compute_indicators,
    compute_regime,
    build_state,
    _build_feature_row,
    MODEL_NAMES,
    STATE_DIM,
    ACTION_DIM,
    TORCH_AVAILABLE,
    PPOPolicyTorch,
    PPOPolicy,
    ACTIONS,
    ACTION_LONG,
    ACTION_SHORT,
    ACTION_HOLD,
)
from model_store import MODEL_DIR, find_latest_prefix, find_latest_rl_weights

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
REQUIRED_MODEL_NAMES = [
    'xgboost', 'xgboost_heavy',
    'lightgbm', 'lightgbm_heavy',
    'randomforest', 'randomforest_heavy',
    'catboost_bayes',
]

# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def get_trained_ticker(prefix):
    """Read a model_info.txt file to determine which ticker was trained."""
    info_path = os.path.join(MODEL_DIR, f'{prefix}xgboost_model_info.txt')
    if os.path.exists(info_path):
        with open(info_path) as f:
            for line in f:
                if line.startswith('ticker:'):
                    return line.split(':', 1)[1].strip()
    # Fallback: try any model info file
    for name in MODEL_NAMES:
        info_path = os.path.join(MODEL_DIR, f'{prefix}{name}_model_info.txt')
        if os.path.exists(info_path):
            with open(info_path) as f:
                for line in f:
                    if line.startswith('ticker:'):
                        return line.split(':', 1)[1].strip()
    return None


def verify_models_exist(prefix):
    """Check all 7 model pkl files exist for the given prefix."""
    missing = []
    for name in REQUIRED_MODEL_NAMES:
        fpath = os.path.join(MODEL_DIR, f'{prefix}{name}_model.pkl')
        if not os.path.exists(fpath):
            missing.append(f'{prefix}{name}_model.pkl')
    return missing


# ---------------------------------------------------------------------------
# DATA FETCHING
# ---------------------------------------------------------------------------

def fetch_recent_data(ticker, months=12):
    """Fetch recent daily OHLCV data for a ticker."""
    try:
        import yfinance as yf
    except ImportError:
        print("Error: yfinance not installed. Install with: pip install yfinance")
        sys.exit(1)

    print(f"Fetching {ticker} data ({months} months)...")
    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(period=f"{months}mo", interval="1d")

    if df.empty:
        print(f"Error: No data found for {ticker}")
        sys.exit(1)

    df.reset_index(inplace=True)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    print(f"  {len(df)} records ({df['Date'].min()} to {df['Date'].max()})")
    return df


# ---------------------------------------------------------------------------
# MODEL PREDICTION
# ---------------------------------------------------------------------------

def load_models(prefix):
    """Load all 7 trained pkl models, scalers, and feature lists."""
    loaded = {}
    for name in REQUIRED_MODEL_NAMES:
        mp = os.path.join(MODEL_DIR, f'{prefix}{name}_model.pkl')
        sp = os.path.join(MODEL_DIR, f'{prefix}{name}_scaler.pkl')
        fp = os.path.join(MODEL_DIR, f'{prefix}{name}_features.txt')

        if not all(os.path.exists(p) for p in [mp, sp, fp]):
            print(f"  Warning: missing files for {name}, skipping")
            continue

        try:
            model  = joblib.load(mp)
            scaler = joblib.load(sp)
            with open(fp) as fh:
                feats = [line.strip() for line in fh if line.strip()]
            loaded[name] = (model, scaler, feats)
            print(f"  Loaded {name} ({len(feats)} features)")
        except Exception as e:
            print(f"  Could not load {name}: {e}")

    return loaded


def predict_all_models(loaded_models, df_indicators):
    """
    Run each model on the latest row and return a dict of signals and probs.
    Uses the same adaptive threshold logic as the training scripts.
    """
    idx = len(df_indicators) - 1
    close = float(df_indicators['Close'].iloc[idx])
    vol_20d_pct = float(df_indicators['Close'].pct_change().tail(20).std() * 100)
    sig_threshold = max(0.15 * vol_20d_pct, 0.1)

    results = {}
    for name, (model, scaler, feats) in loaded_models.items():
        try:
            feat_df = _build_feature_row(df_indicators, idx, feats)
            if feat_df is None:
                results[name] = {'signal': 0, 'signal_text': 'HOLD', 'prob': 0.5,
                                 'move_pct': 0.0, 'error': 'Feature build failed'}
                continue

            X = scaler.transform(feat_df)
            pred = float(model.predict(X)[0])
            move_pct = pred

            if move_pct > sig_threshold:
                sig, sig_text = 1, 'BUY (LONG)'
            elif move_pct < -sig_threshold:
                sig, sig_text = -1, 'SHORT (SELL)'
            else:
                sig, sig_text = 0, 'HOLD (No clear signal)'

            prob = min(0.5 + abs(move_pct) / 5, 0.95)

            results[name] = {
                'signal': sig,
                'signal_text': sig_text,
                'prob': prob,
                'move_pct': move_pct,
                'error': None,
            }
        except Exception as e:
            results[name] = {'signal': 0, 'signal_text': 'HOLD', 'prob': 0.5,
                             'move_pct': 0.0, 'error': str(e)}

    return results


# ---------------------------------------------------------------------------
# RL AGENT LOADING
# ---------------------------------------------------------------------------

def load_rl_policy(ticker):
    """Load the latest dated RL agent policy for a specific ticker."""
    weights_path, kind = find_latest_rl_weights(ticker)

    if weights_path is None:
        print("  Warning: No RL policy found - using voting fallback")
        return None, None

    print(f"  Loading RL policy: {os.path.basename(weights_path)}")
    try:
        if kind == 'torch' and TORCH_AVAILABLE:
            policy = PPOPolicyTorch(state_dim=STATE_DIM, hidden=128)
            policy.load_state_dict(__import__('torch').load(weights_path))
            policy.eval()
            return policy, 'torch'
        elif kind == 'numpy':
            policy = PPOPolicy(state_dim=STATE_DIM)
            w = np.load(weights_path)
            policy.W1 = w['W1']; policy.b1 = w['b1']
            policy.W2 = w['W2']; policy.b2 = w['b2']
            policy.W3 = w['W3']; policy.b3 = w['b3']
            policy.Wv1 = w['Wv1']; policy.bv1 = w['bv1']
            policy.Wv2 = w['Wv2']; policy.bv2 = w['bv2']
            return policy, 'numpy'
        else:
            print(f"  Warning: RL weights are {kind} but PyTorch not available - using voting fallback")
            return None, None
    except Exception as e:
        print(f"  Warning: Could not load RL policy ({e}) - using voting fallback")
        return None, None


# ---------------------------------------------------------------------------
# TP/SL CALCULATION
# ---------------------------------------------------------------------------

def calculate_tp_sl(close, atr, signal_int):
    """ATR-based TP/SL - must match TradingEnv.step() and all training scripts."""
    sl_dist = 1.0 * atr
    tp_dist = 1.5 * atr

    if signal_int == 1:   # LONG
        sl = close - sl_dist
        tp = close + tp_dist
    elif signal_int == -1:  # SHORT
        sl = close + sl_dist
        tp = close - tp_dist
    else:  # HOLD
        sl = close
        tp = close

    return sl, tp


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_recount(ticker, current_price, leverage=5.0, months=12):
    print("=" * 70)
    print("RECOUNT - LIVE TRADING PREDICTION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Verify models exist and match ticker
    # ------------------------------------------------------------------
    print("\n[1/5] Verifying trained models...")
    prefix, date_str = find_latest_prefix(ticker)
    if prefix is None:
        print(f"\nERROR: No trained models found for '{ticker}' in MODELS/.")
        print("You must train the models first. Run:")
        print(f"  python main.py --ticker {ticker}")
        sys.exit(1)

    missing = verify_models_exist(prefix)
    if missing:
        print(f"\nERROR: {len(missing)} model file(s) missing for {ticker}:")
        for f in missing:
            print(f"  - {f}")
        print("\nYou must train the models first. Run:")
        print(f"  python main.py --ticker {ticker}")
        sys.exit(1)

    trained_ticker = get_trained_ticker(prefix)
    if trained_ticker and trained_ticker.upper() != ticker.upper():
        print(f"\nERROR: Models were trained for '{trained_ticker}', "
              f"not '{ticker}'.")
        print(f"Re-train for {ticker} first:")
        print(f"  python main.py --ticker {ticker}")
        sys.exit(1)

    print(f"  All 7 models present, trained for {ticker} ({date_str})")

    # ------------------------------------------------------------------
    # Step 2: Fetch recent data and compute indicators
    # ------------------------------------------------------------------
    print(f"\n[2/5] Fetching recent data for {ticker}...")
    df_raw = fetch_recent_data(ticker, months=months)
    df = compute_indicators(df_raw)
    print(f"  {len(df)} records after indicator computation")

    # ------------------------------------------------------------------
    # Step 3: Load models and predict
    # ------------------------------------------------------------------
    print("\n[3/5] Loading models and generating predictions...")
    loaded = load_models(prefix)
    if not loaded:
        print("\nERROR: No models could be loaded.")
        sys.exit(1)

    model_results = predict_all_models(loaded, df)

    # ------------------------------------------------------------------
    # Step 4: Build state and run RL agent
    # ------------------------------------------------------------------
    print("\n[4/5] Running RL agent...")
    rl_policy, rl_type = load_rl_policy(ticker)

    # Build the signals row for state construction
    last_idx = len(df) - 1
    row = {
        'close': float(df['Close'].iloc[last_idx]),
        'rsi': float(df['RSI_14'].iloc[last_idx]),
        'rsi_7': float(df['RSI_7'].iloc[last_idx]),
        'atr': float(df['ATR_14'].iloc[last_idx]),
        'volatility': float(df['Volatility_20d'].iloc[last_idx] if 'Volatility_20d' in df.columns else df['Volatility'].iloc[last_idx]),
        'trend': float(df['Close_SMA20_ratio'].iloc[last_idx]),
        'macd_hist': float(df['MACD_hist'].iloc[last_idx]),
        'bb_pct': float(df['BB_pct'].iloc[last_idx]),
        'stoch_k': float(df['STOCH_K'].iloc[last_idx]),
        'volume_ratio': float(df['Volume_MA20_ratio'].iloc[last_idx]),
        'sma50_ratio': float(df['Close_SMA50_ratio'].iloc[last_idx]),
        'regime': compute_regime(df, last_idx),
        'adx': float(df['ADX_14'].iloc[last_idx]),
        'chop': float(df['CHOP_14'].iloc[last_idx]),
        'ao': float(df['AO'].iloc[last_idx]),
        'dpo': float(df['DPO_20'].iloc[last_idx]),
    }

    # Add model signals and probs
    for name in MODEL_NAMES:
        if name in model_results:
            row[f'{name}_signal'] = model_results[name]['signal']
            row[f'{name}_prob']   = model_results[name]['prob']
        else:
            row[f'{name}_signal'] = 0
            row[f'{name}_prob']   = 0.5

    state = build_state(row)

    # Get RL agent action
    if rl_policy is not None:
        action, prob, _ = rl_policy.act_greedy(state)
    else:
        # Voting fallback
        votes = [model_results.get(n, {}).get('signal', 0) for n in MODEL_NAMES]
        vote_long = votes.count(1)
        vote_short = votes.count(-1)
        if vote_long > vote_short:
            action, prob = ACTION_LONG, vote_long / len(votes)
        elif vote_short > vote_long:
            action, prob = ACTION_SHORT, vote_short / len(votes)
        else:
            action, prob = ACTION_HOLD, 0.5

    # Multi-tier consensus filter (mirrors get_current_action in agent_trader.py)
    signals_raw = [model_results.get(n, {}).get('signal', 0) for n in MODEL_NAMES]
    n_long  = signals_raw.count(1)
    n_short = signals_raw.count(-1)
    n_agree = max(n_long, n_short)
    regime  = float(row['regime'])

    if n_agree >= 5:
        prob = prob * 1.0
    elif n_agree >= 4 and (
        (action == ACTION_LONG and regime > -0.5) or
        (action == ACTION_SHORT and regime < 0.5)):
        prob = prob * 0.85
    elif n_agree >= 3 and (
        (action == ACTION_LONG and regime > 0) or
        (action == ACTION_SHORT and regime < 0)):
        prob = prob * 0.6
    else:
        action = ACTION_HOLD
        prob = 0.3

    # ------------------------------------------------------------------
    # Step 5: Calculate TP/SL and print results
    # ------------------------------------------------------------------
    atr_val = max(float(row['atr']), 0.01 * current_price)
    csv_close = float(row['close'])
    sl, tp = calculate_tp_sl(current_price, atr_val, action)

    print("\n[5/5] Final decision")

    # ---- Print individual model signals ----
    print("\n" + "=" * 70)
    print("INDIVIDUAL MODEL SIGNALS")
    print("=" * 70)
    print(f"{'Model':<22s} {'Signal':>16s} {'Move %':>8s} {'Prob':>7s}")
    print("-" * 55)

    vote_long = 0
    vote_short = 0
    vote_hold = 0
    for name in MODEL_NAMES:
        r = model_results.get(name, {})
        sig = r.get('signal_text', 'N/A')
        move = r.get('move_pct', 0)
        prob_val = r.get('prob', 0.5)
        err = r.get('error')
        flag = f"  [!] {err}" if err else ""

        if r.get('signal') == 1:
            vote_long += 1
        elif r.get('signal') == -1:
            vote_short += 1
        else:
            vote_hold += 1

        print(f"{name:<22s} {sig:>16s} {move:>+7.2f}% {prob_val:>6.1%} {flag}")

    print("-" * 55)
    print(f"{'Consensus':<22s} LONG={vote_long} SHORT={vote_short} HOLD={vote_hold}  "
          f"(agree={max(vote_long, vote_short)}/7)")
    regime_text = {1.0: 'BULL', -1.0: 'BEAR', 0.0: 'RANGING'}.get(regime, 'N/A')
    print(f"{'Market Regime':<22s} {regime_text}")

    # ---- Print RL agent decision ----
    print("\n" + "=" * 70)
    print("RECOUNT DECISION")
    print("=" * 70)

    action_label = ACTIONS[action]
    if action == ACTION_LONG:
        action_color = 'GREEN'
    elif action == ACTION_SHORT:
        action_color = 'RED'
    else:
        action_color = 'AMBER'

    print(f"\n  >>> ACTION:  {action_label} ({action_color})")
    print(f"  >>> Confidence: {prob*100:.1f}%")
    print(f"\n  Entry Price:    ${current_price:,.2f}")
    print(f"  Stop Loss:      ${sl:,.2f}  ({(sl - current_price) / current_price * 100:+.2f}%)")
    print(f"  Take Profit:    ${tp:,.2f}  ({(tp - current_price) / current_price * 100:+.2f}%)")

    # Leverage P&L
    if leverage > 1.0:
        sl_lev = (sl - current_price) / current_price * 100 * leverage
        tp_lev = (tp - current_price) / current_price * 100 * leverage
        rr = abs(tp_lev / sl_lev) if abs(sl_lev) > 0.001 else 0
        print(f"\n  --- {leverage:.0f}x Leverage Position P&L ---")
        print(f"  Stop Loss P&L:    {sl_lev:+.1f}%")
        print(f"  Take Profit P&L:  {tp_lev:+.1f}%")
        print(f"  Risk/Reward:      {rr:.1f}:1")
        max_risk_pct = 2.0
        safe_size = max_risk_pct / max(abs(sl_lev), 0.01)
        print(f"  Max safe position (2% risk): {safe_size:.1%} of account")
        if abs(sl_lev) > 5.0:
            print(f"  [!] WARNING: SL={abs(sl_lev):.1f}% is high. Reduce leverage or use smaller position.")

    # ---- CSV close vs current price note ----
    if abs(current_price - csv_close) / csv_close > 0.005:
        print(f"\n  Note: CSV last close was ${csv_close:.2f}, "
              f"current price is ${current_price:.2f} "
              f"({(current_price - csv_close) / csv_close * 100:+.2f}%)")

    print("\n" + "=" * 70)
    print("DISCLAIMER: This is a statistical prediction, NOT financial advice.")
    print("Always do your own research and manage risk appropriately.")
    print("=" * 70)

    return action_label, prob, sl, tp


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Recount - Live trading prediction using pre-trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python recount.py --ticker MSFT --current-price 441.31
  python recount.py --ticker MSFT --current-price 441.31 --leverage 5
  python recount.py --ticker AAPL --current-price 312.50 --leverage 5 --months 12

Requirements:
  Models must already be trained for this ticker via:
    python main.py --ticker <TICKER>
        """
    )
    parser.add_argument('--ticker', type=str, required=True,
                        help='Stock ticker symbol (e.g., MSFT, AAPL)')
    parser.add_argument('--current-price', type=float, required=True,
                        help='Live current price for entry/TP/SL calculation')
    parser.add_argument('--leverage', type=float, default=5.0,
                        help='Leverage multiplier for displayed P&L (default: 5)')
    parser.add_argument('--months', type=int, default=12,
                        help='Months of recent data to fetch for indicators (default: 12)')

    args = parser.parse_args()
    run_recount(args.ticker, args.current_price, args.leverage, args.months)
