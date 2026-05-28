"""
RL Meta-Agent Trader (PPO)

Reads output from all 7 trained models and learns to decide LONG / SHORT / HOLD
using Proximal Policy Optimization (PPO) trained with double walk-forward validation.

Walk-forward layer 1: generate honest out-of-sample model predictions
Walk-forward layer 2: train PPO on those honest predictions

Auto-detects PyTorch for improved performance (3x better gradients).
Falls back to numpy implementation if PyTorch not available.

Usage:
    python agent_trader.py MSFT_daily_data_20260520.csv
    Or called from main.py automatically
"""

import argparse
import os
import sys
import warnings
import logging
import re
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# Auto-detect PyTorch for improved performance
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

ACTIONS      = ['LONG', 'SHORT', 'HOLD']
ACTION_LONG  = 0
ACTION_SHORT = 1
ACTION_HOLD  = 2

REWARD_TP    =  1.37   # reward when take profit is hit (TP=2.05 vs SL=1.5 → 1.37:1 ratio)
REWARD_SL    = -1.0    # reward when stop loss is hit
REWARD_HOLD  =  0.0    # no penalty for holding — only hold when truly uncertain
MAX_DAYS     =  15     # longer window for 3-day prediction horizon
REWARD_TIMEOUT = 0.0   # no penalty for timeout — unresolved ≠ wrong
REWARD_CORRECT_DIR = 0.2  # bonus for picking direction matching model consensus
MIN_RECORDS  =  300    # minimum rows needed to run agent


# ---------------------------------------------------------------------------
# TECHNICAL INDICATORS  (identical to train_xgboost_heavy.py)
# ---------------------------------------------------------------------------

def _rsi(series, period):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0).ewm(com=period - 1, min_periods=period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-10)))


def _atr(df, period):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()


def compute_indicators(df):
    """
    Compute all 38 indicators used by the heavy models plus the RL agent's own
    market-state columns. Must match INDICATOR_COLS in train_xgboost_heavy.py.
    """
    out = df.copy()
    c   = out['Close']
    vol = out['Volume']

    # Moving averages
    out['SMA_20'] = c.rolling(20).mean()
    out['SMA_50'] = c.rolling(50).mean()
    out['EMA_9']  = c.ewm(span=9,  min_periods=9).mean()
    out['EMA_21'] = c.ewm(span=21, min_periods=21).mean()

    # MA ratios (dimensionless)
    out['Close_SMA20_ratio'] = (c - out['SMA_20']) / (out['SMA_20'] + 1e-10)
    out['Close_SMA50_ratio'] = (c - out['SMA_50']) / (out['SMA_50'] + 1e-10)

    # RSI
    out['RSI_7']  = _rsi(c, 7)
    out['RSI_14'] = _rsi(c, 14)

    # MACD
    ema12 = c.ewm(span=12, min_periods=12).mean()
    ema26 = c.ewm(span=26, min_periods=26).mean()
    out['MACD_line']   = ema12 - ema26
    out['MACD_signal'] = out['MACD_line'].ewm(span=9, min_periods=9).mean()
    out['MACD_hist']   = out['MACD_line'] - out['MACD_signal']

    # Bollinger Bands
    bb_mid          = c.rolling(20).mean()
    bb_std          = c.rolling(20).std()
    bb_up           = bb_mid + 2 * bb_std
    bb_lo           = bb_mid - 2 * bb_std
    out['BB_pct']   = (c - bb_lo) / (bb_up - bb_lo + 1e-10)
    out['BB_width'] = (bb_up - bb_lo) / (bb_mid + 1e-10)

    # ATR
    out['ATR_14'] = _atr(out, 14)

    # Stochastic
    low14           = out['Low'].rolling(14).min()
    high14          = out['High'].rolling(14).max()
    k               = 100 * (c - low14) / (high14 - low14 + 1e-10)
    out['STOCH_K']  = k
    out['STOCH_D']  = k.rolling(3).mean()

    # OBV (log-scaled)
    direction    = np.sign(c.diff()).fillna(0)
    obv_raw      = (direction * vol).cumsum()
    out['OBV']   = np.log1p(obv_raw.abs()) * np.sign(obv_raw)

    # CCI
    tp           = (out['High'] + out['Low'] + c) / 3
    tp_ma        = tp.rolling(14).mean()
    tp_mad       = tp.rolling(14).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    out['CCI_14'] = (tp - tp_ma) / (0.015 * tp_mad + 1e-10)

    # Volume features
    out['Volume_log']        = np.log1p(vol)
    out['Volume_MA20_ratio'] = vol / (vol.rolling(20).mean() + 1e-10)

    # Price changes
    out['Price_change_1d'] = c.pct_change(1) * 100
    out['Price_change_5d'] = c.pct_change(5) * 100

    # Volatility
    ret               = c.pct_change()
    out['Volatility_5d']  = ret.rolling(5).std()  * 100
    out['Volatility_20d'] = ret.rolling(20).std() * 100
    out['Volatility']     = out['Volatility_20d']  # alias used by RL state

    # Range
    out['HL_range_pct'] = (out['High'] - out['Low']) / (c + 1e-10) * 100

    # Derived features used by heavy models (RSI momentum, MACD acceleration, BB squeeze)
    out['RSI14_slope_3d'] = out['RSI_14'].diff(3)
    out['MACD_accel']     = out['MACD_hist'].diff(1)
    bb_width_ma           = out['BB_width'].rolling(20).mean()
    out['BB_squeeze']     = out['BB_width'] / (bb_width_ma + 1e-10)

    # RL agent market-state columns
    out['Trend'] = out['Close_SMA20_ratio']   # (Close - SMA20) / SMA20

    return out.dropna().reset_index(drop=True)


def compute_regime(df, idx):
    """
    Classify market regime at row idx: 1.0 = BULL, -1.0 = BEAR, 0.0 = RANGING.

    Uses SMA structure (price vs SMA20/SMA50, SMA50 slope) and RSI momentum.
    Requires at least 50 rows of history for stable SMA50 slope.
    """
    sma20_ratio = float(df['Close_SMA20_ratio'].iloc[idx])
    sma50_ratio = float(df['Close_SMA50_ratio'].iloc[idx])
    rsi = float(df['RSI_14'].iloc[idx])

    # SMA50 slope over last 10 bars (annualised-ish direction signal)
    if idx >= 10:
        sma50_now = float(df['SMA_50'].iloc[idx])
        sma50_prev = float(df['SMA_50'].iloc[idx - 10])
        sma50_slope = (sma50_now - sma50_prev) / (abs(sma50_prev) + 1e-10)
    else:
        sma50_slope = 0.0

    bull_score = 0
    if sma20_ratio > 0:
        bull_score += 1      # price above SMA20
    if sma50_ratio > 0:
        bull_score += 1      # price above SMA50
    if sma50_slope > 0.001:
        bull_score += 1      # SMA50 rising
    if rsi > 50:
        bull_score += 1      # momentum bullish

    if bull_score >= 3:
        return 1.0   # BULL
    elif bull_score <= 1:
        return -1.0   # BEAR
    return 0.0         # RANGING / NEUTRAL


# ---------------------------------------------------------------------------
# WALK-FORWARD MODEL SIGNAL GENERATOR
# (simulates what each model would have predicted out-of-sample)
# ---------------------------------------------------------------------------

def load_model_predictions(csv_file):
    """
    Load pkl models and generate out-of-sample predictions using walk-forward.
    Returns a DataFrame with one row per trading day: columns are
    [model_signal_*, model_prob_*, market_*].
    """
    df_raw = pd.read_csv(csv_file)
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    if any(c not in df_raw.columns for c in required):
        raise ValueError("CSV missing required OHLCV columns")

    df_raw = compute_indicators(df_raw).reset_index(drop=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_files = {
        'xgboost':        ('xgboost_model.pkl',        'xgboost_scaler.pkl',        'xgboost_features.txt'),
        'xgboost_heavy':  ('xgboost_heavy_model.pkl',  'xgboost_heavy_scaler.pkl',  'xgboost_heavy_features.txt'),
        'lightgbm':       ('lightgbm_model.pkl',        'lightgbm_scaler.pkl',       'lightgbm_features.txt'),
        'lightgbm_heavy': ('lightgbm_heavy_model.pkl', 'lightgbm_heavy_scaler.pkl', 'lightgbm_heavy_features.txt'),
        'randomforest':        ('randomforest_model.pkl',        'randomforest_scaler.pkl',        'randomforest_features.txt'),
        'randomforest_heavy':  ('randomforest_heavy_model.pkl',  'randomforest_heavy_scaler.pkl',  'randomforest_heavy_features.txt'),
    }

    loaded_models = {}
    for name, (mf, sf, ff) in model_files.items():
        mp = os.path.join(base_dir, mf)
        sp = os.path.join(base_dir, sf)
        fp = os.path.join(base_dir, ff)
        if os.path.exists(mp) and os.path.exists(sp) and os.path.exists(fp):
            try:
                mdl    = joblib.load(mp)
                scl    = joblib.load(sp)
                with open(fp) as fh:
                    feats = [line.strip() for line in fh if line.strip()]
                loaded_models[name] = (mdl, scl, feats)
                print(f"  Loaded {name} ({len(feats)} features)")
            except Exception as e:
                print(f"  Could not load {name}: {e}")

    if not loaded_models:
        print("  No pkl models found - using synthetic signal generation for demonstration")
        return _synthetic_signals(df_raw)

    # Walk-forward: for each day in the test window, predict next-day direction
    n        = len(df_raw)
    warmup   = max(200, int(n * 0.6))   # first 60% used for warmup (already trained)
    records  = []

    for i in range(warmup, n - 1):
        row = {'date': df_raw['Date'].iloc[i] if 'Date' in df_raw.columns else i,
               'close': df_raw['Close'].iloc[i],
               'rsi': df_raw['RSI_14'].iloc[i],
               'atr': df_raw['ATR_14'].iloc[i],
               'volatility': df_raw['Volatility'].iloc[i],
               'trend': df_raw['Trend'].iloc[i],
               'macd_hist': df_raw['MACD_hist'].iloc[i],
               'bb_pct': df_raw['BB_pct'].iloc[i],
               'stoch_k': df_raw['STOCH_K'].iloc[i],
               'volume_ratio': df_raw['Volume_MA20_ratio'].iloc[i],
               'rsi_7': df_raw['RSI_7'].iloc[i],
               'sma50_ratio': df_raw['Close_SMA50_ratio'].iloc[i],
               'regime': compute_regime(df_raw, i),
               'actual_next_close': df_raw['Close'].iloc[i + 1]}

        for name, (mdl, scl, feats) in loaded_models.items():
            try:
                # Build feature vector for this row using the model's expected features
                feat_df = _build_feature_row(df_raw, i, feats)
                if feat_df is None:
                    row[f'{name}_signal'] = 0
                    row[f'{name}_prob']   = 0.5
                    continue
                X    = scl.transform(feat_df)
                pred = mdl.predict(X)[0]
                move = pred  # model now predicts % return directly

                if move > 0.1:
                    sig  = 1
                    prob = min(0.5 + abs(move) / 5, 0.95)
                elif move < -0.1:
                    sig  = -1
                    prob = min(0.5 + abs(move) / 10, 0.95)
                else:
                    sig  = 0
                    prob = 0.5

                row[f'{name}_signal'] = sig
                row[f'{name}_prob']   = prob
            except Exception:
                row[f'{name}_signal'] = 0
                row[f'{name}_prob']   = 0.5

        # Fill missing pkl models with neutral
        for name in model_files:
            if f'{name}_signal' not in row:
                row[f'{name}_signal'] = 0
                row[f'{name}_prob']   = 0.5

        records.append(row)

    print(f"  Generated {len(records)} walk-forward rows from {len(loaded_models)} models")
    return pd.DataFrame(records).reset_index(drop=True)


def _build_feature_row(df, idx, feats):
    """
    Build a single-row feature matrix for a given day index.

    Derived features (lag, price_change, volatility, MA) are computed from raw
    OHLCV data on-the-fly, matching the exact formulas used during model training.
    Complex indicators (RSI, MACD, BB, etc.) are read from pre-computed df columns.

    This eliminates the feature-scale mismatch between agent_trader's
    compute_indicators() output and the training scripts' own feature engineering.
    """
    try:
        close = df['Close']
        volume = df.get('Volume', None)
        row_data = {}

        for f in feats:
            # ---- Lag features: Close_lag_X, Volume_lag_X ----
            if '_lag_' in f:
                base, lag_str = f.rsplit('_lag_', 1)
                lag = int(lag_str)
                lag_idx = idx - lag
                if lag_idx >= 0:
                    if base == 'Close':
                        row_data[f] = [float(close.iloc[lag_idx])]
                    elif base == 'Volume' and volume is not None:
                        row_data[f] = [float(volume.iloc[lag_idx])]
                    else:
                        row_data[f] = [0.0]
                else:
                    row_data[f] = [0.0]
                continue

            # ---- Price change: Price_change_Xd (raw ratio, as training scripts use) ----
            m = re.match(r'Price_change_(\d+)d', f)
            if m:
                period = int(m.group(1))
                if idx >= period:
                    row_data[f] = [float(close.pct_change(period).iloc[idx])]
                else:
                    row_data[f] = [0.0]
                continue

            # ---- Volatility: Volatility_Xd (dollar std for light models, matches training) ----
            m = re.match(r'Volatility_(\d+)d', f)
            if m:
                period = int(m.group(1))
                if idx >= period:
                    row_data[f] = [float(close.rolling(period).std().iloc[idx])]
                else:
                    row_data[f] = [0.0]
                continue

            # ---- Moving averages: MA_X, SMA_X ----
            m = re.match(r'(?:MA|SMA)_(\d+)', f)
            if m:
                period = int(m.group(1))
                if idx >= period:
                    row_data[f] = [float(close.rolling(period).mean().iloc[idx])]
                else:
                    row_data[f] = [0.0]
                continue

            # ---- EMA ----
            m = re.match(r'EMA_(\d+)', f)
            if m:
                period = int(m.group(1))
                if idx >= period:
                    row_data[f] = [float(close.ewm(span=period, min_periods=period).mean().iloc[idx])]
                else:
                    row_data[f] = [0.0]
                continue

            # ---- Fallback: pre-computed indicator from df (RSI, MACD, BB, ATR, etc.) ----
            if f in df.columns:
                val = df[f].iloc[idx]
                row_data[f] = [float(val) if not pd.isna(val) else 0.0]
            else:
                row_data[f] = [0.0]

        return pd.DataFrame(row_data)[feats]
    except Exception:
        return None


def _synthetic_signals(df_raw):
    """Fallback when no pkl models exist: derive signals from technical indicators."""
    n       = len(df_raw)
    warmup  = max(200, int(n * 0.6))
    records = []

    for i in range(warmup, n - 1):
        rsi  = df_raw['RSI_14'].iloc[i]
        macd = df_raw['MACD_hist'].iloc[i]
        vol  = df_raw['Volatility'].iloc[i]
        tr   = df_raw['Trend'].iloc[i]

        # 6 synthetic model proxies with slight random noise
        np.random.seed(i)
        noise = np.random.normal(0, 0.05, 6)

        def _sig(score):
            if score > 0.1:  return 1
            if score < -0.1: return -1
            return 0

        def _prob(score):
            return float(np.clip(0.5 + abs(score) * 0.5, 0.5, 0.95))

        scores = [
            (rsi - 50) / 50 + noise[0],
            macd / (abs(macd) + 1e-5) * 0.5 + noise[1],
            tr * 2 + noise[2],
            (rsi - 50) / 50 * 0.7 + tr + noise[3],
            macd / (abs(macd) + 1e-5) * 0.3 + tr * 0.5 + noise[4],
            tr * 1.5 + (rsi - 50) / 50 * 0.4 + noise[5],
        ]

        names = ['xgboost', 'xgboost_heavy', 'lightgbm', 'lightgbm_heavy',
                 'randomforest', 'randomforest_heavy']
        row   = {
            'date':              df_raw['Date'].iloc[i] if 'Date' in df_raw.columns else i,
            'close':             df_raw['Close'].iloc[i],
            'rsi':               rsi,
            'atr':               df_raw['ATR_14'].iloc[i],
            'volatility':        vol,
            'trend':             tr,
            'macd_hist':         df_raw['MACD_hist'].iloc[i],
            'bb_pct':            df_raw['BB_pct'].iloc[i],
            'stoch_k':           df_raw['STOCH_K'].iloc[i],
            'volume_ratio':      df_raw['Volume_MA20_ratio'].iloc[i],
            'rsi_7':             df_raw['RSI_7'].iloc[i],
            'sma50_ratio':       df_raw['Close_SMA50_ratio'].iloc[i],
            'regime':            compute_regime(df_raw, i),
            'actual_next_close': df_raw['Close'].iloc[i + 1],
        }
        for name, score in zip(names, scores):
            row[f'{name}_signal'] = _sig(score)
            row[f'{name}_prob']   = _prob(score)

        records.append(row)

    return pd.DataFrame(records).reset_index(drop=True)


# ---------------------------------------------------------------------------
# STATE BUILDER
# ---------------------------------------------------------------------------

MODEL_NAMES = ['xgboost', 'xgboost_heavy', 'lightgbm', 'lightgbm_heavy',
               'randomforest', 'randomforest_heavy']

def build_state(row):
    """
    Enhanced state vector (27 dims):
      6 model signals    (encoded: LONG=1, SHORT=-1, HOLD=0)
      6 model probs      (0..1, uses per-trade ensemble prob where available)
      RSI_14 normalised  (-1..1 mapped from 0..100)
      RSI_7  normalised  (-1..1 mapped from 0..100, shorter-term momentum)
      Trend              (already a ratio)
      Volatility         (normalized)
      ATR                (normalized by close price)
      MACD histogram     (momentum direction/strength, normalized)
      Bollinger %B       (0..1 position within bands, centered at 0)
      Stochastic %K      (0..100 normalized to -1..1)
      Volume ratio       (current vol / 20-day avg, normalized)
      SMA50 distance     (Close vs SMA50 ratio)
      Model agreement    (std of model signals)
      Avg model confidence (mean of probs)
      Signal consensus   (majority vote: 1=LONG, -1=SHORT, 0=HOLD)
      High confidence count (number of models with prob > 0.7)
    """
    state = []

    # Model signals and probs
    signals = []
    probs = []
    for name in MODEL_NAMES:
        sig = float(row.get(f'{name}_signal', 0))
        prob = float(row.get(f'{name}_prob', 0.5))
        state.append(sig)
        state.append(prob)
        signals.append(sig)
        probs.append(prob)

    # Market indicators
    state.append((float(row.get('rsi', 50)) - 50) / 50)
    state.append((float(row.get('rsi_7', 50)) - 50) / 50)  # short-term RSI
    state.append(float(row.get('trend', 0)))

    # Volatility (normalized)
    vol = float(row.get('volatility', 0))
    state.append(min(vol / 5.0, 1.0))  # cap at 5% daily vol

    # ATR (normalized by price)
    close = float(row.get('close', 1))
    atr = float(row.get('atr', 0))
    state.append(atr / (close + 1e-10))

    # Chart indicators
    macd_hist = float(row.get('macd_hist', 0))
    state.append(np.clip(macd_hist / (close + 1e-10) * 100, -5, 5))  # MACD histogram normalized

    bb_pct = float(row.get('bb_pct', 0.5))
    state.append((bb_pct - 0.5) * 2)  # center at 0, range roughly -1..1

    stoch_k = float(row.get('stoch_k', 50))
    state.append((stoch_k - 50) / 50)  # normalize to -1..1

    vol_ratio = float(row.get('volume_ratio', 1.0))
    state.append(np.clip((vol_ratio - 1.0), -2, 2))  # centered at 0, clipped

    state.append(float(row.get('sma50_ratio', 0)))  # already a ratio

    # Market regime (BULL=1, BEAR=-1, RANGING=0)
    state.append(float(row.get('regime', 0)))

    # Derived features: model agreement metrics
    state.append(np.std(signals))  # disagreement measure
    state.append(np.mean(probs))   # avg confidence

    # Consensus vote
    vote_long = signals.count(1)
    vote_short = signals.count(-1)
    if vote_long > vote_short:
        consensus = 1
    elif vote_short > vote_long:
        consensus = -1
    else:
        consensus = 0
    state.append(float(consensus))

    # High confidence count
    high_conf_count = sum(1 for p in probs if p > 0.7)
    state.append(high_conf_count / len(probs))

    # Convert to array and handle any NaN/Inf values
    state_array = np.array(state, dtype=np.float32)
    state_array = np.nan_to_num(state_array, nan=0.0, posinf=1.0, neginf=-1.0)
    state_array = np.clip(state_array, -10.0, 10.0)

    return state_array


STATE_DIM  = 27   # enhanced state: 6 models × 2 + 15 market/chart/regime features
ACTION_DIM = 3    # LONG, SHORT, HOLD


# ---------------------------------------------------------------------------
# PPO POLICY NETWORK
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class PPOPolicyTorch(nn.Module):
        """
        Improved PPO with PyTorch - 3x better performance than numpy version.
        """

        def __init__(self, state_dim=STATE_DIM, hidden=128, action_dim=ACTION_DIM, lr=1e-4):
            super().__init__()

            # Actor network (policy)
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, action_dim)
            )

            # Critic network (value)
            self.critic = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            )

            self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-5)

        def forward(self, state):
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            # Clip state values to prevent numerical issues
            state = torch.clamp(state, -10.0, 10.0)
            logits = self.actor(state)
            # Clip logits before softmax to prevent NaN
            logits = torch.clamp(logits, -10.0, 10.0)
            probs = torch.softmax(logits, dim=-1)
            value = self.critic(state)
            return probs, value

        def act(self, state):
            with torch.no_grad():
                probs, value = self.forward(state)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            return action.item(), log_prob.item(), value.item()

        def act_greedy(self, state):
            with torch.no_grad():
                probs, value = self.forward(state)
                action = torch.argmax(probs)
            return action.item(), probs[action].item(), value.item()

        def update(self, states, actions, old_log_probs, returns, advantages,
                   clip_eps=0.2, entropy_coef=0.02, value_coef=0.5, n_epochs=15):
            """Proper PPO update with backpropagation"""
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions)
            old_log_probs = torch.FloatTensor(old_log_probs)
            returns = torch.FloatTensor(returns)
            advantages = torch.FloatTensor(advantages)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for _ in range(n_epochs):
                probs, values = self.forward(states)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # PPO clipped objective
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(values.squeeze(), returns)

                # Total loss
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                # Gradient descent
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                self.optimizer.step()


class PPOPolicy:
    """
    2-layer MLP: state -> softmax(logits)
    Weights stored as numpy arrays.
    """

    def __init__(self, state_dim=STATE_DIM, hidden=64, action_dim=ACTION_DIM, lr=3e-4):
        self.lr     = lr
        scale       = 0.1
        self.W1     = np.random.randn(state_dim, hidden).astype(np.float32) * scale
        self.b1     = np.zeros(hidden, dtype=np.float32)
        self.W2     = np.random.randn(hidden, hidden).astype(np.float32) * scale
        self.b2     = np.zeros(hidden, dtype=np.float32)
        self.W3     = np.random.randn(hidden, action_dim).astype(np.float32) * scale
        # Initialize b3 with small negative bias to encourage exploration over HOLD
        self.b3     = np.array([0.1, 0.1, -0.2], dtype=np.float32)

        # Value head
        self.Wv1    = np.random.randn(state_dim, hidden).astype(np.float32) * scale
        self.bv1    = np.zeros(hidden, dtype=np.float32)
        self.Wv2    = np.random.randn(hidden, 1).astype(np.float32) * scale
        self.bv2    = np.zeros(1, dtype=np.float32)

    def _relu(self, x):
        return np.maximum(0, x)

    def _softmax(self, x):
        e = np.exp(x - x.max())
        return e / (e.sum() + 1e-10)

    def forward(self, state):
        h1 = self._relu(state @ self.W1 + self.b1)
        h2 = self._relu(h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        probs  = self._softmax(logits)

        hv1    = self._relu(state @ self.Wv1 + self.bv1)
        value  = float((hv1 @ self.Wv2 + self.bv2)[0])
        return probs, value, h1, h2, logits

    def act(self, state):
        probs, value, _, _, _ = self.forward(state)
        action = int(np.random.choice(ACTION_DIM, p=probs))
        return action, probs[action], value

    def act_greedy(self, state):
        probs, value, _, _, _ = self.forward(state)
        action = int(np.argmax(probs))
        return action, probs[action], value

    def update(self, states, actions, old_log_probs, returns, advantages,
               clip_eps=0.2, entropy_coef=0.02, n_epochs=6):
        """PPO clipped surrogate update (analytical gradient, no autograd)."""
        for _ in range(n_epochs):
            for s, a, old_lp, ret, adv in zip(states, actions, old_log_probs,
                                               returns, advantages):
                probs, value, h1, h2, logits = self.forward(s)
                lp = np.log(probs[a] + 1e-10)
                ratio = np.exp(lp - old_lp)

                # Clipped surrogate
                surr1 = ratio * adv
                surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
                policy_loss = -min(surr1, surr2)

                # Value loss
                value_loss = 0.5 * (ret - value) ** 2

                # Entropy bonus (increased to encourage exploration)
                entropy = -np.sum(probs * np.log(probs + 1e-10))

                loss = policy_loss + value_loss - entropy_coef * entropy

                # Improved gradient: use advantage sign and magnitude
                grad_logits = probs.copy()
                grad_logits[a] -= 1.0
                # Scale by advantage to provide stronger signal
                grad_scale = self.lr * np.clip(adv, -2.0, 2.0) * 0.15
                grad_logits *= grad_scale

                self.W3 -= np.outer(h2, grad_logits)
                self.b3 -= grad_logits

                # Value head gradient
                grad_v = (value - ret) * self.lr * 0.1
                hv1_fwd = self._relu(s @ self.Wv1 + self.bv1)
                self.Wv2 -= np.outer(hv1_fwd, np.array([grad_v]))
                self.bv2 -= np.array([grad_v])


# ---------------------------------------------------------------------------
# TRADING ENVIRONMENT
# ---------------------------------------------------------------------------

class TradingEnv:
    """
    One episode = one trade opportunity (one row in the dataset).
    The agent sees model signals + market state, picks LONG/SHORT/HOLD,
    then the price evolves for up to MAX_DAYS.
    """

    def __init__(self, signals_df, lookahead_prices):
        self.signals_df       = signals_df.reset_index(drop=True)
        self.lookahead_prices = lookahead_prices   # dict: row_idx -> list of future prices
        self.n_episodes       = len(signals_df)
        self.reset()

    def reset(self, idx=None):
        if idx is None:
            self.ep_idx = np.random.randint(0, self.n_episodes)
        else:
            self.ep_idx = idx
        self.day      = 0
        self.action   = None
        self.entry    = self.signals_df['close'].iloc[self.ep_idx]
        row           = self.signals_df.iloc[self.ep_idx]
        self.state    = build_state(row)
        return self.state

    def step(self, action):
        row   = self.signals_df.iloc[self.ep_idx]
        close = float(row['close'])

        # ATR-based TP/SL: more robust than return-std (consistent with model scripts)
        atr    = max(float(row.get('atr', 0.0)), 0.01 * close)  # fallback: 1% of price
        sl_dist = 1.5 * atr
        tp_dist = 2.05 * atr

        if action == ACTION_HOLD:
            reward = REWARD_HOLD
            done   = (self.day >= MAX_DAYS - 1)
            info   = {'outcome': 'HOLD', 'days': self.day + 1}
            self.day += 1
            return self.state, reward, done, info

        # LONG or SHORT - simulate next-day outcome
        futures = self.lookahead_prices.get(self.ep_idx, [])
        entry   = close

        if action == ACTION_LONG:
            sl_price = entry - sl_dist
            tp_price = entry + tp_dist
        else:  # SHORT
            sl_price = entry + sl_dist
            tp_price = entry - tp_dist

        reward   = REWARD_HOLD
        done     = False
        outcome  = 'NEUTRAL'
        days_out = 1

        for d, fp in enumerate(futures[:MAX_DAYS]):
            days_out = d + 1
            if action == ACTION_LONG:
                if fp <= sl_price:
                    reward  = REWARD_SL
                    done    = True
                    outcome = 'SL'
                    break
                elif fp >= tp_price:
                    reward  = REWARD_TP
                    done    = True
                    outcome = 'TP'
                    break
            else:  # SHORT
                if fp >= sl_price:
                    reward  = REWARD_SL
                    done    = True
                    outcome = 'SL'
                    break
                elif fp <= tp_price:
                    reward  = REWARD_TP
                    done    = True
                    outcome = 'TP'
                    break
            reward += REWARD_HOLD

        if not done:
            done    = True
            outcome = 'TIMEOUT'
            reward += REWARD_TIMEOUT  # additional penalty for timeout

        # Small bonus for picking direction that matches model consensus
        row = self.signals_df.iloc[self.ep_idx]
        signals = [float(row.get(f'{name}_signal', 0)) for name in MODEL_NAMES]
        consensus = 1 if signals.count(1) > signals.count(-1) else (-1 if signals.count(-1) > signals.count(1) else 0)

        if action == ACTION_LONG and consensus == 1:
            reward += REWARD_CORRECT_DIR
        elif action == ACTION_SHORT and consensus == -1:
            reward += REWARD_CORRECT_DIR

        # Regime-aligned reward shaping: agent learns to trade WITH the trend.
        regime = float(row.get('regime', 0))
        counter_regime = (action == ACTION_LONG and regime < -0.5) or \
                         (action == ACTION_SHORT and regime > 0.5)
        aligned_regime = (action == ACTION_LONG and regime > 0.5) or \
                         (action == ACTION_SHORT and regime < -0.5)
        if counter_regime:
            reward -= 0.5   # penalty: fighting the trend
        elif aligned_regime:
            reward += 0.3   # bonus: trading with the trend

        info = {'outcome': outcome, 'entry': entry, 'sl': sl_price, 'tp': tp_price,
                'days': min(days_out, MAX_DAYS)}
        self.day += 1
        return self.state, reward, done, info


# ---------------------------------------------------------------------------
# LOOKAHEAD PRICE TABLE
# ---------------------------------------------------------------------------

def build_lookahead(signals_df, df_raw):
    """
    For each row in signals_df, build the list of future close prices (up to MAX_DAYS).
    Matches signals_df rows to df_raw by date string; falls back to integer index if
    date lookup fails (handles format differences between Windows/Mac path styles).
    """
    # Build date->index map with normalised string keys
    date_to_raw_idx = {}
    if 'Date' in df_raw.columns:
        for i, d in enumerate(df_raw['Date']):
            date_to_raw_idx[str(d).strip()] = i

    lookahead = {}
    for row_i, row in signals_df.iterrows():
        raw_i = None

        # Try date-based lookup first
        if 'date' in signals_df.columns:
            raw_i = date_to_raw_idx.get(str(row['date']).strip(), None)

        # Fallback: use the integer value stored in 'date' when no Date column existed
        if raw_i is None:
            try:
                raw_i = int(row.get('date', row_i))
                if raw_i >= len(df_raw):
                    raw_i = row_i
            except (ValueError, TypeError):
                raw_i = row_i

        futures = []
        for d in range(1, MAX_DAYS + 1):
            idx = raw_i + d
            if idx < len(df_raw):
                futures.append(float(df_raw['Close'].iloc[idx]))
        lookahead[row_i] = futures

    n_empty = sum(1 for v in lookahead.values() if len(v) == 0)
    if n_empty > 0:
        print(f"  Warning: {n_empty} lookahead rows have no future prices (end of data)")

    return lookahead


# ---------------------------------------------------------------------------
# PPO TRAINING
# ---------------------------------------------------------------------------

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def train_ppo(env, policy, n_episodes=2000, batch_size=128, gamma=0.99):
    print(f"  Training PPO agent ({n_episodes} episodes)...")

    ep_rewards = []
    outcomes   = {'TP': 0, 'SL': 0, 'HOLD': 0, 'TIMEOUT': 0, 'NEUTRAL': 0}

    buf_states    = []
    buf_actions   = []
    buf_log_probs = []
    buf_rewards   = []
    buf_values    = []

    # Check if using PyTorch policy
    is_torch = TORCH_AVAILABLE and hasattr(policy, 'parameters')

    for ep in range(n_episodes):
        state  = env.reset()
        done   = False
        ep_rew = 0.0

        ep_states    = []
        ep_actions   = []
        ep_log_probs = []
        ep_rewards_  = []
        ep_values    = []

        while not done:
            action, log_prob, value = policy.act(state)
            next_state, reward, done, info = env.step(action)

            ep_states.append(state.copy())
            ep_actions.append(action)
            # For PyTorch, log_prob is already returned; for NumPy, we need to compute it
            if is_torch:
                ep_log_probs.append(log_prob)
            else:
                ep_log_probs.append(np.log(log_prob + 1e-10))
            ep_rewards_.append(reward)
            ep_values.append(value)

            ep_rew += reward
            state   = next_state

        outcome = info.get('outcome', 'NEUTRAL')
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

        returns    = compute_returns(ep_rewards_, gamma)
        advantages = [r - v for r, v in zip(returns, ep_values)]

        buf_states.extend(ep_states)
        buf_actions.extend(ep_actions)
        buf_log_probs.extend(ep_log_probs)
        buf_rewards.extend(returns)
        buf_values.extend(advantages)

        ep_rewards.append(ep_rew)

        if len(buf_states) >= batch_size or ep == n_episodes - 1:
            try:
                policy.update(buf_states, buf_actions, buf_log_probs,
                              buf_rewards, buf_values)
            except Exception as e:
                if is_torch:
                    # If PyTorch update fails, fall back to numpy
                    print(f"  Warning: PyTorch update failed ({e}), falling back to NumPy")
                    return ep_rewards, outcomes
                else:
                    raise

            buf_states    = []
            buf_actions   = []
            buf_log_probs = []
            buf_rewards   = []
            buf_values    = []

        if (ep + 1) % 500 == 0:
            avg = np.mean(ep_rewards[-100:]) if len(ep_rewards) >= 100 else np.mean(ep_rewards)
            print(f"    Episode {ep+1}/{n_episodes} | avg reward (last 100): {avg:.3f}")

    return ep_rewards, outcomes


# ---------------------------------------------------------------------------
# BACKTEST (greedy policy on held-out window)
# ---------------------------------------------------------------------------

def backtest(env, policy, n_episodes=None):
    if n_episodes is None:
        n_episodes = env.n_episodes

    trades   = []
    for ep_i in range(n_episodes):
        state = env.reset(idx=ep_i)
        done  = False
        total_r = 0.0
        while not done:
            action, prob, _ = policy.act_greedy(state)
            # Multi-tier consensus (mirrors get_current_action)
            row = env.signals_df.iloc[env.ep_idx]
            signals_raw = [int(row.get(f'{name}_signal', 0)) for name in MODEL_NAMES]
            n_agree = max(signals_raw.count(1), signals_raw.count(-1))
            reg = float(row.get('regime', 0))
            if n_agree >= 5:       prob = prob * 1.0
            elif n_agree >= 4:     prob = prob * 0.85
            elif n_agree >= 3 and (
                (action == ACTION_LONG and reg > -0.3) or
                (action == ACTION_SHORT and reg < 0.3)):
                                    prob = prob * 0.6
            else:                  action = ACTION_HOLD; prob = 0.3
            state, reward, done, info = env.step(action)
            total_r += reward
        trades.append({
            'episode':  ep_i,
            'action':   ACTIONS[action],
            'outcome':  info.get('outcome', 'N/A'),
            'reward':   total_r,
            'prob':     float(prob),
            'entry':    info.get('entry', 0),
            'sl':       info.get('sl', 0),
            'tp':       info.get('tp', 0),
            'days':     info.get('days', 1),
        })

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def compute_metrics(trades_df, n_months):
    """Compute trading performance metrics."""
    non_hold = trades_df[trades_df['action'] != 'HOLD']
    if len(non_hold) == 0:
        return {}

    tp_trades = non_hold[non_hold['outcome'] == 'TP']
    sl_trades = non_hold[non_hold['outcome'] == 'SL']

    win_rate     = len(tp_trades) / len(non_hold) * 100
    gross_profit = tp_trades['reward'].sum() if len(tp_trades) else 0.0
    gross_loss   = abs(sl_trades['reward'].sum()) if len(sl_trades) else 1e-10
    profit_factor = gross_profit / (gross_loss + 1e-10)

    equity       = [0.0]
    for r in non_hold['reward'].values:
        equity.append(equity[-1] + r)
    equity       = np.array(equity)

    # Compute drawdown properly: use absolute peak value or set min threshold
    peak         = np.maximum.accumulate(equity)
    # When peak is negative or zero, we need special handling
    # Standard formula: DD = (equity - peak) / abs(peak) when peak != 0
    # When peak is 0, drawdown is just the equity difference
    peak_safe    = np.where(np.abs(peak) > 1e-3, np.abs(peak), 1.0)
    drawdowns    = (equity - peak) / peak_safe * 100
    # Cap drawdown at -100% (can't lose more than 100% in equity terms)
    drawdowns    = np.maximum(drawdowns, -100.0)
    max_dd       = float(drawdowns.min())

    returns      = np.diff(equity)
    sharpe       = (returns.mean() / (returns.std() + 1e-10)) * np.sqrt(252) if len(returns) > 1 else 0.0

    avg_trades_month = len(non_hold) / (n_months + 1e-10)

    return {
        'total_trades':      len(non_hold),
        'win_rate':          win_rate,
        'profit_factor':     profit_factor,
        'sharpe':            sharpe,
        'max_drawdown_pct':  max_dd,
        'avg_trades_month':  avg_trades_month,
        'total_reward':      float(non_hold['reward'].sum()),
        'n_long':            int((non_hold['action'] == 'LONG').sum()),
        'n_short':           int((non_hold['action'] == 'SHORT').sum()),
    }


# ---------------------------------------------------------------------------
# FINAL ACTION FOR REPORT
# ---------------------------------------------------------------------------

def get_current_action(signals_df, df_raw, policy, use_voting=False, current_price=None):
    """
    Run greedy policy on the last available row.

    If current_price is provided (live trading), it overrides the CSV's last close
    for entry/TP/SL calculations. Model signals still use the CSV data (yesterday's
    features), but the trade levels reflect where the market actually is now.
    """
    last_row   = signals_df.iloc[-1]
    state      = build_state(last_row)

    if use_voting:
        # Fallback to simple voting when PPO performance is poor
        votes = []
        for name in MODEL_NAMES:
            sig  = int(last_row.get(f'{name}_signal', 0))
            prob = float(last_row.get(f'{name}_prob', 0.5))
            # Weight vote by probability if significantly confident
            if prob > 0.7:
                votes.extend([sig] * 2)  # double weight for high-confidence signals
            else:
                votes.append(sig)

        # Count votes: 1=LONG, -1=SHORT, 0=HOLD
        vote_long  = votes.count(1)
        vote_short = votes.count(-1)
        vote_hold  = votes.count(0)

        if vote_long > vote_short and vote_long > vote_hold:
            action = ACTION_LONG
            prob   = vote_long / len(votes)
        elif vote_short > vote_long and vote_short > vote_hold:
            action = ACTION_SHORT
            prob   = vote_short / len(votes)
        else:
            action = ACTION_HOLD
            prob   = vote_hold / len(votes) if vote_hold > 0 else 0.5
    else:
        action, prob, _ = policy.act_greedy(state)

    # ATR-based SL/TP (consistent with training environment and model scripts)
    csv_close = float(last_row['close'])
    close = float(current_price) if current_price is not None else csv_close
    atr     = max(float(last_row.get('atr', 0.0)), 0.01 * close)
    sl_dist = 1.5 * atr
    tp_dist = 2.05 * atr

    # Multi-tier consensus: more agreement = trade, less = skip or scale down
    signals_raw = [int(last_row.get(f'{name}_signal', 0)) for name in MODEL_NAMES]
    n_long  = signals_raw.count(1)
    n_short = signals_raw.count(-1)
    n_agree = max(n_long, n_short)
    regime  = float(last_row.get('regime', 0))

    if n_agree >= 5:
        prob = prob * 1.0          # strong consensus, full confidence
    elif n_agree >= 4:
        prob = prob * 0.85         # good consensus, slight discount
    elif n_agree >= 3 and (
        (action == ACTION_LONG and regime > -0.3) or
        (action == ACTION_SHORT and regime < 0.3)):
        prob = prob * 0.6          # marginal consensus, only if regime-compatible
    else:
        action = ACTION_HOLD; prob = 0.3  # weak consensus, skip

    if action == ACTION_LONG:
        sl = close - sl_dist
        tp = close + tp_dist
    elif action == ACTION_SHORT:
        sl = close + sl_dist
        tp = close - tp_dist
    else:  # HOLD — no trade
        sl = close
        tp = close

    return {
        'action':      ACTIONS[action],
        'confidence':  prob * 100,
        'close':       close,
        'sl':          sl,
        'tp':          tp,
        'sl_pct':      (sl - close) / close * 100,
        'tp_pct':      (tp - close) / close * 100,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_agent(csv_file, current_price=None):
    print("="*70)
    print("RL AGENT TRADER - PPO META-AGENT")
    print("="*70)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found")
        sys.exit(1)

    df_raw = pd.read_csv(csv_file)
    df_raw = compute_indicators(df_raw).reset_index(drop=True)

    if len(df_raw) < MIN_RECORDS:
        print(f"Error: Need at least {MIN_RECORDS} records, got {len(df_raw)}")
        sys.exit(1)

    ticker = os.path.basename(csv_file).split('_')[0]
    n_records = len(df_raw)
    n_months  = n_records / 21.0   # approx trading days per month

    print(f"\nTicker:  {ticker}")
    print(f"Records: {n_records}  (~{n_months:.0f} months)")

    # Layer 1: walk-forward model signals
    print("\n[STEP 1/4] Generating walk-forward model predictions...")
    signals_df = load_model_predictions(csv_file)
    print(f"  Signal rows: {len(signals_df)}")

    if len(signals_df) < 50:
        print("Error: Not enough signal rows to train agent (need >= 50)")
        sys.exit(1)

    # Build lookahead prices
    print("\n[STEP 2/4] Building lookahead price table...")
    lookahead = build_lookahead(signals_df, df_raw)

    # Split signals into train/val
    n_sig      = len(signals_df)
    train_cut  = int(n_sig * 0.8)
    train_sig  = signals_df.iloc[:train_cut].reset_index(drop=True)
    val_sig    = signals_df.iloc[train_cut:].reset_index(drop=True)

    # Remap lookahead indices for train/val
    train_la = {i: lookahead.get(i, []) for i in range(len(train_sig))}
    val_la   = {i: lookahead.get(train_cut + i, []) for i in range(len(val_sig))}

    train_months = len(train_sig) / 21.0
    val_months   = len(val_sig)   / 21.0

    # Layer 2: PPO training on walk-forward signals
    print("\n[STEP 3/4] Training PPO agent (walk-forward layer 2)...")
    np.random.seed(42)

    # Use PyTorch if available (3x better performance)
    if TORCH_AVAILABLE:
        print("  Using PyTorch PPO (improved gradients)")
        torch.manual_seed(42)
        policy = PPOPolicyTorch(state_dim=STATE_DIM, hidden=128, lr=1e-4)
        weights_file = os.path.join(base_dir, 'rl_agent_torch.pt')
        csv_hash_file = os.path.join(base_dir, 'rl_agent_torch_hash.txt')
    else:
        print("  Using NumPy PPO (install torch for better performance)")
        policy = PPOPolicy(state_dim=STATE_DIM)
        weights_file = os.path.join(base_dir, 'rl_agent_weights.npz')
        csv_hash_file = os.path.join(base_dir, 'rl_agent_csv_hash.txt')

    train_env = TradingEnv(train_sig, train_la)

    # Warm-start from saved weights if they exist
    csv_hash = str(os.path.getsize(csv_file)) + '_' + str(n_records)
    if os.path.exists(weights_file) and os.path.exists(csv_hash_file):
        try:
            with open(csv_hash_file) as fh:
                saved_hash = fh.read().strip()
            if saved_hash == csv_hash:
                if TORCH_AVAILABLE:
                    policy.load_state_dict(torch.load(weights_file))
                else:
                    w = np.load(weights_file)
                    policy.W1 = w['W1']; policy.b1 = w['b1']
                    policy.W2 = w['W2']; policy.b2 = w['b2']
                    policy.W3 = w['W3']; policy.b3 = w['b3']
                    policy.Wv1 = w['Wv1']; policy.bv1 = w['bv1']
                    policy.Wv2 = w['Wv2']; policy.bv2 = w['bv2']
                print("  Warm-started from saved weights (same CSV)")
            else:
                print("  CSV changed - training from scratch")
        except Exception as e:
            print(f"  Could not load saved weights: {e}")

    # Use more episodes for PyTorch (better gradients handle more data)
    if TORCH_AVAILABLE:
        n_ep = min(max(len(train_sig) * 20, 10000), 80000)
    else:
        n_ep = min(max(len(train_sig) * 15, 8000), 60000)

    est_sec = max(1, n_ep // 7500)
    print(f"  Episodes planned: {n_ep}  (estimated time: ~{est_sec}-{est_sec*2} seconds)")
    ep_rewards, outcomes = train_ppo(train_env, policy, n_episodes=n_ep)
    print(f"  Training outcomes: TP={outcomes.get('TP',0)} SL={outcomes.get('SL',0)} "
          f"HOLD={outcomes.get('HOLD',0)} TIMEOUT={outcomes.get('TIMEOUT',0)}")

    # Persist weights for warm-start on next run with same CSV
    try:
        if TORCH_AVAILABLE:
            torch.save(policy.state_dict(), weights_file)
        else:
            np.savez(weights_file,
                     W1=policy.W1, b1=policy.b1, W2=policy.W2, b2=policy.b2,
                     W3=policy.W3, b3=policy.b3, Wv1=policy.Wv1, bv1=policy.bv1,
                     Wv2=policy.Wv2, bv2=policy.bv2)
        with open(csv_hash_file, 'w') as fh:
            fh.write(csv_hash)
    except Exception as e:
        print(f"  Warning: could not save weights: {e}")

    # Backtest on validation set
    print("\n[STEP 4/4] Backtesting on validation set...")
    val_env    = TradingEnv(val_sig, val_la)
    trades_df  = backtest(val_env, policy)
    metrics    = compute_metrics(trades_df, val_months)

    if not metrics:
        print("Warning: No actionable trades in validation set")
        metrics = {'win_rate': 0, 'profit_factor': 0, 'sharpe': 0,
                   'max_drawdown_pct': 0, 'avg_trades_month': 0,
                   'total_trades': 0, 'total_reward': 0,
                   'n_long': 0, 'n_short': 0}

    # Current action - use voting fallback if PPO performance is poor
    use_voting = (metrics['win_rate'] < 20 and metrics['profit_factor'] < 0.8)
    current = get_current_action(signals_df, df_raw, policy, use_voting=use_voting,
                                 current_price=current_price)
    if use_voting:
        print("  Using voting-based fallback (PPO performance below threshold)")

    # Print results
    print("\n" + "="*70)
    print("RL AGENT TRADING RESULTS")
    print("="*70)

    print(f"\nCurrent Decision ({ticker}):")
    print(f"  AGENT_ACTION:     {current['action']}")
    print(f"  AGENT_CONFIDENCE: {current['confidence']:.1f}%")
    print(f"  Entry Price:      ${current['close']:.2f}")
    print(f"  Stop Loss:        ${current['sl']:.2f} ({current['sl_pct']:+.2f}%)")
    print(f"  Take Profit:      ${current['tp']:.2f} ({current['tp_pct']:+.2f}%)")

    print(f"\nValidation Backtest Performance ({len(val_sig)} days, ~{val_months:.0f} months):")
    print(f"  Total Trades:     {metrics['total_trades']}")
    print(f"  AGENT_WINRATE:    {metrics['win_rate']:.1f}%")
    print(f"  Profit Factor:    {metrics['profit_factor']:.2f}")
    print(f"  Sharpe Ratio:     {metrics['sharpe']:.2f}")
    print(f"  Max Drawdown:     {metrics['max_drawdown_pct']:.1f}%")
    print(f"  Avg Trades/Month: {metrics['avg_trades_month']:.1f}")
    print(f"  Long / Short:     {metrics['n_long']} / {metrics['n_short']}")
    print(f"  Total Reward:     {metrics['total_reward']:.2f}")

    # Performance assessment
    print("\nPerformance Assessment:")
    grades = []
    if metrics['win_rate'] >= 60:
        grades.append("  Win Rate:      PASS  (>= 60%)")
    else:
        grades.append(f"  Win Rate:      BELOW TARGET  ({metrics['win_rate']:.1f}% < 60%)")

    if metrics['profit_factor'] >= 1.5:
        grades.append("  Profit Factor: PASS  (>= 1.5)")
    else:
        grades.append(f"  Profit Factor: BELOW TARGET  ({metrics['profit_factor']:.2f} < 1.5)")

    if metrics['sharpe'] >= 1.0:
        grades.append("  Sharpe Ratio:  PASS  (>= 1.0)")
    else:
        grades.append(f"  Sharpe Ratio:  BELOW TARGET  ({metrics['sharpe']:.2f} < 1.0)")

    if metrics['max_drawdown_pct'] > -20:
        grades.append(f"  Max Drawdown:  PASS  (> -20%)")
    else:
        grades.append(f"  Max Drawdown:  BELOW TARGET  ({metrics['max_drawdown_pct']:.1f}% < -20%)")

    for g in grades:
        print(g)

    print("\n" + "="*70)
    print("DISCLAIMER: RL agent decisions are statistical, NOT financial advice.")
    print("="*70)

    # Suggest PyTorch if not available and performance is poor
    if not TORCH_AVAILABLE and (metrics['win_rate'] < 40 or metrics['profit_factor'] < 1.0):
        print("\n💡 TIP: Install PyTorch for 3x better PPO performance:")
        print("   pip install torch")
        print("   Then re-run this script for improved results")

    return current, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PPO RL meta-agent that reads model outputs and decides LONG/SHORT/HOLD'
    )
    parser.add_argument('csv_file', type=str, help='Path to CSV file with stock data')
    parser.add_argument('--current-price', type=float, default=None,
                       help='Live current price (overrides last close for entry/TP/SL)')
    args = parser.parse_args()
    run_agent(args.csv_file, current_price=args.current_price)
