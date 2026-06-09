# Stock Market Entry Signal

7-model ensemble with PPO reinforcement learning agent for daily swing trading.

**Features:**
- 7 models (XGBoost, XGBoost-Heavy, LightGBM, LightGBM-Heavy, RandomForest, RF-Heavy, CatBoost-Bayes)
- PPO RL meta-agent with 34-dim state vector (model signals + chart indicators + regime)
- Multi-tier consensus filter with regime-aware trade blocking
- Position sizing by confidence tier
- Next-day return prediction with walk-forward validation
- ~50 technical indicators in heavy models (incl. ADX, Choppiness, Coppock, AO, DPO)
- LSTM-BO-CatBoost hybrid model (Sun & Tian 2023)
- HTML dashboard reports with embedded prediction plots
- Stock ranking tool (`rank_stocks.py`)
- Live recount tool (`recount.py`) for real-time prediction with current price
- Live price override (`--current-price`) and leverage P&L (`--leverage`)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline on a ticker (trains all 7 models + RL agent)
python main.py --ticker AAPL

# With live price (for live trading)
python main.py --ticker AAPL --current-price 312.50

# Live recount: predict using already-trained models + current price
python recount.py --ticker AAPL --current-price 312.50 --leverage 5

# Rank all stocks in target_stock.txt
python rank_stocks.py
```

## Architecture

```
target_stock.txt          →  rank_stocks.py  →  stock-ranking-result.txt
fetch_stock_data.py        →  fetches OHLCV  →  {TICKER}_daily_data.csv
main.py                    →  orchestrates   →  REPORT/RESULT-{TICKER}-{DATE}.html
  ├── train_xgboost.py            (5 OHLCV + lags, 2000 trees)
  ├── train_xgboost_heavy.py      (~50 indicators, 5000 trees)
  ├── train_lightgbm.py           (5 OHLCV + lags, 2000 trees)
  ├── train_lightgbm_heavy.py     (~50 indicators, 5000 trees)
  ├── train_randomforest.py       (1000 trees, 5-fold walk-forward)
  ├── train_randomforest_heavy.py (1500 trees, depth 20, 7-fold walk-forward)
  ├── train_catboost_bayes.py     (LSTM features + Bayesian opt. CatBoost)
  ├── agent_trader.py             (PPO RL, 34-dim state, consensus filter → {TICKER}_{YYYYMMDD}_rl_agent_torch.pt)
  └── recount.py                  (live prediction from saved MODELS/)
  
MODELS/                     →  persists trained pkl/scaler/features for recount.py
model_store.py              →  shared path helpers for MODELS/ naming + find_latest_rl_weights()
```

## recount.py — Live Trading Prediction

`recount.py` uses **already-trained models** from `MODELS/` to generate a fresh
BUY/SHORT/HOLD decision at the current market price without re-training.

**Why recount?** The CSV data is lagged (yesterday's close), but as a trader you
know the current live price. recount loads all 7 saved models + the RL agent policy,
fetches the last 12 months of data for indicator computation, runs every model on
the latest row, and feeds their signals into the RL agent — all using the live price
for entry/TP/SL while keeping model predictions strictly from historical features
(no look-ahead).

**Workflow:**
1. Verify `MODELS/` has a complete model set for the requested ticker
2. Fetch recent data (default 12 months) and compute 50+ indicators
3. Load all 7 pkl models + scalers + feature lists
4. Predict next-day return from each model → signal (BUY/SHORT/HOLD) + probability
5. Build 34-dim RL state vector and run PPO policy (or voting fallback)
6. Apply multi-tier consensus filter with regime check
7. Calculate TP/SL at the live current price, show 5x leverage P&L

**Usage:**
```bash
# Requires models already trained for this ticker
python recount.py --ticker MSFT --current-price 441.31
python recount.py --ticker MSFT --current-price 441.31 --leverage 5
python recount.py --ticker AAPL --current-price 312.50 --leverage 5 --months 6

# If models aren't trained yet, you'll get a clear error:
# ERROR: No trained models found for 'AAPL' in MODELS/.
# You must train the models first. Run:
#   python main.py --ticker AAPL
```

**Output example:**
```
INDIVIDUAL MODEL SIGNALS
Model                    Signal              Move %    Prob
xgboost                  SHORT (SELL)        -1.22%   74.5%
xgboost_heavy            HOLD (No signal)    +0.08%   51.6%
...
Consensus                LONG=0 SHORT=3 HOLD=4  (agree=3/7)

RECOUNT DECISION
  >>> ACTION:  HOLD (AMBER)
  >>> Confidence: 30.0%

  Entry Price:    $441.31
  Stop Loss:      $441.31  (+0.00%)
  Take Profit:    $441.31  (+0.00%)

  --- 5x Leverage Position P&L ---
  Stop Loss P&L:    +0.0%
  Take Profit P&L:  +0.0%

  Note: CSV last close was $416.67, current price is $441.31 (+5.91%)
```

## Current Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Prediction | Next-day return | `pct_change().shift(-1) * 100` |
| TP | 1.5 × ATR | Take profit distance |
| SL | 1.0 × ATR | Stop loss distance |
| Ratio | 1.5:1 | TP / SL |
| Break-even | 40% | Minimum winrate to profit |
| Consensus | Multi-tier (3-7/7) | Regime-aware at 4/7, strict at 3/7 |
| Position size | 100% → 75% → 50% → 0% | By consensus tier |
| State dim | 34 | 7 models × 2 + 20 indicators |

## Key Design Decisions

See [LEARN.md](LEARN.md) for the complete study guide including FAQ (Section 23).
- **1-day prediction** (not multi-day): less error compounding, faster feedback
- **TP > SL** (non-negotiable): 1.5:1 reward/risk ratio
- **7 models**: 3 families (XGBoost, LightGBM, RandomForest) in light/heavy + CatBoost-Bayes hybrid
- **Tree models dominate**: standalone LSTM/TFT removed (45.8% accuracy, below coin-flip)
- **KNN removed**: always predicted ~0% return (market efficiency)
- **AdaBoost/CatBoost removed**: underperformed; only BO-CatBoost hybrid survived
- **Return prediction** (not price): scale-invariant target, 6x better MAE
- **NumPy PPO full backprop**: all layer gradients (W1/W2/W3 + value head) are computed from pre-update weights before any weight is modified; corrupted ordering would break hidden-layer learning
- **Signal dtype**: signals stored as `int` for vote counting, cast to `float` only when appended to the numpy state vector -- prevents silent `float.count()` equality bugs
- **Directional consensus (n_dir not n_agree)**: consensus filter counts models voting FOR the chosen direction, not the global max across both sides; `n_agree = max(n_long, n_short)` would pass a SHORT trade when 5 models say LONG
