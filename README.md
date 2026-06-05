# Stock Market Entry Signal

7-model ensemble with PPO reinforcement learning agent for daily swing trading.

**Features:**
- 7 models (XGBoost, XGBoost-Heavy, LightGBM, LightGBM-Heavy, RandomForest, RF-Heavy, CatBoost-Bayes)
- PPO RL meta-agent with 33-dim state vector (model signals + chart indicators + regime)
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
  ├── agent_trader.py             (PPO RL, 33-dim state, consensus filter)
  └── recount.py                  (live prediction from saved MODELS/)
  
MODELS/                     →  persists trained pkl/scaler/features for recount.py
model_store.py              →  shared path helpers for MODELS/ naming
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
| State dim | 33 | 7 models × 2 + 19 indicators |

## Key Design Decisions

See [LEARN.md](LEARN.md) for the complete study guide including FAQ (Section 23).
- **1-day prediction** (not multi-day): less error compounding, faster feedback
- **TP > SL** (non-negotiable): 1.5:1 reward/risk ratio
- **7 models**: 3 families (XGBoost, LightGBM, RandomForest) in light/heavy + CatBoost-Bayes hybrid
- **Tree models dominate**: standalone LSTM/TFT removed (45.8% accuracy, below coin-flip)
- **KNN removed**: always predicted ~0% return (market efficiency)
- **AdaBoost/CatBoost removed**: underperformed; only BO-CatBoost hybrid survived
- **Return prediction** (not price): scale-invariant target, 6× better MAE
