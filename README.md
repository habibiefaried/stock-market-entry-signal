# Stock Market Entry Signal

6-model ensemble with PPO reinforcement learning agent for daily swing trading.

**Features:**
- 6 tree models (XGBoost, XGBoost-Heavy, LightGBM, LightGBM-Heavy, RandomForest, RF-Heavy)
- PPO RL meta-agent with 31-dim state vector (model signals + chart indicators + regime)
- Multi-tier consensus filter with regime-aware trade blocking
- Position sizing by confidence tier
- Next-day return prediction with walk-forward validation
- 45 technical indicators in heavy models (incl. ADX, Choppiness, Coppock, AO, DPO)
- HTML dashboard reports with embedded prediction plots
- Stock ranking tool (`rank_stocks.py`)
- Live price override (`--current-price`) and leverage P&L (`--leverage`)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline on a ticker
python main.py --ticker AAPL

# With live price (for live trading)
python main.py --ticker AAPL --current-price 312.50

# With leverage P&L display
python main.py --ticker AAPL --leverage 5

# Rank all stocks in target_stock.txt
python rank_stocks.py

# RL agent standalone (requires trained models)
python agent_trader.py AAPL_daily_data.csv --current-price 312.50 --leverage 5
```

## Architecture

```
target_stock.txt          →  rank_stocks.py  →  stock-ranking-result.txt
fetch_stock_data.py        →  fetches OHLCV  →  {TICKER}_daily_data.csv
main.py                    →  orchestrates   →  REPORT/RESULT-{TICKER}-{DATE}.html
  ├── train_xgboost.py            (20 features, 2000 trees)
  ├── train_xgboost_heavy.py      (45 indicators, 3000 trees)
  ├── train_lightgbm.py           (20 features, 2000 trees)
  ├── train_lightgbm_heavy.py     (45 indicators, 3000 trees)
  ├── train_randomforest.py       (1000 trees, walk-forward)
  ├── train_randomforest_heavy.py (1500 trees, depth 20, 7-fold walk-forward)
  └── agent_trader.py             (PPO RL, 31-dim state, consensus filter)
```

## Current Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Prediction | Next-day return | `pct_change().shift(-1) * 100` |
| TP | 1.5 × ATR | Take profit distance |
| SL | 1.0 × ATR | Stop loss distance |
| Ratio | 1.5:1 | TP / SL |
| Break-even | 40% | Minimum winrate to profit |
| Consensus | Multi-tier (3-6/6) | Regime-aware at 4/6, strict at 3/6 |
| Position size | 100% → 75% → 50% → 0% | By consensus tier |
| State dim | 31 | 6 models × 2 + 19 indicators |

## Performance (10-stock benchmark, 7 years data)

| Stock | Winrate | Profit Factor | Confidence |
|-------|---------|---------------|------------|
| WMT | 55.0% | 3.33 | 97.9% |
| INTC | 51.9% | 3.67 | 80.7% |
| AMZN | 50.0% | 2.91 | 30.0% (HOLD) |
| ADBE | 49.5% | 2.75 | 30.0% (HOLD) |
| MSFT | 45.2% | 3.55 | 91.7% |
| VZ | 44.4% | 3.23 | 30.0% (HOLD) |
| AMD | 42.1% | 3.24 | 50.5% |
| NVDA | 40.8% | 1.90 | 90.0% |
| META | 36.8% | 2.02 | 95.0% |
| PFE | 34.2% | 2.02 | 30.0% (HOLD) |

**Average: 45.0% winrate, 2.86 profit factor. All 10 stocks profitable.**

Full rankings with SL%/TP% at `stock-ranking-result.txt`.

## Key Design Decisions

See [LEARN.md](LEARN.md) for the complete study guide including FAQ (Section 23).
- **1-day prediction** (not multi-day): less error compounding, faster feedback
- **TP > SL** (non-negotiable): 1.5:1 reward/risk ratio
- **6 models** (not more): AdaBoost/CatBoost tested and removed (diluted consensus)
- **Tree models only**: LSTM/TFT removed (45.8% accuracy, below coin-flip)
- **KNN removed**: always predicted ~0% return (market efficiency)
- **Return prediction** (not price): scale-invariant target, 6× better MAE
