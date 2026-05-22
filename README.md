# Stock Market Entry Signal

Advanced multi-model stock price prediction system with probability analysis for swing trading.

**Features:**
- 🤖 4 Machine Learning Models (LSTM, XGBoost, LightGBM, RandomForest)
- 🎯 Multi-Approach Win Probability Analysis (3 independent methods)
- 📊 Beautiful HTML Dashboard Reports
- 💹 5x Leverage Position Calculations for IQ Option
- ⚡ Parallel Training with GPU Acceleration
- 🔄 Walk-Forward Validation (RandomForest)

## Installation

### Prerequisites

- **pyenv** for Python version management  
- **Python 3.11.9** (required for TensorFlow-DirectML GPU support)

### Setup

**Step 1: Verify Python version**
```bash
# Navigate to project directory
cd C:\Users\Davis\Documents\code\stock-market-entry-signal

# Python 3.11.9 should already be set (via .python-version file)
python --version  # Should show Python 3.11.9
```

**Step 2: Install core dependencies**
```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

### Check GPU/CUDA Availability

Run the GPU test script to verify your setup:

```bash
python test_gpu.py
```

**What it checks:**
- PyTorch GPU support (LSTM uses Keras with PyTorch backend)
- XGBoost CUDA support
- System GPU via nvidia-smi

**Expected GPU Training Times (6 months data):**
- LSTM: ~1-2 minutes
- XGBoost: ~30-60 seconds

## Quick Start

### Option 1: Streamlined Approach (Recommended)

Fetch data and train all models in one command:

```bash
# Fetch 4 years of MSFT data and train all models
python main.py --ticker MSFT

# Fetch 5 years of data
python main.py --ticker MSFT --months 60

# Cryptocurrency example
python main.py --ticker BTC-USD --months 48
```

This will:
1. Fetch historical stock data (default: 48 months = 4 years)
2. Train all 4 models in parallel (LSTM, XGBoost, LightGBM, RandomForest)
3. Run multi-approach probability analysis for each model
4. Generate beautiful HTML dashboard: `RESULT-{TICKER}-{DATE}.html`

**Training Details:**
- LSTM, XGBoost, LightGBM: 90/10 split (90% training, 10% testing)
- RandomForest: Walk-forward validation (5 folds, rolling windows)
- With 48 months data: ~43 months training, ~5 months testing
- Training time: 3-7 minutes (depending on GPU availability)

### Option 2: Manual Steps

**Step 1: Fetch data separately**
```bash
python fetch_stock_data.py MSFT --months 48
```

**Step 2: Train all models**
```bash
python main.py MSFT_daily_data_20260520.csv
```

## Usage

### Basic Command

```bash
python fetch_stock_data.py <TICKER> --months <NUMBER_OF_MONTHS>
```

### Parameters

- `ticker` (required): The ticker symbol for the asset you want to fetch
- `--months` (optional): Number of months of historical data (default: 48 = 4 years)

## Examples

### Stock Market Data

```bash
# Microsoft (48 months - default)
python fetch_stock_data.py MSFT

# Microsoft (60 months = 5 years)
python fetch_stock_data.py MSFT --months 60

# Apple (48 months)
python fetch_stock_data.py AAPL

# Google (36 months)
python fetch_stock_data.py GOOGL --months 36

# Tesla (48 months)
python fetch_stock_data.py TSLA --months 48

# Amazon (48 months)
python fetch_stock_data.py AMZN --months 48

# NVIDIA (48 months)
python fetch_stock_data.py NVDA --months 48
```

### Cryptocurrency Data

```bash
# Bitcoin (48 months)
python fetch_stock_data.py BTC-USD --months 48

# Ethereum (48 months)
python fetch_stock_data.py ETH-USD --months 48

# Solana (24 months)
python fetch_stock_data.py SOL-USD --months 24

# Cardano (48 months)
python fetch_stock_data.py ADA-USD --months 48

# Dogecoin (48 months)
python fetch_stock_data.py DOGE-USD --months 48

# Ripple (48 months)
python fetch_stock_data.py XRP-USD --months 48

# Polkadot (48 months)
python fetch_stock_data.py DOT-USD --months 48
```

### Other Assets

```bash
# S&P 500 Index
python fetch_stock_data.py ^GSPC --months 6

# Gold Futures
python fetch_stock_data.py GC=F --months 6

# Crude Oil Futures
python fetch_stock_data.py CL=F --months 6

# EUR/USD Forex
python fetch_stock_data.py EURUSD=X --months 6
```

## Output

The script generates a CSV file named `{TICKER}_daily_data_{DATE}.csv` containing:

### OHLCV Data (Price & Volume)
- **Date**: Trading date
- **Open**: Opening price
- **High**: Highest price of the day
- **Low**: Lowest price of the day
- **Close**: Closing price
- **Volume**: Trading volume

### Technical Indicators (For Trading Signals)

**Note:** These indicators are saved in the CSV but NOT used for model training.
Models only use OHLCV data. Technical indicators are for implementing separate trading rules/signals.

#### Moving Averages (MA)
- **MA_5, MA_10, MA_20, MA_50, MA_200**: Moving averages for trend analysis

#### Momentum Indicators
- **RSI_14**: Relative Strength Index (overbought/oversold)
- **MACD, MACD_Signal, MACD_Hist**: MACD indicators

#### Volatility Indicators
- **BB_Upper, BB_Middle, BB_Lower**: Bollinger Bands

#### Volume Indicators
- **Volume_MA_20**: 20-day volume moving average

## Data Split for Training

For machine learning models:
- **36 months of data**: ~756 trading days
  - Training: ~630 days (30 months)
  - Testing: ~126 days (6 months)

## Training Models

After fetching the data, train prediction models using machine learning algorithms.

**Model Types:**
- **Deep Learning Models** (LSTM): Neural networks that learn sequential patterns
- **Gradient Boosting Models** (LightGBM, XGBoost): Tree-based ensemble methods

**Training Data:** All models use **OHLCV** features (Open, High, Low, Close, Volume).
**Train/Test Split:** 90/10 (90% training, 10% testing) - with 48 months data, this gives ~43 months for training and ~5 months for testing.

### LSTM Model (Deep Learning)

Train an LSTM model using Keras:

```bash
# Basic usage (default parameters)
python train_lstm.py MSFT_daily_data_20260519.csv

# Custom parameters
python train_lstm.py MSFT_daily_data_20260519.csv --lookback 60 --epochs 100 --batch_size 32
```

**Parameters:**
- `--lookback`: Number of days to look back for sequences (default: 60)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 32)

**Understanding LSTM Parameters:**

1. **Lookback (Sequence Length)**
   - How many past days the model uses to predict the next day
   - `--lookback 60` means: use 60 days of history to predict day 61
   - **Too small (e.g., 5-10)**: Model can't learn long-term patterns
   - **Too large (e.g., 100+)**: May not have enough training data, slower training
   - **Recommended**: 30-60 days for daily stock data

2. **Epochs**
   - Number of times the model sees the entire training dataset
   - `--epochs 100` means: train for 100 complete passes through data
   - **Too few (e.g., 10-20)**: Model underfits, poor predictions
   - **Too many (e.g., 500+)**: Model overfits, memorizes noise
   - **Note**: Early stopping automatically stops if no improvement for 10 epochs
   - **Recommended**: 50-100 epochs (early stopping will handle overfitting)

3. **Batch Size**
   - Number of samples processed before updating model weights
   - `--batch_size 32` means: process 32 sequences, then update weights
   - **Smaller (e.g., 8-16)**: More frequent updates, better for small datasets, slower training
   - **Larger (e.g., 64-128)**: Faster training, more stable gradients, needs more memory
   - **Recommended**: 16-32 for small datasets, 32-64 for larger datasets

**LSTM Model Architecture (Ultra-Simplified):**

The model uses 1 LSTM layer for maximum simplicity:
- **Layer 1**: 50 LSTM units (learns patterns from sequences)
- **Dropout**: 40% after LSTM layer (strong regularization)
- **Output Layer**: 1 unit (predicted closing price)
- **Total Parameters**: ~15K (90% reduction from original 130K)
- **Benefits**: Fastest training (<1 min), minimal overfitting, best for noisy financial data

**How LSTM Works for Stock Prediction:**

1. Takes a sequence of past days (lookback period)
2. Each day contains all features (Open, High, Low, Close, Volume, technical indicators)
3. LSTM "remembers" important patterns across the sequence
4. Predicts the next day's closing price
5. Example: Days 1-60 → Predict Day 61, Days 2-61 → Predict Day 62, etc.

**Outputs:**
- `best_lstm_model.keras`: Trained LSTM model (saved with best validation performance)
- `lstm_model_info.txt`: Model performance metrics
- `lstm_training_history.png`: Training/validation loss plot
- `lstm_predictions.png`: Actual vs predicted prices plot

### LightGBM Model (Gradient Boosting - Recommended)

Train a LightGBM model:

```bash
# Basic usage (default parameters)
python train_lightgbm.py MSFT_daily_data_20260520.csv

# Custom parameters
python train_lightgbm.py MSFT_daily_data_20260520.csv --n_estimators 1000 --learning_rate 0.01 --num_leaves 31
```

**Parameters:**
- `--n_estimators`: Number of trees (default: 1000)
- `--learning_rate`: Learning rate (default: 0.01)
- `--num_leaves`: Maximum leaves per tree (default: 31)

**Understanding LightGBM Parameters:**

1. **N_estimators (Number of Trees)**
   - How many decision trees to build sequentially
   - `--n_estimators 1000` means: build 1000 trees, each correcting previous errors
   - Same as XGBoost's n_estimators
   - **Recommended**: 500-1000 trees

2. **Learning Rate (Shrinkage)**
   - Controls how much each tree contributes to final prediction
   - `--learning_rate 0.01` means: each tree contributes 1% of its prediction
   - Same concept as XGBoost
   - **Recommended**: 0.01-0.05 for stock prediction

3. **Num Leaves (LightGBM Specific)**
   - Maximum number of leaves in one tree
   - `--num_leaves 31` means: tree can have up to 31 leaf nodes
   - **Different from XGBoost's max_depth!**
   - Roughly equivalent to `2^max_depth` in XGBoost
   - **Shallow (e.g., 15-31)**: Simple patterns, prevents overfitting
   - **Deep (e.g., 63-127)**: Complex patterns, may overfit
   - **Recommended**: 15-31 for financial data

**LightGBM vs XGBoost Key Differences:**

1. **Tree Growth Strategy:**
   - XGBoost: Level-wise (balanced tree, slower but safer)
   - LightGBM: Leaf-wise (splits leaf with max loss reduction - faster, often more accurate)

2. **Speed:**
   - LightGBM is 2-4x faster on large datasets
   - Uses histogram-based algorithms

3. **Memory:**
   - LightGBM uses less memory (more efficient)

4. **Accuracy:**
   - Often better out-of-the-box accuracy
   - Less hyperparameter tuning needed

**How LightGBM Works for Stock Prediction:**

1. **Feature Engineering**: Same as XGBoost - creates lag features:
   - `Close_lag_1, Close_lag_2, ...`: Previous days' closing prices
   - `Volume_lag_1, Volume_lag_2, ...`: Previous days' volumes
   - `Price_change_1d, Price_change_5d`: Price change percentages
   - `Volatility_5d, Volatility_10d`: Rolling price volatility

2. **Leaf-wise Tree Building**:
   - Finds leaf with maximum delta loss
   - Splits that leaf (vs XGBoost splitting all leaves at same level)
   - More efficient path to optimal tree structure

3. **Final Prediction**: Sum of all tree predictions × learning_rate

4. **Feature Importance**: Shows which features matter most (saved as plot)

**Advantages of LightGBM:**
- Faster training than XGBoost (especially on large datasets)
- Often better accuracy with default parameters
- Less memory usage
- Interpretable (can see which features are important)
- GPU support (CUDA)

**Outputs:**
- `lightgbm_model.pkl`: Trained LightGBM model
- `lightgbm_scaler.pkl`: Feature scaler (for standardizing inputs)
- `lightgbm_features.txt`: List of all features used (including lag features)
- `lightgbm_model_info.txt`: Model performance metrics
- `lightgbm_feature_importance.png`: Feature importance plot (shows what model focuses on)
- `lightgbm_predictions.png`: Actual vs predicted prices plot

### XGBoost Model (Gradient Boosting - Alternative)

Train an XGBoost model:

```bash
# Basic usage (default parameters)
python train_xgboost.py MSFT_daily_data_20260519.csv

# Custom parameters
python train_xgboost.py MSFT_daily_data_20260519.csv --n_estimators 1000 --learning_rate 0.01 --max_depth 7
```

**Parameters:**
- `--n_estimators`: Number of trees (default: 1000)
- `--learning_rate`: Learning rate (default: 0.01)
- `--max_depth`: Maximum tree depth (default: 7)

**Understanding XGBoost Parameters:**

1. **N_estimators (Number of Trees)**
   - How many decision trees to build sequentially
   - `--n_estimators 1000` means: build 1000 trees, each correcting previous errors
   - **Too few (e.g., 50-100)**: Model underfits, misses patterns
   - **Too many (e.g., 5000+)**: Slower training, diminishing returns
   - **Note**: Model tracks validation performance and you can stop early if needed
   - **Recommended**: 500-1000 trees

2. **Learning Rate (Shrinkage)**
   - Controls how much each tree contributes to final prediction
   - `--learning_rate 0.01` means: each tree contributes 1% of its prediction
   - **Higher (e.g., 0.1-0.3)**: Faster learning, risk of overfitting, fewer trees needed
   - **Lower (e.g., 0.001-0.01)**: Slower learning, better generalization, more trees needed
   - **Trade-off**: Lower learning rate + more trees = better performance but slower
   - **Recommended**: 0.01-0.05 for stock prediction

3. **Max Depth**
   - Maximum depth of each decision tree
   - `--max_depth 7` means: tree can have up to 7 levels of decisions
   - **Shallow (e.g., 3-5)**: Simple patterns, may underfit
   - **Deep (e.g., 10-15)**: Complex patterns, may overfit
   - **Recommended**: 5-8 for financial data

**Additional Parameters (Fixed in Code):**
- **subsample**: 0.8 (use 80% of data for each tree - prevents overfitting)
- **colsample_bytree**: 0.8 (use 80% of features for each tree - increases diversity)
- **tree_method**: 'hist' (faster histogram-based tree building)
- **device**: 'cuda' (uses GPU if available, auto-fallback to CPU)

**How XGBoost Works for Stock Prediction:**

1. **Feature Engineering**: Automatically creates lag features:
   - `Close_lag_1, Close_lag_2, ...`: Previous days' closing prices
   - `Volume_lag_1, Volume_lag_2, ...`: Previous days' volumes
   - `Price_change_1d, Price_change_5d`: Price change percentages
   - `Volatility_5d, Volatility_10d`: Rolling price volatility

2. **Sequential Tree Building**:
   - Tree 1: Makes initial price prediction
   - Tree 2: Corrects errors from Tree 1
   - Tree 3: Corrects remaining errors from Tree 1 + Tree 2
   - Continue for N trees...

3. **Final Prediction**: Sum of all tree predictions × learning_rate

4. **Feature Importance**: Shows which features matter most (saved as plot)

**Advantages of XGBoost:**
- Fast training (faster than LSTM)
- Interpretable (can see which features are important)
- Handles missing data automatically
- Less prone to overfitting with proper parameters
- Works well with technical indicators

**Outputs:**
- `xgboost_model.pkl`: Trained XGBoost model
- `xgboost_scaler.pkl`: Feature scaler (for standardizing inputs)
- `xgboost_features.txt`: List of all features used (including lag features)
- `xgboost_model_info.txt`: Model performance metrics
- `xgboost_feature_importance.png`: Feature importance plot (shows what model focuses on)
- `xgboost_predictions.png`: Actual vs predicted prices plot

### Random Forest Model (Ensemble with Walk-Forward Validation)

Train a Random Forest model with walk-forward validation:

```bash
# Basic usage (default parameters)
python train_randomforest.py MSFT_daily_data_20260520.csv

# Custom parameters
python train_randomforest.py MSFT_daily_data_20260520.csv --n_estimators 1000 --max_depth 20 --max_features 40
```

**Parameters:**
- `--n_estimators`: Number of trees in forest (default: 500)
- `--max_depth`: Maximum tree depth (default: 15)
- `--max_features`: Number of top features to use (default: 30)

**Understanding Random Forest Parameters:**

1. **N_estimators (Number of Trees)**
   - How many decision trees to train independently
   - `--n_estimators 500` means: build 500 trees, each on random subset of data
   - **Too few (e.g., 50-100)**: High variance, unstable predictions
   - **Too many (e.g., 2000+)**: Diminishing returns, slower inference
   - **Recommended**: 500-1000 trees

2. **Max Depth**
   - Maximum depth of each decision tree
   - `--max_depth 15` means: tree can have up to 15 levels
   - **Shallow (e.g., 5-10)**: Simple patterns, may underfit
   - **Deep (e.g., 20-30)**: Complex patterns, may overfit
   - **Recommended**: 15-20 for financial data

3. **Max Features**
   - Number of top features to select (by correlation with target)
   - `--max_features 30` means: use top 30 most correlated features
   - This reduces noise and speeds up training
   - **Fewer (e.g., 15-20)**: Faster training, less overfitting
   - **More (e.g., 40-50)**: More information, potential overfitting
   - **Recommended**: 25-35 features

**Walk-Forward Validation:**

Unlike simple train/test split, walk-forward validation simulates real trading:

```
Example with 1000 days, 5 folds:
┌─────────────────────────────────────────────────────┐
│ Fold 1: Train [1-600]    → Test [601-800]         │
│ Fold 2: Train [1-800]    → Test [801-900]         │
│ Fold 3: Train [1-900]    → Test [901-950]         │
│ Fold 4: Train [1-950]    → Test [951-975]         │
│ Fold 5: Train [1-975]    → Test [976-1000]        │
└─────────────────────────────────────────────────────┘
```

**Benefits:**
- Tests on truly unseen future data
- No look-ahead bias
- Mimics real trading conditions
- More realistic performance estimates

**How Random Forest Works for Stock Prediction:**

1. **Bootstrap Aggregation (Bagging)**:
   - Each tree trained on random sample of data (with replacement)
   - Each tree sees different subset of features at each split
   - Reduces overfitting through diversity

2. **Feature Engineering**: Same as XGBoost/LightGBM:
   - `Close_lag_1, Close_lag_2, ...`: Previous days' closing prices
   - `Volume_lag_1, Volume_lag_2, ...`: Previous days' volumes
   - `Price_change_1d, Price_change_5d, Price_change_10d`: Price changes
   - `Volatility_5d, Volatility_10d`: Rolling volatility
   - `MA_5, MA_10, MA_20`: Moving averages
   - **Top 30 features selected by correlation**

3. **Prediction**: Average of all tree predictions (regression)

4. **Feature Importance**: Shows which features matter most (saved as plot)

**Advantages of Random Forest:**
- **Robust to overfitting**: Ensemble of many trees averages out errors
- **Walk-forward validation**: Most realistic evaluation method
- **No GPU required**: Efficiently uses all CPU cores
- **Handles non-linear relationships**: Decision trees capture complex patterns
- **Interpretable**: Can see which features are important
- **Stable predictions**: Less sensitive to small data changes than single tree

**Outputs:**
- `randomforest_model.pkl`: Trained Random Forest model
- `randomforest_scaler.pkl`: Feature scaler (for standardizing inputs)
- `randomforest_features.txt`: List of all features used (including lag features)
- `randomforest_model_info.txt`: Model performance metrics (with walk-forward results)
- `randomforest_feature_importance.png`: Feature importance plot
- `randomforest_predictions.png`: Actual vs predicted prices plot (last 100 test samples)

### Model Evaluation Metrics

All models provide:

**Regression Metrics** (Price Prediction):
- **MAE** (Mean Absolute Error): Average price prediction error in dollars
- **RMSE** (Root Mean Squared Error): Penalizes larger errors more

**Classification Metrics** (Direction Prediction - Up/Down):
- **Accuracy**: Percentage of correct direction predictions
- **Precision**: Of predicted "up" days, how many were actually up
- **Recall**: Of actual "up" days, how many were correctly predicted
- **F1-Score**: Harmonic mean of precision and recall

### Example Training Workflow

```bash
# Step 1: Fetch 48 months of MSFT data and train all models (recommended)
python main.py --ticker MSFT

# Or manually:
# Step 1: Fetch 48 months of MSFT data (default)
python fetch_stock_data.py MSFT

# Step 2: Train LSTM model (deep learning - captures sequences)
python train_lstm.py MSFT_daily_data_20260520.csv

# Step 3: Train LightGBM model (fast, accurate gradient boosting)
python train_lightgbm.py MSFT_daily_data_20260520.csv

# Step 4: Train XGBoost model (alternative gradient boosting)
python train_xgboost.py MSFT_daily_data_20260520.csv

# Step 5: Train Random Forest model (ensemble with walk-forward validation)
python train_randomforest.py MSFT_daily_data_20260520.csv

# Step 6: Compare all models - use main.py to get HTML report
python main.py MSFT_daily_data_20260520.csv
```

## 🎯 Multi-Approach Win Probability Analysis

**NEW FEATURE**: All models now include advanced probability analysis that answers: **"What's the probability this trade hits Take Profit before Stop Loss?"**

Instead of just predicting tomorrow's price, the system uses **3 independent approaches** to calculate your actual win probability over the next 5 days:

### The Three Approaches

#### 1️⃣ Multi-Day Sequential Prediction
- Uses the trained model to predict next 5 days sequentially
- Simulates the most likely price path according to the model
- Checks which day (if any) hits your TP or SL first
- **Output**: Binary result (will hit TP: 100% or will hit SL: 0%)

#### 2️⃣ Monte Carlo Simulation (Default: 1000 runs)
- Generates thousands of random price paths based on historical volatility
- Each path includes model's predicted trend + market randomness
- Counts how many simulations hit TP vs SL first
- **Output**: Statistical win rate (e.g., "714 out of 1000 hit TP = 71% probability")

#### 3️⃣ Historical Pattern Matching (Default: 50 patterns)
- Searches last 200 days for similar market conditions:
  - Similar RSI (momentum indicator)
  - Similar volatility level
  - Similar trend direction
- Checks what happened in those similar setups
- **Output**: Historical success rate (e.g., "31 out of 48 won = 64% probability")

### Ensemble Decision

The system combines all three approaches using weighted average:
- **Multi-Day Prediction**: 40% weight
- **Monte Carlo**: 35% weight
- **Historical Patterns**: 25% weight

**Final Output:**
```
✨ ENSEMBLE WIN PROBABILITY: 68%
   Confidence Level: HIGH
   Recommendation: TAKE TRADE
```

### Configurable Parameters

All parameters are in `trade_probability_analyzer.py` and can be adjusted:

```python
# Multi-Day Prediction
PREDICTION_DAYS = 5              # Days to predict ahead (default: 5 = 1 week)
MIN_CONFIDENCE_THRESHOLD = 60.0  # Min win % to recommend trade (default: 60%)

# Monte Carlo Simulation
MONTE_CARLO_SIMULATIONS = 1000   # Number of simulations (default: 1000)
MC_DRIFT_INFLUENCE = 0.3         # Trend bias strength 0-1 (default: 0.3)

# Historical Pattern Matching
PATTERN_LOOKBACK = 200           # Days to search back (default: 200)
PATTERN_MATCH_COUNT = 50         # Similar patterns to find (default: 50)
RSI_TOLERANCE = 5                # RSI similarity ± (default: 5)
VOLATILITY_TOLERANCE = 0.2       # Volatility similarity ± (default: 20%)

# Ensemble Weights (must sum to 1.0)
WEIGHT_PREDICTION = 0.4          # Multi-day prediction weight
WEIGHT_MONTE_CARLO = 0.35        # Monte Carlo weight
WEIGHT_PATTERN = 0.25            # Pattern matching weight

# Confidence Levels
CONFIDENCE_HIGH = 75.0           # Above this = HIGH confidence
CONFIDENCE_MEDIUM = 65.0         # Medium-High threshold
```

### How to Use

The probability analysis runs automatically when you train any model:

```bash
# Runs probability analysis for all 3 models
python main.py --ticker TSLA

# Or for individual model
python train_lightgbm.py TSLA_daily_data_20260522.csv
```

### Understanding the Output

**Console Output Example:**
```
📊 TRADE PROBABILITY ANALYSIS
══════════════════════════════════════════════════════════════════════
Current Price: $417.85
Signal: SHORT (SELL)
Stop Loss: $424.14 (+1.50%)
Take Profit: $407.37 (-2.51%)

──────────────────────────────────────────────────────────────────────
APPROACH 1: Multi-Day Sequential Prediction
──────────────────────────────────────────────────────────────────────
✅ Predicts TAKE PROFIT hit on Day 2
   Win Probability: 100%

   Predicted Path (5-day):
   Day 1: $410.84
   Day 2: $408.12 ← 🎯 HIT TAKE PROFIT
   Day 3: $406.50
   Day 4: $409.20
   Day 5: $411.00

──────────────────────────────────────────────────────────────────────
APPROACH 2: Monte Carlo Simulation (1000 runs)
──────────────────────────────────────────────────────────────────────
Win Probability: 71.4%
  • 714 simulations hit Take Profit
  • 286 simulations hit Stop Loss
  • 0 simulations hit neither
  • Avg days to TP: 2.3
  • Avg days to SL: 1.8

──────────────────────────────────────────────────────────────────────
APPROACH 3: Historical Pattern Matching
──────────────────────────────────────────────────────────────────────
Win Probability: 64.6%
  • Found 48 similar historical setups
  • 31 times TP was hit first
  • 17 times SL was hit first
  • 0 times neither was hit

══════════════════════════════════════════════════════════════════════
🎖️  ENSEMBLE DECISION
══════════════════════════════════════════════════════════════════════

✨ ENSEMBLE WIN PROBABILITY: 78.4%
   Confidence Level: HIGH
   Recommendation: TAKE TRADE

   Contributing Methods:
   • Multi-Day Prediction: 100.0%
   • Monte Carlo: 71.4%
   • Historical Patterns: 64.6%

✅ RECOMMENDATION: SHORT (SELL)
   This trade has 78.4% probability of hitting
   Take Profit before Stop Loss in the next 5 days.
```

### Trading Strategy with Probability

**Recommended Rules:**
- **Probability ≥ 75% (HIGH)**: Strong trade, full position size
- **Probability 65-75% (MEDIUM)**: Good trade, reduce position size 50%
- **Probability 60-65% (LOW)**: Marginal trade, smallest position or skip
- **Probability < 60%**: Skip trade, wait for better setup

**IQ Option Settings (5x Leverage):**
The models automatically calculate your leveraged position P&L:
```
5x Leverage Position P&L (for IQ Option auto-close):
  Stop Loss %:   +7.5%    (stock moves 1.5% against you)
  Take Profit %: -12.5%   (stock moves 2.5% in your favor)
  Risk/Reward:   1.67:1
```

### Swing Trading Configuration

Current settings are optimized for **1-2 day swing trades with 5x leverage**:
- Stop Loss: 0.6× volatility (~1.5% stock move)
- Take Profit: 1.0× volatility (~2.5% stock move)

To change risk/reward, edit the training scripts (`train_lightgbm.py`, etc.):
```python
# Line ~457-458 in each training script
stop_loss_distance = 0.6 * volatility  # Change multiplier for tighter/wider stops
take_profit_distance = 1.0 * volatility  # Change multiplier for tighter/wider targets
```

### HTML Report Integration

The probability analysis is automatically included in the HTML report generated by `main.py`. Each model's section shows:
- Ensemble win probability
- Confidence level (HIGH/MEDIUM/LOW)
- Recommendation (TAKE TRADE / SKIP TRADE)
- Individual method probabilities

## 📋 Summary & Notes

### Model Comparison

| Model | Type | Validation | Speed | GPU | Best For |
|-------|------|------------|-------|-----|----------|
| **LSTM** | Deep Learning | 90/10 Split | Slow | ✅ Yes | Sequential patterns, trends |
| **XGBoost** | Gradient Boosting | 90/10 Split | Medium | ✅ Yes | Feature-rich data, interpretability |
| **LightGBM** | Gradient Boosting | 90/10 Split | Fast | ✅ Yes | Large datasets, speed + accuracy |
| **RandomForest** | Ensemble Trees | Walk-Forward (5 folds) | Medium | ❌ CPU | Robustness, realistic evaluation |

### Key Features

- ✅ **4 Models**: LSTM, XGBoost, LightGBM, RandomForest
- ✅ **3 Probability Approaches**: Multi-day prediction, Monte Carlo, Historical patterns
- ✅ **Beautiful HTML Reports**: Professional dashboard with cards, tables, and badges
- ✅ **5x Leverage Calculations**: Ready for IQ Option auto-close settings
- ✅ **Walk-Forward Validation**: RandomForest uses realistic backtesting
- ✅ **Parallel Training**: All 4 models train simultaneously (3-7 minutes)
- ✅ **GPU Acceleration**: LSTM, XGBoost, LightGBM support CUDA
- ✅ **Feature Engineering**: 30+ lag features, volatility, moving averages

### Data & Training

- **Data Source**: Yahoo Finance via yfinance library
- **Default History**: 48 months (4 years) of daily data
- **Train/Test Split**: 
  - LSTM, XGBoost, LightGBM: 90/10 split (~43 months train, ~5 months test)
  - RandomForest: Walk-forward validation (5 rolling folds)
- **Cryptocurrency**: 24/7 trading → more data points than stocks

### Model Architecture

- **LSTM**: 1 layer (50 units) + 40% dropout → ultra-fast, minimal overfitting
- **XGBoost**: 1000 trees, depth 7, level-wise growth
- **LightGBM**: 1000 trees, 31 leaves, leaf-wise growth (2-4x faster than XGBoost)
- **RandomForest**: 500 trees, depth 15, top 30 features, walk-forward validation

### Performance

- **Training Time**: 3-7 minutes for all 4 models (with GPU)
- **Probability Analysis**: Automatic, ~5-10 seconds per model
- **Report Generation**: Beautiful HTML dashboard with all results

### Trading Strategy

- **Stop Loss/Take Profit**: Based on volatility (0.6× and 1.0× daily volatility)
- **Optimized for**: 1-2 day swing trades with 5x leverage
- **Win Probability**: Ensemble of 3 approaches (prediction + Monte Carlo + patterns)
- **Minimum Threshold**: 60% probability to recommend trade
- **Confidence Levels**: HIGH (≥75%), MEDIUM (65-75%), LOW (<65%)
