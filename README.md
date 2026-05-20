# Stock Market Entry Signal

A tool to fetch and analyze stock market and cryptocurrency data with technical indicators for price prediction.

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
2. Train all 3 models in parallel (LSTM, XGBoost, LightGBM)
3. Generate comparison report: `RESULT-{TICKER}-{DATE}.html`

**Training Details:**
- Train/Test Split: 90/10 (90% training, 10% testing)
- With 48 months data: ~43 months training, ~5 months testing

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

# Step 3: Train LightGBM model (recommended - fast, accurate)
python train_lightgbm.py MSFT_daily_data_20260520.csv

# Step 4: Train XGBoost model (alternative gradient boosting)
python train_xgboost.py MSFT_daily_data_20260520.csv

# Step 5: Compare all models - use main.py to get HTML report
python main.py MSFT_daily_data_20260520.csv
```

## Notes

- All data is fetched with daily intervals (1 day per candle)
- Default historical data: 48 months (4 years)
- Cryptocurrency markets trade 24/7, so they may have more data points than stocks
- Data is fetched from Yahoo Finance via yfinance library
- LSTM models are better for capturing long-term dependencies in time series
- LightGBM is recommended for tabular data (often faster and more accurate than XGBoost)
- XGBoost and LightGBM are easier to interpret (feature importance plots)
- Train/test split: 90/10 (with 48 months = ~43 months training, ~5 months testing)
- LSTM ultra-simplified to 1 layer (50 units) with 40% dropout - fastest training, minimal overfitting
- LightGBM uses leaf-wise tree growth (2-4x faster than XGBoost's level-wise)
- All models support GPU acceleration for faster training
