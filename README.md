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

**Step 3: Already done! TensorFlow and Keras are installed**

TensorFlow 2.21 is installed and will use **CPU** for training.

**Note:** TensorFlow 2.11+ does not support GPU on native Windows without WSL2. CPU training works perfectly fine:
- LSTM: ~5-10 minutes for 6 months of data
- XGBoost: ~1-2 minutes for 6 months of data

### Check GPU/CUDA Availability

To check if CUDA/GPU is available for training:

```bash
# Check TensorFlow GPU detection
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); gpus = tf.config.list_physical_devices('GPU'); print('GPU devices:', gpus)"

# Check if CUDA is available on system
nvidia-smi

# Check XGBoost GPU support
python -c "import xgboost as xgb; print('XGBoost version:', xgb.__version__); print('XGBoost built with CUDA:', xgb.build_info()['USE_CUDA'])"
```

**Test script to verify all GPU components:**

```python
# test_gpu.py
import tensorflow as tf
import xgboost as xgb
import sys

print("="*60)
print("GPU/CUDA Availability Check")
print("="*60)

# Python version
print(f"\nPython version: {sys.version}")

# TensorFlow GPU check
print(f"\nTensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"TensorFlow GPU devices: {len(gpus)}")
if gpus:
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
else:
    print("  No GPU detected by TensorFlow")

# XGBoost GPU check
print(f"\nXGBoost version: {xgb.__version__}")
try:
    build_info = xgb.build_info()
    use_cuda = build_info.get('USE_CUDA', False)
    print(f"XGBoost CUDA support: {use_cuda}")
except:
    print("XGBoost CUDA support: Unknown")

print("\n" + "="*60)
print("Note: On Windows, TensorFlow 2.11+ does not support native GPU.")
print("XGBoost may still use GPU if CUDA toolkit is installed.")
print("="*60)
```

Run the test script:
```bash
python test_gpu.py
```

### Training Time Comparison

| Model   | Dataset Size | CPU Time | GPU Time (DirectML) |
|---------|-------------|----------|---------------------|
| LSTM    | 6 months    | ~5-10 min | ~1-2 min           |
| XGBoost | 6 months    | ~1-2 min  | ~30-60 sec         |

**Note:** GPU support requires Python 3.11 and TensorFlow-DirectML. XGBoost auto-detects GPU if CUDA is available.

## Usage

### Basic Command

```bash
python fetch_stock_data.py <TICKER> --months <NUMBER_OF_MONTHS>
```

### Parameters

- `ticker` (required): The ticker symbol for the asset you want to fetch
- `--months` (optional): Number of months of historical data (default: 6)

## Examples

### Stock Market Data

```bash
# Microsoft (6 months - default)
python fetch_stock_data.py MSFT

# Microsoft (12 months)
python fetch_stock_data.py MSFT --months 12

# Apple (6 months)
python fetch_stock_data.py AAPL --months 6

# Google (3 months)
python fetch_stock_data.py GOOGL --months 3

# Tesla (12 months)
python fetch_stock_data.py TSLA --months 12

# Amazon (6 months)
python fetch_stock_data.py AMZN --months 6

# NVIDIA (6 months)
python fetch_stock_data.py NVDA --months 6
```

### Cryptocurrency Data

```bash
# Bitcoin (6 months)
python fetch_stock_data.py BTC-USD --months 6

# Ethereum (12 months)
python fetch_stock_data.py ETH-USD --months 12

# Solana (3 months)
python fetch_stock_data.py SOL-USD --months 3

# Cardano (6 months)
python fetch_stock_data.py ADA-USD --months 6

# Dogecoin (6 months)
python fetch_stock_data.py DOGE-USD --months 6

# Ripple (6 months)
python fetch_stock_data.py XRP-USD --months 6

# Polkadot (6 months)
python fetch_stock_data.py DOT-USD --months 6
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

### Technical Indicators

#### Moving Averages (MA)
- **MA_5**: 5-day moving average
- **MA_10**: 10-day moving average
- **MA_20**: 20-day moving average
- **MA_50**: 50-day moving average
- **MA_200**: 200-day moving average (requires sufficient data)

#### Momentum Indicators
- **RSI_14**: 14-period Relative Strength Index (overbought/oversold indicator)
- **MACD**: Moving Average Convergence Divergence line
- **MACD_Signal**: MACD signal line
- **MACD_Hist**: MACD histogram (MACD - Signal)

#### Volatility Indicators
- **BB_Upper**: Upper Bollinger Band (2 std dev above 20-day MA)
- **BB_Middle**: Middle Bollinger Band (20-day MA)
- **BB_Lower**: Lower Bollinger Band (2 std dev below 20-day MA)

#### Volume Indicators
- **Volume_MA_20**: 20-day volume moving average

## Data Split for Training

For machine learning models:
- **6 months of data**: ~123 trading days
  - Training: ~102 days (5 months)
  - Testing: ~21 days (1 month)

## Training Models

After fetching the data, you can train prediction models using either LSTM or XGBoost.

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

**LSTM Model Architecture:**

The model uses 3 LSTM layers with decreasing units:
- **Layer 1**: 128 LSTM units (captures complex patterns)
- **Layer 2**: 64 LSTM units (intermediate patterns)
- **Layer 3**: 32 LSTM units (refined patterns)
- **Dropout**: 20% after each LSTM layer (prevents overfitting)
- **Dense Layer**: 16 units with ReLU activation
- **Output Layer**: 1 unit (predicted closing price)

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

### XGBoost Model (Gradient Boosting)

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

Both models provide:

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
# Step 1: Fetch 6 months of MSFT data
python fetch_stock_data.py MSFT --months 6

# Step 2: Train LSTM model
python train_lstm.py MSFT_daily_data_20260519.csv

# Step 3: Train XGBoost model
python train_xgboost.py MSFT_daily_data_20260519.csv

# Step 4: Compare results and choose the best model
```

## Notes

- All data is fetched with daily intervals (1 day per candle)
- Some indicators (like MA_200) may have null values if insufficient historical data
- Cryptocurrency markets trade 24/7, so they may have more data points than stocks
- Data is fetched from Yahoo Finance via yfinance library
- LSTM models are better for capturing long-term dependencies in time series
- XGBoost models are faster to train and easier to interpret (feature importance)
- Both models use 5 months for training and 1 month for testing
