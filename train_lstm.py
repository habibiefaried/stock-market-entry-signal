"""
LSTM Stock Price Prediction Model (Keras with PyTorch Backend + GPU)

This script trains an LSTM (Long Short-Term Memory) neural network to predict stock prices.
LSTM is a type of Recurrent Neural Network (RNN) that can learn patterns in sequences of data.

Key Concepts:
- LSTM: Remembers long-term patterns in time series (unlike simple RNNs that forget)
- Sequential Data: Each data point depends on previous points (time matters!)
- Backpropagation Through Time: LSTM learns by looking at sequences, not individual points
- GPU Acceleration: Uses PyTorch backend with CUDA for 5-10x faster training

Why LSTM for stocks?
- Stock prices are sequential - today's price depends on yesterday's, last week's, etc.
- LSTM can "remember" important patterns from days/weeks ago
- Can learn complex non-linear relationships between features
"""

# === Environment Setup ===
# CRITICAL: Set PyTorch as backend BEFORE importing Keras!
# This must be the very first thing in the script
import os
os.environ['KERAS_BACKEND'] = 'torch'  # Use PyTorch instead of TensorFlow
# This enables GPU acceleration on Windows!

# Disable warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# === Imports ===
import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
from sklearn.preprocessing import MinMaxScaler  # Scale data to 0-1 range
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score  # Evaluation
import keras  # High-level neural network API
from keras.models import Sequential  # Linear stack of layers
from keras.layers import LSTM, Dense, Dropout  # Neural network layers
from keras.callbacks import EarlyStopping, ModelCheckpoint  # Training callbacks
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt  # Plotting
import argparse  # Command-line arguments
import torch  # PyTorch (for GPU check)
from datetime import datetime
import logging  # Suppress matplotlib logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)  # Suppress matplotlib warnings
logging.getLogger('PIL').setLevel(logging.ERROR)  # Suppress PIL warnings
logging.getLogger('keras').setLevel(logging.ERROR)  # Suppress Keras warnings
logging.getLogger('torch').setLevel(logging.ERROR)  # Suppress PyTorch warnings

# Import trade probability analyzer
from trade_probability_analyzer import (
    predict_multi_day_path,
    monte_carlo_simulation,
    find_similar_patterns,
    calculate_ensemble_probability,
    format_analysis_report
)

# Print backend info
print(f"Keras backend: {keras.backend.backend()}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def load_and_prepare_data(csv_file):
    """
    Load and clean stock data from CSV

    Args:
        csv_file (str): Path to CSV with stock data

    Returns:
        df (DataFrame): Cleaned data
        feature_cols (list): List of feature column names

    What this does:
    - Loads CSV file with stock prices and technical indicators
    - Removes rows with missing values (NaN)
    - Selects relevant features for the LSTM model
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    # Select ONLY OHLCV features for training
    # Technical indicators (MA, RSI, MACD, etc.) are kept in CSV for trading signals,
    # but NOT used for model training
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Verify all required columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"Using features: {feature_cols}")
    print(f"Total records: {len(df)} (no NaN removal needed for OHLCV)")

    return df, feature_cols

def prepare_data_for_lstm(df, feature_cols):
    """
    Prepare data for LSTM training - keep only necessary columns

    Args:
        df (DataFrame): Full dataframe with all columns
        feature_cols (list): Feature columns to use (OHLCV)

    Returns:
        df_clean (DataFrame): Dataframe with only Date + features
    """
    # Only keep Date + feature columns to avoid NaN from unused technical indicators
    keep_cols = ['Date'] + feature_cols
    df_clean = df[keep_cols].copy()
    print(f"Using columns for LSTM: {df_clean.columns.tolist()}")
    return df_clean

def create_sequences(data, target, lookback=60):
    """
    Create sequences for LSTM training

    LSTM needs sequences, not individual data points!
    This function transforms flat data into sequences.

    Args:
        data (array): Feature data (scaled)
        target (array): Target values (scaled prices)
        lookback (int): How many time steps to look back (default: 60 days)

    Returns:
        X (array): Sequences of shape (samples, lookback, features)
        y (array): Target values (next day's price)

    Example with lookback=3:
    Input data: [Day1, Day2, Day3, Day4, Day5, ...]

    Sequences created:
    X[0] = [Day1, Day2, Day3] -> y[0] = Day4
    X[1] = [Day2, Day3, Day4] -> y[1] = Day5
    X[2] = [Day3, Day4, Day5] -> y[2] = Day6
    ...

    The LSTM sees 60 days of history and predicts day 61!
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        # Take a slice of 'lookback' days as input
        X.append(data[i-lookback:i])  # Days (i-60) to (i-1)
        # The target is the next day's price
        y.append(target[i])  # Day i
    return np.array(X), np.array(y)

def split_train_test(df, feature_cols, train_ratio=9/10):
    """
    Split data chronologically into train and test sets

    CRITICAL: For time series, we MUST preserve time order!
    - Training data: Past (first 90%)
    - Test data: Future (last 10%)

    Random splitting would leak future information into training!

    Args:
        df (DataFrame): Input data
        feature_cols (list): Feature column names
        train_ratio (float): Proportion for training (9/10 = 90%)

    Returns:
        train_df, test_df: Chronologically split dataframes
    """
    split_idx = int(len(df) * train_ratio)

    # Chronological split - do NOT shuffle!
    train_df = df[:split_idx]  # First 90%
    test_df = df[split_idx:]   # Last 10%

    print(f"\nData split:")
    print(f"Training set: {len(train_df)} records ({train_df['Date'].min()} to {train_df['Date'].max()})")
    print(f"Test set: {len(test_df)} records ({test_df['Date'].min()} to {test_df['Date'].max()})")

    return train_df, test_df

def build_lstm_model(input_shape):
    """
    Build LSTM neural network architecture (ULTRA-SIMPLIFIED)

    Architecture:
    Input -> LSTM(50) -> Dropout(0.4) -> Dense(1)

    Args:
        input_shape (tuple): (lookback, num_features) e.g., (60, 5)

    Returns:
        model: Compiled Keras model

    Layer Explanation:
    1. LSTM(50, return_sequences=False): Single LSTM layer with 50 memory units
       - Only 50 units (vs previous 64->32 = 2 layers)
       - return_sequences=False: Output only final state (no sequence passing)
       - Learns patterns from sequences and outputs single representation
       - Minimal complexity to prevent overfitting

    2. Dropout(0.4): Randomly drops 40% of connections during training
       - Increased to 40% for maximum regularization
       - Prevents overfitting on noisy financial data
       - Forces model to learn only robust patterns

    3. Dense(1): Output layer
       - Single neuron = single prediction (tomorrow's price)
       - No activation = linear output (can be any price value)
       - Direct connection from LSTM to prediction

    Why this ultra-simple architecture?
    - 1 LSTM layer instead of 2 or 3: Fastest training, least overfitting
    - Only 50 units: ~15K parameters (vs ~30K with 2 layers, ~130K with 3 layers)
    - 40% dropout: Maximum regularization for financial data
    - No stacked layers: Simplest possible LSTM model that still works
    - Best for noisy data: Financial prices are mostly unpredictable (random walk)
    - Trains in <1 minute: Fast experimentation

    Benefits:
    - 5x faster training than 3-layer model
    - Less prone to overfitting
    - Good baseline - if this works, more complex models might not help
    """
    model = Sequential([
        # Single LSTM layer
        LSTM(50, return_sequences=False, input_shape=input_shape),
        Dropout(0.4),  # Strong regularization

        # Output layer
        Dense(1)  # Single output = predicted price
    ])

    # Compile model - configure training process
    model.compile(
        optimizer='adam',  # Adam: Adaptive learning rate optimizer (very popular, works well)
        loss='mse',  # MSE: Mean Squared Error - standard loss for regression
        metrics=['mae']  # MAE: Mean Absolute Error - easier to interpret than MSE
    )

    return model

def calculate_direction_metrics(y_true, y_pred):
    """
    Calculate classification metrics for price direction prediction

    While LSTM predicts exact prices, traders care about DIRECTION:
    "Will price go UP or DOWN tomorrow?"

    This converts continuous price predictions to binary direction:
    - 1 = UP (price increased)
    - 0 = DOWN (price decreased)

    Metrics:
    - Accuracy: % of correct direction predictions
    - Precision: Of predicted UPs, how many were actually UP?
    - Recall: Of actual UPs, how many did we catch?
    - F1: Harmonic mean of precision and recall

    Args:
        y_true (array): Actual prices
        y_pred (array): Predicted prices

    Returns:
        accuracy, precision, recall, f1 (floats)
    """
    # np.diff calculates day-to-day changes
    # > 0 checks if change is positive (UP)
    # .astype(int) converts True/False to 1/0
    y_true_direction = (np.diff(y_true) > 0).astype(int)
    y_pred_direction = (np.diff(y_pred) > 0).astype(int)

    accuracy = accuracy_score(y_true_direction, y_pred_direction)
    precision = precision_score(y_true_direction, y_pred_direction, zero_division=0)
    recall = recall_score(y_true_direction, y_pred_direction, zero_division=0)
    f1 = f1_score(y_true_direction, y_pred_direction, zero_division=0)

    return accuracy, precision, recall, f1

def train_lstm_model(csv_file, lookback=60, epochs=100, batch_size=32):
    """
    Main training function for LSTM model

    Training Process:
    1. Load and prepare data
    2. Scale features to 0-1 range (LSTM works best with normalized data)
    3. Create sequences (60-day windows)
    4. Build LSTM model
    5. Train with early stopping (stops if no improvement)
    6. Evaluate on test set
    7. Save model and results

    Args:
        csv_file (str): Path to stock data CSV
        lookback (int): Sequence length (default: 60 days)
        epochs (int): Maximum training iterations (default: 100)
        batch_size (int): Samples per training batch (default: 32)

    Parameters Explained:
    - lookback (60): LSTM sees 60 days of history to predict day 61
      Too small -> can't learn long-term patterns
      Too large -> not enough training samples
    - epochs (100): How many times to go through entire dataset
      Early stopping will stop earlier if model stops improving
    - batch_size (32): Process 32 sequences at once
      Larger = faster but needs more GPU memory
      Smaller = slower but more stable training
    """

    # === Step 1: Load Data ===
    df, feature_cols = load_and_prepare_data(csv_file)

    # === Step 2: Keep Only Necessary Columns ===
    df = prepare_data_for_lstm(df, feature_cols)

    # === Step 3: Split Train/Test ===
    train_df, test_df = split_train_test(df, feature_cols)

    # === Step 3: Prepare Features and Target ===
    # Extract feature columns and target (Close price)
    X_train_data = train_df[feature_cols].values
    y_train_data = train_df['Close'].values

    X_test_data = test_df[feature_cols].values
    y_test_data = test_df['Close'].values

    # === Step 4: Scale Data ===
    # MinMaxScaler: Transforms data to range [0, 1]
    # Why? Neural networks train better with normalized inputs
    # Formula: x_scaled = (x - min) / (max - min)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # fit_transform on training data: Learn min/max and transform
    X_train_scaled = scaler_X.fit_transform(X_train_data)
    y_train_scaled = scaler_y.fit_transform(y_train_data.reshape(-1, 1))

    # transform on test data: Use training min/max (no peeking at test data!)
    X_test_scaled = scaler_X.transform(X_test_data)
    y_test_scaled = scaler_y.transform(y_test_data.reshape(-1, 1))

    # === Step 5: Create Sequences ===
    # Transform flat data into sequences for LSTM
    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, lookback)
    X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, lookback)

    print(f"\nSequence shape:")
    print(f"X_train shape: {X_train.shape}")  # (samples, lookback, features)
    print(f"y_train shape: {y_train.shape}")  # (samples, 1)
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Check if we have enough data
    if len(X_train) == 0 or len(X_test) == 0:
        print("\n" + "="*60)
        print("ERROR: Not enough data to create sequences!")
        print("="*60)
        print(f"Lookback period: {lookback} days")
        print(f"Training data: {len(y_train_scaled)} days")
        print(f"Test data: {len(y_test_scaled)} days")
        print(f"\nSequences created:")
        print(f"  Training sequences: {len(X_train)}")
        print(f"  Test sequences: {len(X_test)}")

        # Calculate recommended lookback
        max_lookback = min(len(y_train_scaled), len(y_test_scaled)) - 1
        recommended_lookback = max(10, max_lookback // 2)

        print(f"\nSOLUTION:")
        print(f"  Reduce lookback to: {recommended_lookback} days or less")
        print(f"  Example: python train_lstm.py {csv_file} --lookback {recommended_lookback}")
        print(f"\n  OR fetch more data (e.g., 12 months):")
        print(f"  python fetch_stock_data.py MSFT --months 12")
        print("="*60)
        return

    # === Step 6: Build Model ===
    print("\nBuilding LSTM model...")
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    print(model.summary())  # Print model architecture

    # === Step 7: Configure Training Callbacks ===
    # Callbacks: Functions called during training

    # EarlyStopping: Stop training if validation loss doesn't improve
    early_stop = EarlyStopping(
        monitor='val_loss',  # Watch validation loss
        patience=10,  # Wait 10 epochs before stopping
        restore_best_weights=True  # Load weights from best epoch
    )

    # ModelCheckpoint: Save best model during training
    model_checkpoint = ModelCheckpoint(
        'best_lstm_model.keras',  # Filename
        monitor='val_loss',  # Save when this improves
        save_best_only=True  # Only save if better than previous best
    )

    # === Step 8: Train Model ===
    print("\nTraining LSTM model on GPU...")
    print("This may take 1-2 minutes with GPU, 5-10 minutes with CPU")

    history = model.fit(
        X_train, y_train,  # Training data
        epochs=epochs,  # Maximum number of epochs
        batch_size=batch_size,  # Process 32 sequences at a time
        validation_split=0.1,  # Use 10% of training for validation
        callbacks=[early_stop, model_checkpoint],  # Apply callbacks
        verbose=1  # Print progress
    )

    # === Step 9: Load Best Model ===
    model = keras.models.load_model('best_lstm_model.keras')

    # === Step 10: Make Predictions ===
    print("\nMaking predictions...")
    y_train_pred_scaled = model.predict(X_train)
    y_test_pred_scaled = model.predict(X_test)

    # === Step 11: Inverse Transform (Scale Back to Original Prices) ===
    # Convert from [0, 1] range back to actual dollar prices
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
    y_train_actual = scaler_y.inverse_transform(y_train)
    y_test_actual = scaler_y.inverse_transform(y_test)

    # === Step 12: Calculate Regression Metrics ===
    # MAE: Average absolute difference between predicted and actual
    # RMSE: Square root of average squared difference (penalizes large errors more)
    train_mae = mean_absolute_error(y_train_actual, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
    test_mae = mean_absolute_error(y_test_actual, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))

    # === Step 13: Calculate Direction Metrics ===
    train_acc, train_prec, train_rec, train_f1 = calculate_direction_metrics(
        y_train_actual.flatten(), y_train_pred.flatten()
    )
    test_acc, test_prec, test_rec, test_f1 = calculate_direction_metrics(
        y_test_actual.flatten(), y_test_pred.flatten()
    )

    # === Step 14: Print Results ===
    print("\n" + "="*60)
    print("LSTM MODEL EVALUATION RESULTS")
    print("="*60)

    print("\nREGRESSION METRICS (Price Prediction):")
    print(f"Training MAE:  ${train_mae:.2f}")
    print(f"Training RMSE: ${train_rmse:.2f}")
    print(f"Test MAE:      ${test_mae:.2f}")
    print(f"Test RMSE:     ${test_rmse:.2f}")

    print("\nCLASSIFICATION METRICS (Direction Prediction - Up/Down):")
    print(f"Training Accuracy:  {train_acc*100:.2f}%")
    print(f"Training Precision: {train_prec*100:.2f}%")
    print(f"Training Recall:    {train_rec*100:.2f}%")
    print(f"Training F1-Score:  {train_f1*100:.2f}%")
    print(f"\nTest Accuracy:      {test_acc*100:.2f}%")
    print(f"Test Precision:     {test_prec*100:.2f}%")
    print(f"Test Recall:        {test_rec*100:.2f}%")
    print(f"Test F1-Score:      {test_f1*100:.2f}%")

    print("\n" + "="*60)

    # === Step 15: Plot Training History ===
    plt.figure(figsize=(12, 4))

    # Plot loss over epochs
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot MAE over epochs
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('lstm_training_history.png')
    print("\nTraining history plot saved as: lstm_training_history.png")

    # === Step 16: Plot Predictions ===
    plt.figure(figsize=(15, 6))

    # Get dates for x-axis
    test_dates = test_df['Date'].values[lookback:]
    plt.plot(test_dates, y_test_actual, label='Actual Price', color='blue', linewidth=2)
    plt.plot(test_dates, y_test_pred, label='Predicted Price', color='red', linewidth=2, alpha=0.7)
    plt.title('LSTM Model: Actual vs Predicted Prices (Test Set)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lstm_predictions.png')
    print("Predictions plot saved as: lstm_predictions.png")

    # === Step 17: Generate Trading Signal ===
    print("\n" + "="*60)
    print("TRADING SIGNAL FOR NEXT DAY")
    print("="*60)

    # Get today's actual price (last price in original data)
    today_price = df['Close'].iloc[-1]
    today_date = df['Date'].iloc[-1]

    # Prepare sequence for tomorrow's prediction
    # Use last 'lookback' days of actual data
    recent_data = df[feature_cols].tail(lookback).values
    recent_scaled = scaler_X.transform(recent_data)
    recent_sequence = recent_scaled.reshape(1, lookback, len(feature_cols))

    # Predict tomorrow's price
    tomorrow_pred_scaled = model.predict(recent_sequence, verbose=0)
    tomorrow_pred = scaler_y.inverse_transform(tomorrow_pred_scaled)[0][0]

    # Calculate expected move
    expected_move = tomorrow_pred - today_price
    expected_move_pct = (expected_move / today_price) * 100

    # Determine signal
    if expected_move_pct > 0.5:
        signal = "BUY (LONG)"
        signal_emoji = "[BUY]"
    elif expected_move_pct < -0.5:
        signal = "SHORT (SELL)"
        signal_emoji = "[SHORT]"
    else:
        signal = "HOLD (No clear signal)"
        signal_emoji = "[HOLD]"

    # Calculate stop loss and take profit using recent volatility
    recent_prices = df['Close'].tail(20)
    daily_returns = recent_prices.pct_change().dropna()
    volatility = daily_returns.std() * today_price

    # SWING TRADING MODE (1-2 day trades with 5x leverage)
    # Stop Loss: 0.6x volatility (~1.5% stock move = 7.5% position loss)
    # Take Profit: 1.0x volatility (~2.5% stock move = 12.5% position gain)
    stop_loss_distance = 0.6 * volatility
    take_profit_distance = 1.0 * volatility

    if signal == "BUY (LONG)":
        stop_loss = today_price - stop_loss_distance
        take_profit = today_price + take_profit_distance
    elif signal == "SHORT (SELL)":
        stop_loss = today_price + stop_loss_distance
        take_profit = today_price - take_profit_distance
    else:
        stop_loss = today_price - stop_loss_distance
        take_profit = today_price + take_profit_distance

    confidence = test_acc * 100

    # Print trading signal
    print(f"\n{signal_emoji} SIGNAL: {signal}")
    print(f"\nCurrent Price (Today):     ${today_price:.2f}")
    print(f"Predicted Price (Tomorrow): ${tomorrow_pred:.2f}")
    print(f"Expected Move:             ${expected_move:+.2f} ({expected_move_pct:+.2f}%)")
    print(f"\nRisk Management (Stock Price Levels):")
    print(f"  Stop Loss:     ${stop_loss:.2f} ({((stop_loss - today_price) / today_price * 100):+.2f}%)")
    print(f"  Take Profit:   ${take_profit:.2f} ({((take_profit - today_price) / today_price * 100):+.2f}%)")
    print(f"\n5x Leverage Position P&L (for IQ Option auto-close):")
    print(f"  Stop Loss %:   {((stop_loss - today_price) / today_price * 100 * 5):+.1f}%")
    print(f"  Take Profit %: {((take_profit - today_price) / today_price * 100 * 5):+.1f}%")
    print(f"  Risk/Reward:   1.67:1")
    print(f"\nModel Confidence: {confidence:.1f}% (based on test accuracy)")
    print(f"Recent Volatility: ${volatility:.2f} per day")

    # ========================================================================
    # MULTI-APPROACH PROBABILITY ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("Running Multi-Approach Win Probability Analysis...")
    print("="*70)

    # Approach 1: Multi-Day Sequential Prediction
    print("\n[1/3] Running multi-day sequential prediction...")
    prediction_result = predict_multi_day_path(
        model=model,
        scaler=scaler_X,  # LSTM uses scaler_X for features
        df=df,  # LSTM uses original df with basic OHLCV features
        feature_cols=feature_cols,  # LSTM uses basic features (no lag features)
        current_price=today_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        model_type='lstm'
    )

    # Approach 2: Monte Carlo Simulation
    print("[2/3] Running Monte Carlo simulation...")
    monte_carlo_result = monte_carlo_simulation(
        current_price=today_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        volatility=volatility,
        predicted_move_pct=expected_move_pct
    )

    # Approach 3: Historical Pattern Matching
    print("[3/3] Searching historical patterns...")
    pattern_result = find_similar_patterns(
        df=df,
        current_price=today_price,
        stop_loss=stop_loss,
        take_profit=take_profit
    )

    # Calculate Ensemble Probability
    ensemble_result = calculate_ensemble_probability(
        prediction_result=prediction_result,
        monte_carlo_result=monte_carlo_result,
        pattern_result=pattern_result
    )

    # Print detailed report
    analysis_report = format_analysis_report(
        prediction_result=prediction_result,
        monte_carlo_result=monte_carlo_result,
        pattern_result=pattern_result,
        ensemble_result=ensemble_result,
        signal=signal,
        current_price=today_price,
        stop_loss=stop_loss,
        take_profit=take_profit
    )
    print(analysis_report)

    # Store results for HTML report (will be parsed by main.py)
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

    # === Step 18: Save Model Info ===
    ticker = os.path.basename(csv_file).split('_')[0]
    model_info = {
        'ticker': ticker,
        'model_type': 'LSTM (Keras + PyTorch + GPU)',
        'backend': keras.backend.backend(),
        'gpu_used': torch.cuda.is_available(),
        'lookback': lookback,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_accuracy': test_acc,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'test_f1': test_f1,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open('lstm_model_info.txt', 'w') as f:
        for key, value in model_info.items():
            f.write(f"{key}: {value}\n")

    print("\nModel saved as: best_lstm_model.keras")
    print("Model info saved as: lstm_model_info.txt")

    return model, history, model_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LSTM model for stock price prediction')
    parser.add_argument('csv_file', type=str, help='Path to CSV file with stock data')
    parser.add_argument('--lookback', type=int, default=60, help='Lookback period (default: 60)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')

    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: File {args.csv_file} not found!")
        exit(1)

    train_lstm_model(args.csv_file, args.lookback, args.epochs, args.batch_size)
