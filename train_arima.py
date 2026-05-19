"""
ARIMA Stock Price Prediction Model

This script uses ARIMA (AutoRegressive Integrated Moving Average) for stock price prediction.
ARIMA is a STATISTICAL model, NOT a machine learning model - it uses pure mathematics
to forecast future values based on past patterns.

Key Concepts:

ARIMA vs Machine Learning:
- ML models (LSTM, XGBoost): Learn patterns from data through training
- ARIMA: Uses statistical formulas to model time series behavior
- No "training" in ML sense - just fits mathematical parameters
- Much faster than ML (seconds vs minutes)
- More interpretable - can see exact formula

ARIMA Components (p, d, q):

1. AR (AutoRegressive) - p parameter:
   - Uses past values to predict future
   - p=5 means: use last 5 days to predict tomorrow
   - Formula: price_t = c + φ₁*price_{t-1} + φ₂*price_{t-2} + ... + φₚ*price_{t-p}
   - Similar to linear regression with lag features

2. I (Integrated) - d parameter:
   - Makes data "stationary" (removes trends)
   - d=0: No differencing (data already stationary)
   - d=1: First difference (price_t - price_{t-1}) = daily change
   - d=2: Second difference (change of change) = acceleration
   - Stock prices are NOT stationary (they trend), so d≥1 needed

3. MA (Moving Average) - q parameter:
   - Uses past forecast ERRORS to predict future
   - q=5 means: use last 5 prediction errors
   - Formula: price_t = c + ε_t + θ₁*ε_{t-1} + θ₂*ε_{t-2} + ... + θₑ*ε_{t-q}
   - Helps smooth out random shocks

Example ARIMA(5,1,5):
- Use last 5 prices (AR=5)
- Take first difference to remove trend (I=1)
- Use last 5 errors (MA=5)

Stationarity Explained:
- Stationary: Mean and variance don't change over time (flat, no trend)
- Non-stationary: Has trend (going up/down) or changing volatility
- Stock prices are NON-stationary (they trend up/down)
- ARIMA's "I" (differencing) makes them stationary
- Example: MSFT price $85 → $90 → $95 (trending up, non-stationary)
           Daily change: +$5 → +$5 (flat, stationary)
"""

import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical computations
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score  # Evaluation metrics
import matplotlib.pyplot as plt  # Plotting and visualization
import argparse  # Command-line argument parsing
import os  # Operating system utilities
import joblib  # Model serialization (save/load models)
from datetime import datetime  # Date and time utilities
import warnings  # Suppress warnings
warnings.filterwarnings('ignore')  # ARIMA produces many convergence warnings

# Import ARIMA libraries from statsmodels
# statsmodels is the standard Python library for statistical modeling
from statsmodels.tsa.arima.model import ARIMA  # ARIMA implementation
from statsmodels.tsa.stattools import adfuller  # Test for stationarity (ADF test)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # Diagnostic plots
from statsmodels.tsa.statespace.tools import diff  # Differencing function

def load_and_prepare_data(csv_file):
    """
    Load CSV data and prepare for ARIMA

    IMPORTANT: ARIMA is UNIVARIATE - it only uses ONE variable (Close price)
    Unlike ML models that use multiple features (OHLCV + lag features),
    ARIMA learns patterns from Close prices alone.

    Why only Close price?
    - ARIMA models the time series itself, not relationships between features
    - Open, High, Low, Volume are ignored
    - This is a limitation but also a strength (simpler, more interpretable)

    Args:
        csv_file (str): Path to CSV file containing stock data

    Returns:
        df (DataFrame): Dataframe with Date and Close columns
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    # ARIMA only needs Date and Close
    # We ignore Open, High, Low, Volume (ARIMA doesn't use them)
    if 'Date' not in df.columns or 'Close' not in df.columns:
        raise ValueError("CSV must contain 'Date' and 'Close' columns")

    # Convert Date to datetime and set as index
    # ARIMA needs datetime index to understand time intervals
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Close']].copy()
    df.set_index('Date', inplace=True)

    # Sort by date (ensure chronological order)
    df.sort_index(inplace=True)

    print(f"Using only Close prices for ARIMA (univariate time series)")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    return df

def check_stationarity(timeseries, name="Time Series"):
    """
    Check if time series is stationary using ADF (Augmented Dickey-Fuller) test

    Stationarity is CRITICAL for ARIMA:
    - Stationary: Mean and variance constant over time (no trend)
    - Non-stationary: Has trend or changing variance
    - Stock prices are almost ALWAYS non-stationary (they trend)
    - Need differencing (I parameter) to make stationary

    ADF Test:
    - Null hypothesis (H0): Time series is NON-stationary (has unit root)
    - Alternative (H1): Time series IS stationary
    - If p-value < 0.05: Reject H0 → stationary
    - If p-value > 0.05: Fail to reject H0 → non-stationary

    Args:
        timeseries (Series): Time series data to test
        name (str): Name for printing

    Returns:
        is_stationary (bool): True if stationary, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Stationarity Test: {name}")
    print(f"{'='*60}")

    # Run ADF test
    result = adfuller(timeseries.dropna(), autolag='AIC')

    # Extract results
    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]

    print(f"ADF Statistic: {adf_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Critical Values:")
    for key, value in critical_values.items():
        print(f"  {key}: {value:.4f}")

    # Interpret results
    if p_value < 0.05:
        print(f"\n✓ Result: STATIONARY (p-value < 0.05)")
        print(f"  → Time series has NO unit root")
        print(f"  → Mean and variance are stable over time")
        print(f"  → Can use ARIMA with d=0 (no differencing needed)")
        is_stationary = True
    else:
        print(f"\n✗ Result: NON-STATIONARY (p-value ≥ 0.05)")
        print(f"  → Time series HAS unit root (has trend)")
        print(f"  → Mean/variance change over time")
        print(f"  → Need differencing: use ARIMA with d≥1")
        is_stationary = False

    return is_stationary

def split_train_test(df, train_ratio=5/6):
    """
    Split data into training and testing sets

    IMPORTANT: For time series, we CANNOT use random split!
    We must preserve chronological order:
    - Training data: Past (5/6 of data)
    - Test data: Future (1/6 of data)

    This simulates real-world trading: learn from history,
    then forecast the future.

    Args:
        df (DataFrame): Input dataframe with DatetimeIndex
        train_ratio (float): Proportion for training (5/6 = 83.3%)

    Returns:
        train_df: Training data
        test_df: Test data
    """
    split_idx = int(len(df) * train_ratio)

    # CHRONOLOGICAL split - do NOT shuffle!
    train_df = df[:split_idx]  # First 5/6
    test_df = df[split_idx:]   # Last 1/6

    print(f"\nData split:")
    print(f"Training set: {len(train_df)} records ({train_df.index.min()} to {train_df.index.max()})")
    print(f"Test set: {len(test_df)} records ({test_df.index.min()} to {test_df.index.max()})")

    return train_df, test_df

def find_arima_order(train_data, max_p=5, max_d=2, max_q=5):
    """
    Find optimal ARIMA order (p, d, q) using grid search

    This function tests different combinations of (p, d, q) and picks
    the one with lowest AIC (Akaike Information Criterion).

    AIC (Akaike Information Criterion):
    - Measures model quality: fit vs complexity
    - Lower AIC = better model
    - Formula: AIC = 2k - 2ln(L)
      - k = number of parameters
      - L = likelihood (how well model fits data)
    - Penalizes complex models (prevents overfitting)

    Parameter Ranges:
    - p (AR order): 0-5 (how many past values to use)
    - d (differencing): 0-2 (usually 1 for stock prices)
    - q (MA order): 0-5 (how many past errors to use)

    Args:
        train_data (Series): Training time series
        max_p (int): Maximum AR order to test
        max_d (int): Maximum differencing order to test
        max_q (int): Maximum MA order to test

    Returns:
        best_order (tuple): Best (p, d, q) values
    """
    print("\n" + "="*60)
    print("Finding optimal ARIMA parameters...")
    print("="*60)
    print(f"Testing combinations: p=[0-{max_p}], d=[0-{max_d}], q=[0-{max_q}]")
    print(f"This will test {(max_p+1)*(max_d+1)*(max_q+1)} models...")

    best_aic = np.inf
    best_order = None
    results = []

    # Grid search over all combinations
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    # Fit ARIMA model with these parameters
                    model = ARIMA(train_data, order=(p, d, q))
                    fitted_model = model.fit()
                    aic = fitted_model.aic

                    # Track result
                    results.append({
                        'order': (p, d, q),
                        'aic': aic
                    })

                    # Update best model if this one is better
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)

                except Exception as e:
                    # Some combinations don't converge, skip them
                    continue

    print(f"\nTested {len(results)} valid models")
    print(f"Best order: ARIMA{best_order}")
    print(f"Best AIC: {best_aic:.2f}")

    # Show top 5 models
    results_df = pd.DataFrame(results).sort_values('aic')
    print(f"\nTop 5 models:")
    print(results_df.head(5).to_string(index=False))

    return best_order

def calculate_direction_metrics(y_true, y_pred):
    """
    Calculate classification metrics for price direction

    While ARIMA predicts exact prices (regression), traders care about
    DIRECTION: will price go up or down tomorrow?

    This function converts price predictions to up/down and calculates:
    - Accuracy: % of correct direction predictions
    - Precision: Of predicted "ups", how many were actually up?
    - Recall: Of actual "ups", how many did we predict?
    - F1-Score: Balance of precision and recall

    Args:
        y_true (array): Actual prices
        y_pred (array): Predicted prices

    Returns:
        accuracy, precision, recall, f1 (floats): Classification metrics
    """
    # Convert consecutive prices to direction changes
    # np.diff([100, 102, 101]) = [2, -1]
    # > 0 converts to True/False, .astype(int) converts to 1/0
    y_true_direction = (np.diff(y_true) > 0).astype(int)  # 1=up, 0=down
    y_pred_direction = (np.diff(y_pred) > 0).astype(int)

    # Classification metrics
    accuracy = accuracy_score(y_true_direction, y_pred_direction)
    precision = precision_score(y_true_direction, y_pred_direction, zero_division=0)
    recall = recall_score(y_true_direction, y_pred_direction, zero_division=0)
    f1 = f1_score(y_true_direction, y_pred_direction, zero_division=0)

    return accuracy, precision, recall, f1

def train_arima_model(csv_file, order=None, auto_tune=True):
    """
    Main training function for ARIMA model

    ARIMA Workflow:
    1. Load Close prices (univariate time series)
    2. Check stationarity (ADF test)
    3. Find optimal (p, d, q) parameters (grid search)
    4. Fit ARIMA model (statistical fitting, not ML training)
    5. Make predictions
    6. Evaluate performance

    Args:
        csv_file (str): Path to CSV file with stock data
        order (tuple): Manual (p, d, q) specification, e.g., (5, 1, 5)
        auto_tune (bool): If True, automatically find best parameters

    Returns:
        model: Fitted ARIMA model
        model_info: Dictionary with model metrics
    """

    # Load data
    df = load_and_prepare_data(csv_file)

    # Check stationarity of original data
    is_stationary = check_stationarity(df['Close'], "Original Close Prices")

    # If data is non-stationary, check first difference
    if not is_stationary:
        print("\nTesting first difference (daily change)...")
        df_diff = df['Close'].diff().dropna()
        is_diff_stationary = check_stationarity(df_diff, "First Difference (Daily Change)")

        if is_diff_stationary:
            print("\n→ Recommendation: Use ARIMA with d=1 (first differencing)")
        else:
            print("\n→ Recommendation: Use ARIMA with d=2 (second differencing)")

    # Split train/test
    train_df, test_df = split_train_test(df)
    train_data = train_df['Close']
    test_data = test_df['Close']

    # Find optimal ARIMA order
    if auto_tune and order is None:
        order = find_arima_order(train_data)
        print(f"\nAuto-tuned ARIMA order: {order}")
    elif order is None:
        # Default order if not specified
        order = (5, 1, 5)  # Common default for stock prices
        print(f"\nUsing default ARIMA order: {order}")
    else:
        print(f"\nUsing manual ARIMA order: {order}")

    p, d, q = order

    # Build ARIMA model
    print("\n" + "="*60)
    print(f"Fitting ARIMA{order} model...")
    print("="*60)
    print(f"\nModel Parameters:")
    print(f"  p (AR order): {p} - Uses last {p} prices")
    print(f"  d (Differencing): {d} - Takes {d}-order difference")
    print(f"  q (MA order): {q} - Uses last {q} errors")

    # Fit ARIMA model to training data
    # This is NOT machine learning training!
    # It's maximum likelihood estimation - finding parameters that
    # best fit the statistical model to the observed data
    model = ARIMA(train_data, order=order)
    fitted_model = model.fit()

    print("\nModel fitted successfully!")
    print(f"AIC: {fitted_model.aic:.2f}")
    print(f"BIC: {fitted_model.bic:.2f}")

    # Print model summary (shows coefficients)
    print("\n" + "="*60)
    print("Model Summary:")
    print("="*60)
    print(fitted_model.summary())

    # Make predictions on training set
    print("\nMaking in-sample predictions (training set)...")
    train_pred = fitted_model.fittedvalues

    # Make predictions on test set
    # One-step-ahead forecasting: predict each day using actual previous days
    print("Making out-of-sample predictions (test set)...")
    test_pred = []
    history = train_data.values.tolist()

    for i in range(len(test_data)):
        # Fit ARIMA on history (expanding window)
        model_temp = ARIMA(history, order=order)
        model_fit = model_temp.fit()

        # Forecast next day
        forecast = model_fit.forecast(steps=1)[0]
        test_pred.append(forecast)

        # Add actual value to history for next iteration
        actual = test_data.iloc[i]
        history.append(actual)

        # Print progress every 20 days
        if (i + 1) % 20 == 0:
            print(f"  Predicted {i+1}/{len(test_data)} days...")

    test_pred = np.array(test_pred)

    # Calculate regression metrics
    # Note: train_pred may have NaN values at the start due to differencing
    train_pred_clean = train_pred.dropna()
    train_actual_clean = train_data[train_pred.index]

    train_mae = mean_absolute_error(train_actual_clean, train_pred_clean)
    train_rmse = np.sqrt(mean_squared_error(train_actual_clean, train_pred_clean))
    test_mae = mean_absolute_error(test_data, test_pred)
    test_rmse = np.sqrt(mean_squared_error(test_data, test_pred))

    # Calculate direction metrics
    train_acc, train_prec, train_rec, train_f1 = calculate_direction_metrics(
        train_actual_clean.values, train_pred_clean.values
    )
    test_acc, test_prec, test_rec, test_f1 = calculate_direction_metrics(
        test_data.values, test_pred
    )

    # Print results
    print("\n" + "="*60)
    print("ARIMA MODEL EVALUATION RESULTS")
    print("="*60)

    print(f"\nModel: ARIMA{order}")
    print(f"  p={p}: Uses last {p} price values")
    print(f"  d={d}: {'No differencing' if d==0 else f'Takes {d}-order difference'}")
    print(f"  q={q}: Uses last {q} forecast errors")

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

    # Plot predictions
    plt.figure(figsize=(15, 6))
    plt.plot(test_data.index, test_data.values, label='Actual Price', color='blue', linewidth=2)
    plt.plot(test_data.index, test_pred, label='Predicted Price', color='red', linewidth=2, alpha=0.7)
    plt.title(f'ARIMA{order} Model: Actual vs Predicted Prices (Test Set)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('arima_predictions.png')
    print("\nPredictions plot saved as: arima_predictions.png")

    # Plot residuals (forecast errors)
    # Residuals should be random (white noise) if model is good
    plt.figure(figsize=(15, 10))

    # Residual plot over time
    plt.subplot(3, 1, 1)
    residuals = fitted_model.resid
    plt.plot(residuals)
    plt.title('Residuals (Forecast Errors) Over Time')
    plt.ylabel('Residual')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.grid(True)

    # Residual histogram
    plt.subplot(3, 1, 2)
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.title('Residual Distribution (Should be Normal)')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.grid(True)

    # ACF of residuals (should show no correlation)
    plt.subplot(3, 1, 3)
    plot_acf(residuals.dropna(), lags=40, ax=plt.gca())
    plt.title('ACF of Residuals (Should be within blue bands = no correlation)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('arima_diagnostics.png')
    print("Diagnostic plots saved as: arima_diagnostics.png")

    # Save model
    joblib.dump(fitted_model, 'arima_model.pkl')

    # Save model info
    ticker = os.path.basename(csv_file).split('_')[0]
    model_info = {
        'ticker': ticker,
        'model_type': f'ARIMA{order}',
        'p': p,
        'd': d,
        'q': q,
        'aic': fitted_model.aic,
        'bic': fitted_model.bic,
        'train_size': len(train_data),
        'test_size': len(test_data),
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_accuracy': test_acc,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'test_f1': test_f1,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Save model info to file
    with open('arima_model_info.txt', 'w') as f:
        for key, value in model_info.items():
            f.write(f"{key}: {value}\n")

    print("\nModel saved as: arima_model.pkl")
    print("Model info saved as: arima_model_info.txt")

    return fitted_model, model_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train ARIMA model for stock price prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Auto-tune parameters (recommended)
  python train_arima.py MSFT_daily_data_20260520.csv

  # Manual parameters
  python train_arima.py MSFT_daily_data_20260520.csv --p 5 --d 1 --q 5 --no-auto-tune

ARIMA vs Machine Learning:
  - ARIMA: Statistical model (pure math, no training)
  - ML: Learns patterns from data
  - ARIMA: Faster, more interpretable
  - ML: Can use multiple features (OHLCV + lag features)
  - ARIMA: Only uses Close prices (univariate)

ARIMA Parameters:
  p: AR order (how many past prices to use)
  d: Differencing (0=none, 1=daily change, 2=change of change)
  q: MA order (how many past errors to use)

Typical values for stock prices:
  - ARIMA(5,1,5): Common default
  - ARIMA(1,1,1): Simple model
  - Auto-tune: Let algorithm find best (p,d,q)
        '''
    )

    parser.add_argument('csv_file', type=str, help='Path to CSV file with stock data')
    parser.add_argument('--p', type=int, default=None, help='AR order (default: auto-tune)')
    parser.add_argument('--d', type=int, default=None, help='Differencing order (default: auto-tune)')
    parser.add_argument('--q', type=int, default=None, help='MA order (default: auto-tune)')
    parser.add_argument('--no-auto-tune', action='store_true', help='Disable auto-tuning')

    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: File {args.csv_file} not found!")
        exit(1)

    # Determine order
    if args.p is not None and args.d is not None and args.q is not None:
        order = (args.p, args.d, args.q)
        auto_tune = False
    else:
        order = None
        auto_tune = not args.no_auto_tune

    train_arima_model(args.csv_file, order=order, auto_tune=auto_tune)
