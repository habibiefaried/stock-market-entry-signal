"""
LightGBM Stock Price Prediction Model

This script trains a LightGBM (Light Gradient Boosting Machine) model to predict stock prices.
LightGBM is Microsoft's gradient boosting framework that's often faster and more accurate than XGBoost.

Key Differences from XGBoost:
- Leaf-wise tree growth (vs XGBoost's level-wise): Finds optimal splits faster
- Better memory efficiency: Uses histogram-based algorithms
- Faster training: Especially on large datasets
- Built-in categorical feature support
- Generally better accuracy with less tuning

Key Concepts:
- Gradient Boosting: Trees are added one at a time, each learning from previous mistakes
- Leaf-wise Growth: Grows tree by splitting leaf with max delta loss (more efficient)
- Regularization: Prevents overfitting through num_leaves, min_data_in_leaf, lambda_l1, lambda_l2
- Feature Engineering: Creates lag features and technical indicators for better predictions
"""

import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical computations
from sklearn.preprocessing import StandardScaler  # Feature scaling (normalize data)
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score  # Evaluation metrics
import lightgbm as lgb  # LightGBM library for gradient boosting
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt  # Plotting and visualization
import argparse  # Command-line argument parsing
import os  # Operating system utilities
import joblib  # Model serialization (save/load models)
from datetime import datetime  # Date and time utilities
import warnings  # Warning control
import logging  # Suppress matplotlib logging
warnings.filterwarnings('ignore')  # Disable warnings for cleaner output
logging.getLogger('matplotlib').setLevel(logging.ERROR)  # Suppress matplotlib warnings
logging.getLogger('PIL').setLevel(logging.ERROR)  # Suppress PIL warnings
logging.getLogger('lightgbm').setLevel(logging.ERROR)  # Suppress LightGBM warnings

# Import trade probability analyzer
from trade_probability_analyzer import (
    predict_multi_day_path,
    monte_carlo_simulation,
    find_similar_patterns,
    calculate_ensemble_probability,
    format_analysis_report
)

def load_and_prepare_data(csv_file):
    """
    Load CSV data and prepare basic features for LightGBM

    Args:
        csv_file (str): Path to CSV file containing stock data

    Returns:
        df (DataFrame): Cleaned dataframe with stock data
        feature_cols (list): List of feature column names to use

    What this does:
    - Reads the CSV file with stock data (OHLCV)
    - Removes any rows with missing values (NaN) that would break training
    - Selects relevant features for the model
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

def create_lag_features(df, feature_cols, lags=[1, 2, 3, 5, 10]):
    """
    Create lag features for time series prediction

    LightGBM (like XGBoost) doesn't naturally understand time sequences like LSTM does.
    To give it temporal context, we create "lag features" - past values
    that help the model understand patterns over time.

    Args:
        df (DataFrame): Input dataframe with stock data
        feature_cols (list): Original feature columns (OHLCV)
        lags (list): How many days back to look [1, 2, 3, 5, 10]

    Returns:
        df_lagged (DataFrame): Dataframe with lag features added
        all_features (list): Complete list of feature names

    Example of lag features:
    If today's Close = $100, then:
    - Close_lag_1 = yesterday's close ($98)
    - Close_lag_2 = 2 days ago close ($97)
    This helps model learn: "When price was $97 -> $98 -> $100, what happens next?"
    """
    # IMPORTANT: Only keep Date + feature_cols to avoid NaN from unused technical indicators
    keep_cols = ['Date'] + feature_cols
    df_lagged = df[keep_cols].copy()
    print(f"\nCreating lag features...")
    print(f"Using columns for lag features: {df_lagged.columns.tolist()}")

    # Create lag features for Close price
    # shift(1) moves data down by 1 row, so current row gets previous day's value
    for lag in lags:
        df_lagged[f'Close_lag_{lag}'] = df_lagged['Close'].shift(lag)
        # Example: Close_lag_1 means "closing price 1 day ago"

    # Create lag features for Volume
    # High volume can indicate strong moves, so past volume matters
    for lag in lags:
        df_lagged[f'Volume_lag_{lag}'] = df_lagged['Volume'].shift(lag)

    # Price change features (percentage change)
    # pct_change(1) = (today - yesterday) / yesterday
    # This captures momentum: is price accelerating up or down?
    df_lagged['Price_change_1d'] = df_lagged['Close'].pct_change(1)
    df_lagged['Price_change_5d'] = df_lagged['Close'].pct_change(5)
    df_lagged['Price_change_10d'] = df_lagged['Close'].pct_change(10)

    # Volatility features (standard deviation of price)
    # High volatility = risky, unpredictable price swings
    # Low volatility = stable, predictable movement
    df_lagged['Volatility_5d'] = df_lagged['Close'].rolling(window=5).std()
    df_lagged['Volatility_10d'] = df_lagged['Close'].rolling(window=10).std()

    # Target: Next day's closing price
    # shift(-1) moves data UP by 1 row, so current row gets NEXT day's value
    # This is what we're trying to predict!
    df_lagged['Target'] = df_lagged['Close'].pct_change(5).shift(-5) * 100  # 5-day forward return

    # Drop rows with NaN (created by shift operations)
    # - First few rows: no data for lag features (Close_lag_10 needs 10 prior days)
    # - Last row: no target value (can't predict future beyond our data)
    df_lagged = df_lagged.dropna()

    print(f"Records after creating lag features: {len(df_lagged)}")

    # Get all feature column names (everything except Date and Target)
    # These are our input variables (X) for training
    all_features = [col for col in df_lagged.columns if col not in ['Date', 'Target']]
    print(f"Total features: {len(all_features)}")

    return df_lagged, all_features

def split_train_test(df, train_ratio=9/10):
    """
    Split data into training and testing sets

    IMPORTANT: For time series, we CANNOT use random split!
    We must preserve chronological order:
    - Training data: Past (9/10 of data)
    - Test data: Future (1/10 of data)

    This simulates real-world trading: train on historical data,
    then test on future unseen data.

    Args:
        df (DataFrame): Input dataframe
        train_ratio (float): Proportion for training (9/10 = 90%)

    Returns:
        train_df: Training data
        test_df: Test data

    Why 9/10 split?
    - 48 months total data
    - 43 months (90%) = training (model learns patterns)
    - 5 months (10%) = testing (evaluate real performance)
    """
    split_idx = int(len(df) * train_ratio)

    # CHRONOLOGICAL split - do NOT shuffle!
    train_df = df[:split_idx]  # First 9/10
    test_df = df[split_idx:]   # Last 1/10

    print(f"\nData split:")
    print(f"Training set: {len(train_df)} records ({train_df['Date'].min()} to {train_df['Date'].max()})")
    print(f"Test set: {len(test_df)} records ({test_df['Date'].min()} to {test_df['Date'].max()})")

    return train_df, test_df

def calculate_direction_metrics(y_true, y_pred):
    """
    Calculate classification metrics for price direction

    While our model predicts exact prices (regression), traders care about
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

    Example:
    Day 1: $100 -> Day 2: $102 (UP) [checkmark] predicted correctly = 1
    Day 2: $102 -> Day 3: $101 (DOWN) [x] predicted UP = 0
    Accuracy = 1/2 = 50%
    """
    # Convert consecutive prices to direction changes
    # np.diff([100, 102, 101]) = [2, -1]
    # > 0 converts to True/False, .astype(int) converts to 1/0
    y_true_direction = (np.diff(y_true) > 0).astype(int)  # 1=up, 0=down
    y_pred_direction = (np.diff(y_pred) > 0).astype(int)

    # Classification metrics
    # These tell us how good we are at predicting "will it go up or down?"
    accuracy = accuracy_score(y_true_direction, y_pred_direction)
    precision = precision_score(y_true_direction, y_pred_direction, zero_division=0)
    recall = recall_score(y_true_direction, y_pred_direction, zero_division=0)
    f1 = f1_score(y_true_direction, y_pred_direction, zero_division=0)

    return accuracy, precision, recall, f1

def train_lightgbm_model(csv_file, n_estimators=2000, learning_rate=0.01, num_leaves=31):
    """
    Main training function for LightGBM model

    LightGBM vs XGBoost Key Differences:
    1. Tree Growth Strategy:
       - XGBoost: Level-wise (balanced tree, slower)
       - LightGBM: Leaf-wise (faster, more accurate, can overfit)

    2. Speed:
       - LightGBM is 2-4x faster on large datasets
       - Uses histogram-based algorithms

    3. Memory:
       - LightGBM uses less memory (gradient-based one-side sampling)

    4. Accuracy:
       - Often better out-of-the-box accuracy
       - Less hyperparameter tuning needed

    Args:
        csv_file (str): Path to CSV file with stock data
        n_estimators (int): Number of trees to build
        learning_rate (float): How much each tree contributes
        num_leaves (int): Maximum leaves per tree (LightGBM specific)

    Returns:
        model: Trained LightGBM model
        scaler: Feature scaler
        model_info: Dictionary with model metrics
    """

    # Load data
    df, feature_cols = load_and_prepare_data(csv_file)

    # Create lag features
    df_lagged, all_features = create_lag_features(df, feature_cols)

    # Split train/test
    train_df, test_df = split_train_test(df_lagged)

    # Prepare features and target
    X_train = train_df[all_features].values
    y_train = train_df['Target'].values

    X_test = test_df[all_features].values
    y_test = test_df['Target'].values

    print(f"\nFeature matrix shape:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")

    # Scale features - StandardScaler normalizes each feature to mean=0, std=1
    # Why? Features like Volume (millions) and price changes (%) have very different ranges
    # Scaling puts them on the same scale so the model treats them equally
    # Note: Not strictly required for tree-based models, but can improve performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Learn mean/std from training data
    X_test_scaled = scaler.transform(X_test)  # Apply same scaling to test data

    # Build LightGBM model
    print("\nBuilding LightGBM model...")

    # Try GPU first, fallback to CPU if not available
    # GPU can speed up training 2-4x for large datasets
    _using_gpu = False

    def _make_lgb_model(use_gpu):
        params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_data_in_leaf=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=42,
            verbose=-1,
        )
        if use_gpu:
            params['device'] = 'gpu'
        return lgb.LGBMRegressor(**params)

    # Train model
    # LightGBM builds trees sequentially: Tree1 -> Tree2 -> Tree3 -> ...
    # Each new tree learns to correct the errors of all previous trees
    print("\nTraining LightGBM model...")
    try:
        model = _make_lgb_model(use_gpu=True)
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
            eval_names=['train', 'test'],
            callbacks=[lgb.log_evaluation(period=50)],
        )
        _using_gpu = True
        print("Using GPU acceleration (CUDA)")
    except Exception as e:
        print(f"GPU not available, falling back to CPU: {e}")
        model = _make_lgb_model(use_gpu=False)
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
            eval_names=['train', 'test'],
            callbacks=[lgb.log_evaluation(period=50)],
        )

    # Make predictions (returns in %)
    print("\nMaking predictions...")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Convert returns to prices for MAE/RMSE
    close_idx = all_features.index('Close')
    prev_train = X_train[:, close_idx]
    prev_test  = X_test[:, close_idx]
    y_train_price = prev_train * (1 + y_train / 100)
    y_test_price  = prev_test  * (1 + y_test  / 100)
    y_train_pred_price = prev_train * (1 + y_train_pred / 100)
    y_test_pred_price  = prev_test  * (1 + y_test_pred  / 100)

    train_mae = mean_absolute_error(y_train_price, y_train_pred_price)
    train_rmse = np.sqrt(mean_squared_error(y_train_price, y_train_pred_price))
    test_mae = mean_absolute_error(y_test_price, y_test_pred_price)
    test_rmse = np.sqrt(mean_squared_error(y_test_price, y_test_pred_price))

    train_dir_actual = (y_train > 0).astype(int)
    train_dir_pred   = (y_train_pred > 0).astype(int)
    test_dir_actual  = (y_test > 0).astype(int)
    test_dir_pred    = (y_test_pred > 0).astype(int)
    train_acc  = accuracy_score(train_dir_actual, train_dir_pred)
    train_prec = precision_score(train_dir_actual, train_dir_pred, zero_division=0)
    train_rec  = recall_score(train_dir_actual, train_dir_pred, zero_division=0)
    train_f1   = f1_score(train_dir_actual, train_dir_pred, zero_division=0)
    test_acc  = accuracy_score(test_dir_actual, test_dir_pred)
    test_prec = precision_score(test_dir_actual, test_dir_pred, zero_division=0)
    test_rec  = recall_score(test_dir_actual, test_dir_pred, zero_division=0)
    test_f1   = f1_score(test_dir_actual, test_dir_pred, zero_division=0)

    # Print results
    print("\n" + "="*60)
    print("LIGHTGBM MODEL EVALUATION RESULTS")
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

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15).to_string(index=False))

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importances (LightGBM)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('lightgbm_feature_importance.png')
    print("\nFeature importance plot saved as: lightgbm_feature_importance.png")

    # Plot predictions
    plt.figure(figsize=(15, 6))
    test_dates = test_df['Date'].values
    plt.plot(test_dates, y_test_price, label='Actual Price', color='blue', linewidth=2)
    plt.plot(test_dates, y_test_pred_price, label='Predicted Price', color='red', linewidth=2, alpha=0.7)
    plt.title('LightGBM Model: Actual vs Predicted Prices (Test Set)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lightgbm_predictions.png')
    print("Predictions plot saved as: lightgbm_predictions.png")

    # === Generate Trading Signal ===
    print("\n" + "="*60)
    print("TRADING SIGNAL FOR NEXT DAY")
    print("="*60)

    # Get today's actual price (last row of lagged data before dropna)
    today_price = df[feature_cols + ['Date']].iloc[-1]['Close']

    # Create features for tomorrow's prediction
    # Use the most recent data point from df_lagged (has all lag features)
    recent_features = df_lagged[all_features].iloc[-1:].values
    recent_features_scaled = scaler.transform(recent_features)

    # Predict tomorrow's % return
    tomorrow_return = model.predict(recent_features_scaled)[0]
    expected_move_pct = tomorrow_return
    tomorrow_pred_price = today_price * (1 + tomorrow_return / 100)
    expected_move = tomorrow_pred_price - today_price

    # Adaptive threshold: 0.5x daily vol (min 0.5%)
    recent_ret_pct = df['Close'].pct_change().tail(20).std() * 100
    sig_threshold  = max(0.15 * recent_ret_pct, 0.1)

    if expected_move_pct > sig_threshold:
        signal = "BUY (LONG)"
        signal_emoji = "[BUY]"
    elif expected_move_pct < -sig_threshold:
        signal = "SHORT (SELL)"
        signal_emoji = "[SHORT]"
    else:
        signal = "HOLD (No clear signal)"
        signal_emoji = "[HOLD]"

    # ATR-based TP/SL
    h, l, c = df['High'], df['Low'], df['Close']
    tr  = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(span=14, min_periods=14).mean().iloc[-1])
    stop_loss_distance   = 1.5 * atr
    take_profit_distance = 2.05 * atr
    volatility           = df['Close'].tail(20).pct_change().dropna().std() * today_price

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
    print(f"Predicted Price (Tomorrow): ${tomorrow_pred_price:.2f}")
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
        scaler=scaler,
        df=df_lagged,  # Use the LAGGED dataframe with engineered features
        feature_cols=all_features,  # Use ALL engineered features, not just basic OHLCV
        current_price=today_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        model_type='gbm'
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
        df=df,  # Use ORIGINAL df (not lagged) for pattern matching - it only needs OHLCV
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

    # Save model
    joblib.dump(model, 'lightgbm_model.pkl')
    joblib.dump(scaler, 'lightgbm_scaler.pkl')

    # Save feature names
    with open('lightgbm_features.txt', 'w') as f:
        f.write('\n'.join(all_features))

    # Save model info
    ticker = os.path.basename(csv_file).split('_')[0]
    model_info = {
        'ticker': ticker,
        'model_type': f'LightGBM ({"GPU" if _using_gpu else "CPU"})',
        'n_features': len(all_features),
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

    # Save model info to file
    with open('lightgbm_model_info.txt', 'w') as f:
        for key, value in model_info.items():
            f.write(f"{key}: {value}\n")

    print("\nModel saved as: lightgbm_model.pkl")
    print("Scaler saved as: lightgbm_scaler.pkl")
    print("Features saved as: lightgbm_features.txt")
    print("Model info saved as: lightgbm_model_info.txt")

    return model, scaler, model_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train LightGBM model for stock price prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage (default parameters)
  python train_lightgbm.py MSFT_daily_data_20260520.csv

  # Custom parameters
  python train_lightgbm.py MSFT_daily_data_20260520.csv --n_estimators 1000 --learning_rate 0.01 --num_leaves 31

LightGBM vs XGBoost:
  - LightGBM is often faster and more accurate
  - Uses leaf-wise tree growth (more efficient than XGBoost's level-wise)
  - Better memory efficiency
  - Less hyperparameter tuning needed
        '''
    )

    parser.add_argument('csv_file', type=str, help='Path to CSV file with stock data')
    parser.add_argument('--n_estimators', type=int, default=2000, help='Number of trees (default: 2000)')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--num_leaves', type=int, default=31, help='Max leaves per tree (default: 31)')

    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: File {args.csv_file} not found!")
        exit(1)

    train_lightgbm_model(args.csv_file, args.n_estimators, args.learning_rate, args.num_leaves)
