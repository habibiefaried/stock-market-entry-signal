"""
Random Forest Stock Price Prediction Model with Walk-Forward Validation

This script trains a Random Forest model to predict stock prices using walk-forward validation.
Random Forest is an ensemble of decision trees that reduces overfitting through bootstrap aggregation.

Key Concepts:
- Ensemble Learning: Combines predictions from multiple decision trees
- Bootstrap Aggregation (Bagging): Each tree trained on random subset of data
- Walk-Forward Validation: Realistic backtesting that simulates live trading
- Feature Importance: Shows which features are most predictive

Walk-Forward Validation:
Instead of single train/test split, we use rolling windows:
1. Train on window 1 -> Test on window 2
2. Train on windows 1-2 -> Test on window 3
3. Train on windows 1-3 -> Test on window 4
... and so on. This prevents look-ahead bias and tests model on truly unseen data.
"""

import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical computations
from sklearn.preprocessing import StandardScaler  # Feature scaling (normalize data)
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score  # Evaluation metrics
from sklearn.ensemble import RandomForestRegressor  # Random Forest for regression
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
    Load CSV data and prepare basic features for Random Forest

    Args:
        csv_file (str): Path to CSV file containing stock data

    Returns:
        df (DataFrame): Cleaned dataframe with stock data
        feature_cols (list): List of feature column names to use
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    # Select ONLY OHLCV features for training
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Check for NaN values
    nan_count = df[feature_cols].isna().sum().sum()
    if nan_count > 0:
        print(f"Found {nan_count} NaN values, removing rows with NaN...")
        df = df.dropna(subset=feature_cols)
    else:
        print(f"Total records: {len(df)} (no NaN removal needed for OHLCV)")

    print(f"Using features: {feature_cols}")

    return df, feature_cols

def create_lag_features(df, feature_cols, lags=[1, 2, 3, 5, 10]):
    """
    Create lag features and technical indicators for Random Forest

    Args:
        df (DataFrame): Input dataframe with OHLCV data
        feature_cols (list): List of base feature columns
        lags (list): List of lag periods to create

    Returns:
        df_lagged (DataFrame): Dataframe with lag features added
        all_features (list): List of all feature column names (including lags)
    """
    print("\nCreating lag features...")

    # Keep Date column if it exists
    keep_cols = ['Date'] if 'Date' in df.columns else []
    keep_cols.extend(feature_cols)
    df_lagged = df[keep_cols].copy()

    print(f"Using columns for lag features: {keep_cols}")

    # Create lag features for Close and Volume
    for lag in lags:
        df_lagged[f'Close_lag_{lag}'] = df_lagged['Close'].shift(lag)
        df_lagged[f'Volume_lag_{lag}'] = df_lagged['Volume'].shift(lag)

    # Create price change features (percentage)
    df_lagged['Price_change_1d'] = df_lagged['Close'].pct_change(1) * 100
    df_lagged['Price_change_5d'] = df_lagged['Close'].pct_change(5) * 100
    df_lagged['Price_change_10d'] = df_lagged['Close'].pct_change(10) * 100

    # Create volatility features (rolling standard deviation)
    # High volatility = risky, unpredictable price swings
    # Low volatility = stable, predictable movement
    df_lagged['Volatility_5d'] = df_lagged['Close'].rolling(window=5).std()
    df_lagged['Volatility_10d'] = df_lagged['Close'].rolling(window=10).std()

    # Create rolling averages
    df_lagged['MA_5'] = df_lagged['Close'].rolling(window=5).mean()
    df_lagged['MA_10'] = df_lagged['Close'].rolling(window=10).mean()
    df_lagged['MA_20'] = df_lagged['Close'].rolling(window=20).mean()

    # Create target: next day's closing price
    # We predict tomorrow's close based on today's features
    df_lagged['Target'] = df_lagged['Close'].shift(-1)

    # Drop rows with NaN (from lag/rolling operations)
    # This is necessary because first N rows don't have enough history
    original_len = len(df_lagged)
    df_lagged = df_lagged.dropna()
    print(f"Records after creating lag features: {len(df_lagged)}")

    # Get all feature columns (everything except Date and Target)
    all_features = [col for col in df_lagged.columns if col not in ['Date', 'Target']]
    print(f"Total features: {len(all_features)}")

    return df_lagged, all_features

def walk_forward_validation(df, features, n_splits=5):
    """
    Perform walk-forward validation on time series data

    This mimics real trading: train on past data, test on future data, then roll forward.

    Example with 1000 days, 5 splits:
    - Split 1: Train on days 1-600, test on days 601-800
    - Split 2: Train on days 1-800, test on days 801-900
    - Split 3: Train on days 1-900, test on days 901-950
    - Split 4: Train on days 1-950, test on days 951-975
    - Split 5: Train on days 1-975, test on days 976-1000

    Args:
        df (DataFrame): Dataframe with features and target
        features (list): List of feature column names
        n_splits (int): Number of validation splits (default: 5)

    Returns:
        splits (list): List of (train_idx, test_idx) tuples
    """
    n_samples = len(df)

    # Use 70% for initial training, then test on remaining in chunks
    initial_train_size = int(n_samples * 0.7)

    splits = []
    test_chunk_size = (n_samples - initial_train_size) // n_splits

    for i in range(n_splits):
        train_end = initial_train_size + (i * test_chunk_size)
        test_start = train_end
        test_end = min(test_start + test_chunk_size, n_samples)

        if test_end <= test_start:
            break

        train_idx = list(range(0, train_end))
        test_idx = list(range(test_start, test_end))

        splits.append((train_idx, test_idx))

    return splits

def train_randomforest_model(csv_file, n_estimators=500, max_depth=15, max_features=30):
    """
    Train Random Forest model with walk-forward validation

    Args:
        csv_file (str): Path to CSV file
        n_estimators (int): Number of trees in the forest (default: 500)
        max_depth (int): Maximum depth of each tree (default: 15)
        max_features (int): Max features to consider for each split (default: 30)
    """
    print("="*60)
    print("RANDOM FOREST MODEL WITH WALK-FORWARD VALIDATION")
    print("="*60)

    # Load and prepare data
    df, feature_cols = load_and_prepare_data(csv_file)

    # Create lag features
    df_lagged, all_features = create_lag_features(df, feature_cols)

    # Limit features if max_features is less than total
    if max_features and max_features < len(all_features):
        # Select most correlated features with target
        correlations = df_lagged[all_features].corrwith(df_lagged['Target']).abs()
        top_features = correlations.nlargest(max_features).index.tolist()
        print(f"\nSelecting top {max_features} features by correlation with target")
        all_features = top_features

    print(f"Using {len(all_features)} features for training")

    # Prepare features and target
    X = df_lagged[all_features].values
    y = df_lagged['Target'].values

    print(f"\nData split:")
    print(f"Total samples: {len(X)}")

    # Initialize scaler
    scaler = StandardScaler()

    # Walk-forward validation
    print("\n" + "="*60)
    print("WALK-FORWARD VALIDATION")
    print("="*60)

    splits = walk_forward_validation(df_lagged, all_features, n_splits=5)

    all_train_preds = []
    all_train_actuals = []
    all_test_preds = []
    all_test_actuals = []
    all_train_directions = []
    all_test_directions = []

    for split_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"\nFold {split_idx + 1}/{len(splits)}")
        print(f"Training samples: {len(train_idx)}, Test samples: {len(test_idx)}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features='sqrt',  # Use sqrt(n_features) for each split
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
            verbose=0
        )

        model.fit(X_train_scaled, y_train)

        # Make predictions
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)

        # Store predictions
        all_train_preds.extend(train_pred)
        all_train_actuals.extend(y_train)
        all_test_preds.extend(test_pred)
        all_test_actuals.extend(y_test)

        # Direction predictions (up/down)
        train_prev_prices = X_train[:, all_features.index('Close')]
        test_prev_prices = X_test[:, all_features.index('Close')]

        train_dir_pred = (train_pred > train_prev_prices).astype(int)
        train_dir_actual = (y_train > train_prev_prices).astype(int)
        test_dir_pred = (test_pred > test_prev_prices).astype(int)
        test_dir_actual = (y_test > test_prev_prices).astype(int)

        all_train_directions.extend(zip(train_dir_actual, train_dir_pred))
        all_test_directions.extend(zip(test_dir_actual, test_dir_pred))

    # Convert to numpy arrays
    all_train_preds = np.array(all_train_preds)
    all_train_actuals = np.array(all_train_actuals)
    all_test_preds = np.array(all_test_preds)
    all_test_actuals = np.array(all_test_actuals)

    # Calculate metrics
    train_mae = mean_absolute_error(all_train_actuals, all_train_preds)
    train_rmse = np.sqrt(mean_squared_error(all_train_actuals, all_train_preds))
    test_mae = mean_absolute_error(all_test_actuals, all_test_preds)
    test_rmse = np.sqrt(mean_squared_error(all_test_actuals, all_test_preds))

    # Direction metrics
    train_dir_actual = np.array([d[0] for d in all_train_directions])
    train_dir_pred = np.array([d[1] for d in all_train_directions])
    test_dir_actual = np.array([d[0] for d in all_test_directions])
    test_dir_pred = np.array([d[1] for d in all_test_directions])

    train_acc = accuracy_score(train_dir_actual, train_dir_pred)
    train_prec = precision_score(train_dir_actual, train_dir_pred, zero_division=0)
    train_rec = recall_score(train_dir_actual, train_dir_pred, zero_division=0)
    train_f1 = f1_score(train_dir_actual, train_dir_pred, zero_division=0)

    test_acc = accuracy_score(test_dir_actual, test_dir_pred)
    test_prec = precision_score(test_dir_actual, test_dir_pred, zero_division=0)
    test_rec = recall_score(test_dir_actual, test_dir_pred, zero_division=0)
    test_f1 = f1_score(test_dir_actual, test_dir_pred, zero_division=0)

    print("\n" + "="*60)
    print("RANDOM FOREST MODEL EVALUATION RESULTS (WALK-FORWARD)")
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

    # Train final model on all data for production use
    print("\nTraining final model on all data...")
    X_all_scaled = scaler.fit_transform(X)

    final_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features='sqrt',
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    final_model.fit(X_all_scaled, y)

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15).to_string(index=False))

    # Save feature importance plot
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Random Forest Feature Importance (Top 20)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('randomforest_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nFeature importance plot saved as: randomforest_feature_importance.png")

    # Generate predictions plot
    plt.figure(figsize=(14, 7))

    # Plot last 100 test predictions
    plot_range = min(100, len(all_test_actuals))
    indices = range(plot_range)

    plt.plot(indices, all_test_actuals[-plot_range:], label='Actual Price', linewidth=2, color='#2c3e50')
    plt.plot(indices, all_test_preds[-plot_range:], label='Predicted Price', linewidth=2, color='#e74c3c', linestyle='--')
    plt.xlabel('Time Steps (Last 100 Test Samples)')
    plt.ylabel('Price ($)')
    plt.title('Random Forest: Actual vs Predicted Prices (Walk-Forward Validation)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('randomforest_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Predictions plot saved as: randomforest_predictions.png")

    # ========================================================================
    # TRADING SIGNAL FOR NEXT DAY
    # ========================================================================
    print("\n" + "="*60)
    print("TRADING SIGNAL FOR NEXT DAY")
    print("="*60)

    # Get today's price (last row in dataset)
    today_price = df_lagged['Close'].iloc[-1]

    # Predict tomorrow's price
    recent_features = df_lagged[all_features].iloc[-1:].values
    recent_features_scaled = scaler.transform(recent_features)
    tomorrow_pred = final_model.predict(recent_features_scaled)[0]

    # Calculate expected move
    expected_move = tomorrow_pred - today_price
    expected_move_pct = (expected_move / today_price) * 100

    # Adaptive threshold: 0.3x daily vol (min 0.3%)
    recent_ret_pct = df['Close'].pct_change().tail(20).std() * 100
    sig_threshold  = max(0.3 * recent_ret_pct, 0.3)

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
    stop_loss_distance   = 1.0 * atr
    take_profit_distance = 1.5 * atr
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
        model=final_model,
        scaler=scaler,
        df=df_lagged,  # Use the LAGGED dataframe with engineered features
        feature_cols=all_features,  # Use ALL engineered features
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
        df=df,  # Use ORIGINAL df for pattern matching
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
    model_filename = 'randomforest_model.pkl'
    scaler_filename = 'randomforest_scaler.pkl'
    features_filename = 'randomforest_features.txt'
    info_filename = 'randomforest_model_info.txt'

    joblib.dump(final_model, model_filename)
    joblib.dump(scaler, scaler_filename)

    with open(features_filename, 'w') as f:
        f.write('\n'.join(all_features))

    with open(info_filename, 'w') as f:
        f.write(f"Random Forest Model Information\n")
        f.write(f"="*50 + "\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Source: {csv_file}\n")
        f.write(f"Validation Method: Walk-Forward (5 folds)\n")
        f.write(f"\nHyperparameters:\n")
        f.write(f"  n_estimators: {n_estimators}\n")
        f.write(f"  max_depth: {max_depth}\n")
        f.write(f"  max_features: sqrt\n")
        f.write(f"  Number of features: {len(all_features)}\n")
        f.write(f"\nPerformance:\n")
        f.write(f"  Test MAE: ${test_mae:.2f}\n")
        f.write(f"  Test RMSE: ${test_rmse:.2f}\n")
        f.write(f"  Test Accuracy: {test_acc*100:.2f}%\n")

    print(f"\nModel saved as: {model_filename}")
    print(f"Scaler saved as: {scaler_filename}")
    print(f"Features saved as: {features_filename}")
    print(f"Model info saved as: {info_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Random Forest model with walk-forward validation')
    parser.add_argument('csv_file', type=str, help='Path to CSV file with stock data')
    parser.add_argument('--n_estimators', type=int, default=500, help='Number of trees (default: 500)')
    parser.add_argument('--max_depth', type=int, default=15, help='Maximum tree depth (default: 15)')
    parser.add_argument('--max_features', type=int, default=30, help='Max features to use (default: 30)')

    args = parser.parse_args()

    train_randomforest_model(args.csv_file, args.n_estimators, args.max_depth, args.max_features)
