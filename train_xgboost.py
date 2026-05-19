import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import matplotlib.pyplot as plt
import argparse
import os
import joblib
from datetime import datetime

def load_and_prepare_data(csv_file):
    """Load CSV data and prepare for XGBoost"""
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    # Drop rows with NaN values
    df = df.dropna()

    print(f"Total records after removing NaN: {len(df)}")

    # Select features for training
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'MA_5', 'MA_10', 'MA_20', 'MA_50',
                    'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
                    'BB_Upper', 'BB_Middle', 'BB_Lower', 'Volume_MA_20']

    # Keep only columns that exist
    feature_cols = [col for col in feature_cols if col in df.columns]

    return df, feature_cols

def create_lag_features(df, feature_cols, lags=[1, 2, 3, 5, 10]):
    """Create lag features for time series"""
    df_lagged = df.copy()

    # Create lag features for Close price
    for lag in lags:
        df_lagged[f'Close_lag_{lag}'] = df_lagged['Close'].shift(lag)

    # Create lag features for Volume
    for lag in lags:
        df_lagged[f'Volume_lag_{lag}'] = df_lagged['Volume'].shift(lag)

    # Price change features
    df_lagged['Price_change_1d'] = df_lagged['Close'].pct_change(1)
    df_lagged['Price_change_5d'] = df_lagged['Close'].pct_change(5)
    df_lagged['Price_change_10d'] = df_lagged['Close'].pct_change(10)

    # Volatility features
    df_lagged['Volatility_5d'] = df_lagged['Close'].rolling(window=5).std()
    df_lagged['Volatility_10d'] = df_lagged['Close'].rolling(window=10).std()

    # Target: Next day's closing price
    df_lagged['Target'] = df_lagged['Close'].shift(-1)

    # Drop rows with NaN (from shifting)
    df_lagged = df_lagged.dropna()

    # Get all feature column names (excluding Date and Target)
    all_features = [col for col in df_lagged.columns if col not in ['Date', 'Target']]

    return df_lagged, all_features

def split_train_test(df, train_ratio=5/6):
    """Split data into train and test sets (5 months train, 1 month test)"""
    split_idx = int(len(df) * train_ratio)

    train_df = df[:split_idx]
    test_df = df[split_idx:]

    print(f"\nData split:")
    print(f"Training set: {len(train_df)} records ({train_df['Date'].min()} to {train_df['Date'].max()})")
    print(f"Test set: {len(test_df)} records ({test_df['Date'].min()} to {test_df['Date'].max()})")

    return train_df, test_df

def calculate_direction_metrics(y_true, y_pred):
    """Calculate accuracy, precision, recall for price direction (up/down)"""
    # Convert price predictions to direction (1 = up, 0 = down)
    y_true_direction = (np.diff(y_true) > 0).astype(int)
    y_pred_direction = (np.diff(y_pred) > 0).astype(int)

    accuracy = accuracy_score(y_true_direction, y_pred_direction)
    precision = precision_score(y_true_direction, y_pred_direction, zero_division=0)
    recall = recall_score(y_true_direction, y_pred_direction, zero_division=0)
    f1 = f1_score(y_true_direction, y_pred_direction, zero_division=0)

    return accuracy, precision, recall, f1

def train_xgboost_model(csv_file, n_estimators=1000, learning_rate=0.01, max_depth=7):
    """Main training function for XGBoost model"""

    # Load data
    df, feature_cols = load_and_prepare_data(csv_file)

    # Create lag features
    print("\nCreating lag features...")
    df_lagged, all_features = create_lag_features(df, feature_cols)

    print(f"Total features: {len(all_features)}")

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

    # Scale features (optional for tree-based models, but can help)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build XGBoost model
    print("\nBuilding XGBoost model...")

    # Try GPU first, fallback to CPU if not available
    try:
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',
            device='cuda'
        )
        print("Using GPU acceleration (CUDA)")
    except Exception as e:
        print(f"GPU not available, using CPU: {e}")
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist'
        )

    # Train model with early stopping
    print("\nTraining XGBoost model...")
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
        verbose=50
    )

    # Make predictions
    print("\nMaking predictions...")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Calculate regression metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Calculate direction metrics (classification)
    train_acc, train_prec, train_rec, train_f1 = calculate_direction_metrics(y_train, y_train_pred)
    test_acc, test_prec, test_rec, test_f1 = calculate_direction_metrics(y_test, y_test_pred)

    # Print results
    print("\n" + "="*60)
    print("XGBOOST MODEL EVALUATION RESULTS")
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
    plt.title('Top 20 Feature Importances (XGBoost)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png')
    print("\nFeature importance plot saved as: xgboost_feature_importance.png")

    # Plot predictions
    plt.figure(figsize=(15, 6))
    test_dates = test_df['Date'].values
    plt.plot(test_dates, y_test, label='Actual Price', color='blue', linewidth=2)
    plt.plot(test_dates, y_test_pred, label='Predicted Price', color='red', linewidth=2, alpha=0.7)
    plt.title('XGBoost Model: Actual vs Predicted Prices (Test Set)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('xgboost_predictions.png')
    print("Predictions plot saved as: xgboost_predictions.png")

    # Save model
    joblib.dump(model, 'xgboost_model.pkl')
    joblib.dump(scaler, 'xgboost_scaler.pkl')

    # Save feature names
    with open('xgboost_features.txt', 'w') as f:
        f.write('\n'.join(all_features))

    # Save model info
    ticker = os.path.basename(csv_file).split('_')[0]
    model_info = {
        'ticker': ticker,
        'model_type': 'XGBoost',
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
    with open('xgboost_model_info.txt', 'w') as f:
        for key, value in model_info.items():
            f.write(f"{key}: {value}\n")

    print("\nModel saved as: xgboost_model.pkl")
    print("Scaler saved as: xgboost_scaler.pkl")
    print("Features saved as: xgboost_features.txt")
    print("Model info saved as: xgboost_model_info.txt")

    return model, scaler, model_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train XGBoost model for stock price prediction')
    parser.add_argument('csv_file', type=str, help='Path to CSV file with stock data')
    parser.add_argument('--n_estimators', type=int, default=1000, help='Number of trees (default: 1000)')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--max_depth', type=int, default=7, help='Max tree depth (default: 7)')

    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: File {args.csv_file} not found!")
        exit(1)

    train_xgboost_model(args.csv_file, args.n_estimators, args.learning_rate, args.max_depth)
