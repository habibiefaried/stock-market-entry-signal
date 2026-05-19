import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime

def load_and_prepare_data(csv_file):
    """Load CSV data and prepare for LSTM"""
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    # Drop rows with NaN values (technical indicators need warming up period)
    df = df.dropna()

    print(f"Total records after removing NaN: {len(df)}")

    # Select features for training (exclude Date and target)
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'MA_5', 'MA_10', 'MA_20', 'MA_50',
                    'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
                    'BB_Upper', 'BB_Middle', 'BB_Lower', 'Volume_MA_20']

    # Keep only columns that exist in the dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]

    return df, feature_cols

def create_sequences(data, target, lookback=60):
    """Create sequences for LSTM input"""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(target[i])
    return np.array(X), np.array(y)

def split_train_test(df, feature_cols, train_ratio=5/6):
    """Split data into train and test sets (5 months train, 1 month test)"""
    split_idx = int(len(df) * train_ratio)

    train_df = df[:split_idx]
    test_df = df[split_idx:]

    print(f"\nData split:")
    print(f"Training set: {len(train_df)} records ({train_df['Date'].min()} to {train_df['Date'].max()})")
    print(f"Test set: {len(test_df)} records ({test_df['Date'].min()} to {test_df['Date'].max()})")

    return train_df, test_df

def build_lstm_model(input_shape):
    """Build LSTM model architecture"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

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

def train_lstm_model(csv_file, lookback=60, epochs=100, batch_size=32):
    """Main training function for LSTM model"""

    # Load data
    df, feature_cols = load_and_prepare_data(csv_file)

    # Split train/test
    train_df, test_df = split_train_test(df, feature_cols)

    # Prepare features and target
    X_train_data = train_df[feature_cols].values
    y_train_data = train_df['Close'].values

    X_test_data = test_df[feature_cols].values
    y_test_data = test_df['Close'].values

    # Scale features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train_data)
    y_train_scaled = scaler_y.fit_transform(y_train_data.reshape(-1, 1))

    X_test_scaled = scaler_X.transform(X_test_data)
    y_test_scaled = scaler_y.transform(y_test_data.reshape(-1, 1))

    # Create sequences
    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, lookback)
    X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, lookback)

    print(f"\nSequence shape:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: Not enough data to create sequences with the given lookback period.")
        print(f"Try reducing lookback period (current: {lookback})")
        return

    # Build model
    print("\nBuilding LSTM model...")
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    print(model.summary())

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_lstm_model.keras', monitor='val_loss', save_best_only=True)

    # Train model
    print("\nTraining LSTM model...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop, model_checkpoint],
        verbose=1
    )

    # Load best model
    model = keras.models.load_model('best_lstm_model.keras')

    # Make predictions
    print("\nMaking predictions...")
    y_train_pred_scaled = model.predict(X_train)
    y_test_pred_scaled = model.predict(X_test)

    # Inverse transform predictions
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
    y_train_actual = scaler_y.inverse_transform(y_train)
    y_test_actual = scaler_y.inverse_transform(y_test)

    # Calculate regression metrics
    train_mae = mean_absolute_error(y_train_actual, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
    test_mae = mean_absolute_error(y_test_actual, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))

    # Calculate direction metrics (classification)
    train_acc, train_prec, train_rec, train_f1 = calculate_direction_metrics(
        y_train_actual.flatten(), y_train_pred.flatten()
    )
    test_acc, test_prec, test_rec, test_f1 = calculate_direction_metrics(
        y_test_actual.flatten(), y_test_pred.flatten()
    )

    # Print results
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

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

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

    # Plot predictions
    plt.figure(figsize=(15, 6))

    # Test set predictions
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

    # Save model info
    ticker = os.path.basename(csv_file).split('_')[0]
    model_info = {
        'ticker': ticker,
        'model_type': 'LSTM',
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

    # Save model info to file
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
