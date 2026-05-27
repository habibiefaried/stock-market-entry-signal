"""
CNN-1D + LSTM Stock Price Prediction Model

Architecture:
  1. Technical Indicator Engine  - 50+ signals (RSI, MACD, MA, BB, ATR, Stoch, OBV, CCI, etc.)
  2. CNN-1D Feature Extractor    - learns local patterns across indicator channels
  3. LSTM Sequence Learner       - learns temporal dependencies in extracted features
  4. Dense Head                  - outputs tomorrow's closing price

Why CNN before LSTM?
  - CNN: detects short-range cross-indicator patterns (e.g. RSI divergence + BB squeeze)
  - LSTM: remembers how those patterns evolve over weeks/months
  - Together they separate "what is the pattern NOW" from "how has it been trending"
"""

import os
os.environ['KERAS_BACKEND'] = 'torch'

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             accuracy_score, precision_score, recall_score, f1_score)
import keras
from keras.models import Model
from keras.layers import (Input, Conv1D, MaxPooling1D, LSTM, Dense,
                          Dropout, BatchNormalization, Flatten, Concatenate, Layer, Activation)
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import torch
from datetime import datetime
import logging

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('keras').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)

from trade_probability_analyzer import (
    predict_multi_day_path,
    monte_carlo_simulation,
    find_similar_patterns,
    calculate_ensemble_probability,
    format_analysis_report
)

print(f"Keras backend: {keras.backend.backend()}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

_device = "GPU" if torch.cuda.is_available() else "CPU"


# ============================================================================
# TECHNICAL INDICATOR COMPUTATION
# ============================================================================

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_atr(df, period):
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()


def compute_stochastic(df, k_period=14, d_period=3):
    low_min = df['Low'].rolling(k_period).min()
    high_max = df['High'].rolling(k_period).max()
    k = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(d_period).mean()
    return k, d


def compute_cci(df, period):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * mad + 1e-10)


def compute_obv(df):
    direction = np.sign(df['Close'].diff()).fillna(0)
    return (direction * df['Volume']).cumsum()


def compute_williams_r(df, period=14):
    high_max = df['High'].rolling(period).max()
    low_min = df['Low'].rolling(period).min()
    return -100 * (high_max - df['Close']) / (high_max - low_min + 1e-10)


def compute_technical_indicators(df):
    """
    Compute 50+ technical indicators and append them to df.

    Indicator groups:
      - Moving Averages   : SMA 5/10/20/50/100/200, EMA 9/21/50/100
      - MA Ratios         : Close / SMAx normalised
      - RSI               : periods 7, 14, 21
      - MACD              : fast 12 / slow 26 / signal 9
      - Bollinger Bands   : 20-period, +/-2 std  (+%B and bandwidth)
      - ATR               : periods 7, 14
      - Stochastic        : %K(14,3), %D
      - OBV               : on-balance volume
      - CCI               : periods 14, 20
      - Williams %R       : period 14
      - Rate of Change    : 1, 5, 10 days
      - Momentum          : 5, 10 days
      - Volume features   : log-volume, volume/SMA20-ratio
      - Price changes     : 1d, 3d, 5d % returns
      - Volatility        : rolling std 5d, 10d, 20d
      - High/Low range    : daily range %, range vs ATR
      - Price vs MA flags : normalised distance to SMA20 / SMA50 / SMA200
    """
    out = df.copy()
    c = out['Close']

    # --- Moving averages ---
    for p in [5, 10, 20, 50, 100, 200]:
        out[f'SMA_{p}'] = c.rolling(p).mean()
    for p in [9, 21, 50, 100]:
        out[f'EMA_{p}'] = c.ewm(span=p, min_periods=p).mean()

    # --- Price-to-MA ratios ---
    for p in [20, 50, 200]:
        ma = out[f'SMA_{p}']
        out[f'Close_SMA{p}_ratio'] = (c - ma) / (ma + 1e-10)

    # --- RSI ---
    for p in [7, 14, 21]:
        out[f'RSI_{p}'] = compute_rsi(c, p)

    # --- MACD ---
    ema12 = c.ewm(span=12, min_periods=12).mean()
    ema26 = c.ewm(span=26, min_periods=26).mean()
    out['MACD_line']   = ema12 - ema26
    out['MACD_signal'] = out['MACD_line'].ewm(span=9, min_periods=9).mean()
    out['MACD_hist']   = out['MACD_line'] - out['MACD_signal']

    # --- Bollinger Bands (20-period, 2 std) ---
    bb_mid   = c.rolling(20).mean()
    bb_std   = c.rolling(20).std()
    out['BB_upper']  = bb_mid + 2 * bb_std
    out['BB_lower']  = bb_mid - 2 * bb_std
    out['BB_mid']    = bb_mid
    out['BB_pct']    = (c - out['BB_lower']) / (out['BB_upper'] - out['BB_lower'] + 1e-10)
    out['BB_width']  = (out['BB_upper'] - out['BB_lower']) / (bb_mid + 1e-10)

    # --- ATR ---
    for p in [7, 14]:
        out[f'ATR_{p}'] = compute_atr(out, p)

    # --- Stochastic ---
    out['STOCH_K'], out['STOCH_D'] = compute_stochastic(out, 14, 3)

    # --- OBV (log-scaled) ---
    out['OBV'] = np.log1p(compute_obv(out).abs()) * np.sign(compute_obv(out))

    # --- CCI ---
    for p in [14, 20]:
        out[f'CCI_{p}'] = compute_cci(out, p)

    # --- Williams %R ---
    out['WILLR_14'] = compute_williams_r(out, 14)

    # --- Rate of Change ---
    for p in [1, 5, 10]:
        out[f'ROC_{p}'] = c.pct_change(p) * 100

    # --- Momentum ---
    for p in [5, 10]:
        out[f'MOM_{p}'] = c - c.shift(p)

    # --- Volume features ---
    vol = out['Volume']
    out['Volume_log']      = np.log1p(vol)
    out['Volume_MA20_ratio'] = vol / (vol.rolling(20).mean() + 1e-10)

    # --- Price changes ---
    out['Price_change_1d'] = c.pct_change(1) * 100
    out['Price_change_3d'] = c.pct_change(3) * 100
    out['Price_change_5d'] = c.pct_change(5) * 100

    # --- Volatility ---
    ret = c.pct_change()
    for p in [5, 10, 20]:
        out[f'Volatility_{p}d'] = ret.rolling(p).std() * 100

    # --- Daily high/low range ---
    out['HL_range_pct'] = (out['High'] - out['Low']) / (c + 1e-10) * 100
    out['HL_vs_ATR14']  = (out['High'] - out['Low']) / (out['ATR_14'] + 1e-10)

    return out


FEATURE_COLS = [
    # Raw price & volume
    'Open', 'High', 'Low', 'Close', 'Volume',
    # Moving averages
    'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
    'EMA_9', 'EMA_21', 'EMA_50', 'EMA_100',
    # MA ratios
    'Close_SMA20_ratio', 'Close_SMA50_ratio', 'Close_SMA200_ratio',
    # RSI
    'RSI_7', 'RSI_14', 'RSI_21',
    # MACD
    'MACD_line', 'MACD_signal', 'MACD_hist',
    # Bollinger
    'BB_upper', 'BB_lower', 'BB_mid', 'BB_pct', 'BB_width',
    # ATR
    'ATR_7', 'ATR_14',
    # Stochastic
    'STOCH_K', 'STOCH_D',
    # OBV
    'OBV',
    # CCI
    'CCI_14', 'CCI_20',
    # Williams %R
    'WILLR_14',
    # ROC
    'ROC_1', 'ROC_5', 'ROC_10',
    # Momentum
    'MOM_5', 'MOM_10',
    # Volume
    'Volume_log', 'Volume_MA20_ratio',
    # Price changes
    'Price_change_1d', 'Price_change_3d', 'Price_change_5d',
    # Volatility
    'Volatility_5d', 'Volatility_10d', 'Volatility_20d',
    # Range
    'HL_range_pct', 'HL_vs_ATR14',
]


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

def load_and_prepare_data(csv_file):
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("Computing technical indicators...")
    df = compute_technical_indicators(df)
    df = df.dropna().reset_index(drop=True)

    print(f"Records after indicator computation: {len(df)}")
    print(f"Feature count: {len(FEATURE_COLS)}")

    return df, FEATURE_COLS


def split_train_test(df, train_ratio=9/10):
    split_idx = int(len(df) * train_ratio)
    train_df = df[:split_idx]
    test_df  = df[split_idx:]
    print(f"\nData split:")
    print(f"  Train: {len(train_df)} records  ({train_df['Date'].min()} to {train_df['Date'].max()})")
    print(f"  Test:  {len(test_df)} records  ({test_df['Date'].min()} to {test_df['Date'].max()})")
    return train_df, test_df


def create_sequences(X, y, lookback=60):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


# ============================================================================
# TEMPORAL ATTENTION LAYER
# ============================================================================

class TemporalAttention(Layer):
    """
    Soft self-attention over the time axis of an LSTM output sequence.

    Given LSTM output  H  of shape (batch, timesteps, units), the layer
    learns a scalar importance weight for each timestep, then returns the
    weighted sum - a single context vector of shape (batch, units).

    Why this helps:
      Not every day in the 60-day lookback window matters equally.
      A breakout three weeks ago might be far more predictive than yesterday's
      noise.  Attention lets the model assign high weight to the days that
      carry the most signal, ignoring the rest.

    Maths:
      e_t = tanh(W . h_t + b)   [score for each timestep]
      a_t = softmax(e_t)         [normalised attention weights]
      c   = sum a_t . h_t          [context vector]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # input_shape = (batch, timesteps, units)
        units = input_shape[-1]
        self.W = self.add_weight(name='attn_W', shape=(units, 1),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='attn_b', shape=(1,),
                                 initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, timesteps, units)
        import keras.ops as ops
        score = ops.tanh(ops.matmul(inputs, self.W) + self.b)   # (batch, timesteps, 1)
        score = ops.squeeze(score, axis=-1)                      # (batch, timesteps)
        weights = ops.softmax(score)                             # (batch, timesteps)
        weights = ops.expand_dims(weights, axis=-1)              # (batch, timesteps, 1)
        context = ops.sum(inputs * weights, axis=1)              # (batch, units)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super().get_config()


# ============================================================================
# CNN-1D + LSTM MODEL
# ============================================================================

def build_cnn_lstm_model(
    input_shape,
    # CNN block 1
    cnn1_filters=64,
    cnn1_kernel=3,
    # CNN block 2  (wider by default - more cross-indicator pattern capacity)
    cnn2_filters=256,
    cnn2_kernel=5,
    # CNN block 3
    cnn3_filters=64,
    cnn3_kernel=3,
    pool_size=2,
    cnn_dropout=0.25,
    # LSTM
    lstm1_units=128,
    lstm2_units=64,
    lstm_dropout=0.3,
    lstm_recurrent_dropout=0.0,
    # Dense head
    dense_units=32,
    dense_dropout=0.2,
):
    """
    CNN-1D + stacked-LSTM + Temporal Attention hybrid.

    CNN stack:
      Block 1: Conv1D(cnn1_filters, cnn1_kernel, relu, same) + BN
      Block 2: Conv1D(cnn2_filters, cnn2_kernel, relu, same) + BN + MaxPool
               cnn2_filters=256 gives broader cross-indicator pattern capacity
      Block 3: Conv1D(cnn3_filters, cnn3_kernel, relu, same) + BN + Dropout

    LSTM stack:
      LSTM(lstm1_units, return_sequences=True)  + Dropout
      LSTM(lstm2_units, return_sequences=True)  + Dropout   <- feeds attention

    Attention:
      TemporalAttention - learns a soft weight per timestep, collapses the
      sequence to a single context vector.  The model decides which days in the
      lookback window matter most for tomorrow's prediction.

    Head:
      Dense(dense_units, relu) + Dropout + Dense(1)
    """
    inp = Input(shape=input_shape)                         # (lookback, n_features)

    # --- CNN block 1 ---
    x = Conv1D(cnn1_filters, cnn1_kernel, activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)

    # --- CNN block 2 (wider filters for richer cross-indicator patterns) ---
    x = Conv1D(cnn2_filters, cnn2_kernel, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=pool_size)(x)

    # --- CNN block 3 ---
    x = Conv1D(cnn3_filters, cnn3_kernel, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(cnn_dropout)(x)

    # --- LSTM stack (both return sequences so attention sees all timesteps) ---
    x = LSTM(lstm1_units, return_sequences=True,
             recurrent_dropout=lstm_recurrent_dropout)(x)
    x = Dropout(lstm_dropout)(x)

    x = LSTM(lstm2_units, return_sequences=True,
             recurrent_dropout=lstm_recurrent_dropout)(x)
    x = Dropout(lstm_dropout)(x)

    # --- Temporal Attention (collapses timesteps -> context vector) ---
    x = TemporalAttention(name='temporal_attention')(x)    # (batch, lstm2_units)

    # --- Dense head ---
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dense_dropout)(x)
    out = Dense(1)(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='huber',
        metrics=['mae'],
    )
    return model


# ============================================================================
# DIRECTION METRICS
# ============================================================================

def calculate_direction_metrics(y_true, y_pred):
    y_true_dir = (np.diff(y_true) > 0).astype(int)
    y_pred_dir = (np.diff(y_pred) > 0).astype(int)
    acc  = accuracy_score(y_true_dir, y_pred_dir)
    prec = precision_score(y_true_dir, y_pred_dir, zero_division=0)
    rec  = recall_score(y_true_dir, y_pred_dir, zero_division=0)
    f1   = f1_score(y_true_dir, y_pred_dir, zero_division=0)
    return acc, prec, rec, f1


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_lstm_model(
    csv_file,
    lookback=60,
    epochs=75,
    batch_size=32,
    # CNN hyperparameters
    cnn1_filters=64,
    cnn1_kernel=3,
    cnn2_filters=256,
    cnn2_kernel=5,
    cnn3_filters=64,
    cnn3_kernel=3,
    pool_size=2,
    cnn_dropout=0.25,
    # LSTM hyperparameters
    lstm1_units=128,
    lstm2_units=64,
    lstm_dropout=0.3,
    # Dense head
    dense_units=32,
    dense_dropout=0.2,
):
    # ---- Load data ----
    df, feature_cols = load_and_prepare_data(csv_file)
    train_df, test_df = split_train_test(df, train_ratio=9/10)

    X_train_raw = train_df[feature_cols].values
    y_train_raw = train_df['Close'].values
    X_test_raw  = test_df[feature_cols].values
    y_test_raw  = test_df['Close'].values

    # ---- Scale ----
    # scaler_X: all indicator features  (MinMax -> [0,1])
    # scaler_y: Close price target only
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1))

    X_test_scaled  = scaler_X.transform(X_test_raw)
    y_test_scaled  = scaler_y.transform(y_test_raw.reshape(-1, 1))

    # ---- Create sequences ----
    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, lookback)
    X_test,  y_test  = create_sequences(X_test_scaled,  y_test_scaled,  lookback)

    print(f"\nSequence shapes:")
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")

    if len(X_train) == 0 or len(X_test) == 0:
        print("ERROR: Not enough data to create sequences. Reduce --lookback or fetch more data.")
        return

    # ---- Build model ----
    print(f"\nBuilding CNN-1D + LSTM model  (input: {X_train.shape[1:]})")
    model = build_cnn_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        cnn1_filters=cnn1_filters, cnn1_kernel=cnn1_kernel,
        cnn2_filters=cnn2_filters, cnn2_kernel=cnn2_kernel,
        cnn3_filters=cnn3_filters, cnn3_kernel=cnn3_kernel,
        pool_size=pool_size,       cnn_dropout=cnn_dropout,
        lstm1_units=lstm1_units,   lstm2_units=lstm2_units,
        lstm_dropout=lstm_dropout,
        dense_units=dense_units,   dense_dropout=dense_dropout,
    )
    print(model.summary())

    # ---- Callbacks ----
    early_stop = EarlyStopping(
        monitor='val_loss', patience=10,
        restore_best_weights=True, verbose=0
    )
    checkpoint = ModelCheckpoint(
        'best_lstm_model.keras', monitor='val_loss',
        save_best_only=True, verbose=0
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=5, min_lr=1e-6, verbose=0
    )

    # ---- Train ----
    print(f"\nTraining CNN-LSTM on {_device}...")
    print("This may take 1-3 minutes with GPU, 5-10 minutes with CPU")

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=1,
    )

    model = keras.models.load_model(
        'best_lstm_model.keras',
        custom_objects={'TemporalAttention': TemporalAttention}
    )

    # ---- Predict ----
    print("\nMaking predictions...")
    y_train_pred_scaled = model.predict(X_train, verbose=0)
    y_test_pred_scaled  = model.predict(X_test,  verbose=0)

    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled).flatten()
    y_test_pred  = scaler_y.inverse_transform(y_test_pred_scaled).flatten()
    y_train_act  = scaler_y.inverse_transform(y_train).flatten()
    y_test_act   = scaler_y.inverse_transform(y_test).flatten()

    # ---- Metrics ----
    train_mae  = mean_absolute_error(y_train_act, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_act, y_train_pred))
    test_mae   = mean_absolute_error(y_test_act,  y_test_pred)
    test_rmse  = np.sqrt(mean_squared_error(y_test_act,  y_test_pred))

    train_acc, train_prec, train_rec, train_f1 = calculate_direction_metrics(y_train_act, y_train_pred)
    test_acc,  test_prec,  test_rec,  test_f1  = calculate_direction_metrics(y_test_act,  y_test_pred)

    print("\n" + "="*60)
    print("CNN-LSTM MODEL EVALUATION RESULTS")
    print("="*60)
    print("\nREGRESSION METRICS (Price Prediction):")
    print(f"  Training MAE:   ${train_mae:.2f}")
    print(f"  Training RMSE:  ${train_rmse:.2f}")
    print(f"  Test MAE:       ${test_mae:.2f}")
    print(f"  Test RMSE:      ${test_rmse:.2f}")
    print("\nCLASSIFICATION METRICS (Direction Prediction Up/Down):")
    print(f"  Training Accuracy:  {train_acc*100:.2f}%")
    print(f"  Training Precision: {train_prec*100:.2f}%")
    print(f"  Training Recall:    {train_rec*100:.2f}%")
    print(f"  Training F1-Score:  {train_f1*100:.2f}%")
    print(f"\n  Test Accuracy:      {test_acc*100:.2f}%")
    print(f"  Test Precision:     {test_prec*100:.2f}%")
    print(f"  Test Recall:        {test_rec*100:.2f}%")
    print(f"  Test F1-Score:      {test_f1*100:.2f}%")
    print("\n" + "="*60)

    # ---- Plots ----
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'],     label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'],     label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Model MAE'); plt.xlabel('Epoch'); plt.ylabel('MAE')
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig('lstm_training_history.png')
    print("\nTraining history plot saved as: lstm_training_history.png")

    plt.figure(figsize=(15, 6))
    test_dates = test_df['Date'].values[lookback:]
    plt.plot(test_dates, y_test_act,  label='Actual',    color='blue', linewidth=2)
    plt.plot(test_dates, y_test_pred, label='Predicted', color='red',  linewidth=2, alpha=0.7)
    plt.title('CNN-LSTM: Actual vs Predicted Prices (Test Set)')
    plt.xlabel('Date'); plt.ylabel('Price')
    plt.legend(); plt.xticks(rotation=45); plt.grid(True); plt.tight_layout()
    plt.savefig('lstm_predictions.png')
    print("Predictions plot saved as: lstm_predictions.png")

    # ---- Trading signal ----
    print("\n" + "="*60)
    print("TRADING SIGNAL FOR NEXT DAY")
    print("="*60)

    today_price = df['Close'].iloc[-1]

    recent_data   = df[feature_cols].tail(lookback).values
    recent_scaled = scaler_X.transform(recent_data)
    X_input       = recent_scaled.reshape(1, lookback, len(feature_cols))

    tomorrow_pred_scaled = model.predict(X_input, verbose=0)
    tomorrow_pred        = scaler_y.inverse_transform(tomorrow_pred_scaled)[0][0]

    expected_move     = tomorrow_pred - today_price
    expected_move_pct = (expected_move / today_price) * 100

    # Adaptive threshold: 0.3x daily vol (min 0.3%)
    recent_ret_pct = df['Close'].pct_change().tail(20).std() * 100
    sig_threshold  = max(0.5 * recent_ret_pct, 0.5)

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
    h, l, c_s = df['High'], df['Low'], df['Close']
    tr  = pd.concat([h - l, (h - c_s.shift()).abs(), (l - c_s.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(span=14, min_periods=14).mean().iloc[-1])
    stop_loss_distance   = 1.5 * atr
    take_profit_distance = 2.0 * atr
    volatility           = df['Close'].tail(20).pct_change().dropna().std() * today_price

    if signal == "BUY (LONG)":
        stop_loss   = today_price - stop_loss_distance
        take_profit = today_price + take_profit_distance
    elif signal == "SHORT (SELL)":
        stop_loss   = today_price + stop_loss_distance
        take_profit = today_price - take_profit_distance
    else:
        stop_loss   = today_price - stop_loss_distance
        take_profit = today_price + take_profit_distance

    confidence = test_acc * 100

    print(f"\n{signal_emoji} SIGNAL: {signal}")
    print(f"\nCurrent Price (Today):      ${today_price:.2f}")
    print(f"Predicted Price (Tomorrow): ${tomorrow_pred:.2f}")
    print(f"Expected Move:              ${expected_move:+.2f} ({expected_move_pct:+.2f}%)")
    print(f"\nRisk Management (Stock Price Levels):")
    print(f"  Stop Loss:    ${stop_loss:.2f} ({((stop_loss - today_price) / today_price * 100):+.2f}%)")
    print(f"  Take Profit:  ${take_profit:.2f} ({((take_profit - today_price) / today_price * 100):+.2f}%)")
    print(f"\n5x Leverage Position P&L (for IQ Option auto-close):")
    print(f"  Stop Loss %:   {((stop_loss - today_price) / today_price * 100 * 5):+.1f}%")
    print(f"  Take Profit %: {((take_profit - today_price) / today_price * 100 * 5):+.1f}%")
    print(f"  Risk/Reward:   1.67:1")
    print(f"\nModel Confidence: {confidence:.1f}% (based on test accuracy)")
    print(f"Recent Volatility: ${volatility:.2f} per day")

    # ---- Multi-approach probability analysis ----
    print("\n" + "="*70)
    print("Running Multi-Approach Win Probability Analysis...")
    print("="*70)

    print("\n[1/3] Running multi-day sequential prediction...")
    prediction_result = predict_multi_day_path(
        model=model,
        scaler=scaler_X,
        df=df,
        feature_cols=feature_cols,
        current_price=today_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        model_type='lstm',
        target_scaler=scaler_y,
        lookback=lookback,
    )

    print("[2/3] Running Monte Carlo simulation...")
    monte_carlo_result = monte_carlo_simulation(
        current_price=today_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        volatility=volatility,
        predicted_move_pct=expected_move_pct,
    )

    print("[3/3] Searching historical patterns...")
    pattern_result = find_similar_patterns(
        df=df,
        current_price=today_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )

    ensemble_result = calculate_ensemble_probability(
        prediction_result=prediction_result,
        monte_carlo_result=monte_carlo_result,
        pattern_result=pattern_result,
    )

    analysis_report = format_analysis_report(
        prediction_result=prediction_result,
        monte_carlo_result=monte_carlo_result,
        pattern_result=pattern_result,
        ensemble_result=ensemble_result,
        signal=signal,
        current_price=today_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )
    print(analysis_report)

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

    # ---- Save model info ----
    ticker = os.path.basename(csv_file).split('_')[0]
    model_info = {
        'ticker':         ticker,
        'model_type':     f'CNN-1D + LSTM (Keras + PyTorch + {_device})',
        'backend':        keras.backend.backend(),
        'gpu_used':       torch.cuda.is_available(),
        'n_features':     len(feature_cols),
        'lookback':       lookback,
        'architecture':   f'Conv1D({cnn1_filters},{cnn1_kernel}) -> Conv1D({cnn2_filters},{cnn2_kernel}) -> Conv1D({cnn3_filters},{cnn3_kernel}) -> LSTM({lstm1_units}) -> LSTM({lstm2_units}) -> Dense({dense_units})',
        'train_size':     len(X_train),
        'test_size':      len(X_test),
        'test_mae':       test_mae,
        'test_rmse':      test_rmse,
        'test_accuracy':  test_acc,
        'test_precision': test_prec,
        'test_recall':    test_rec,
        'test_f1':        test_f1,
        'timestamp':      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open('lstm_model_info.txt', 'w') as f:
        for key, value in model_info.items():
            f.write(f"{key}: {value}\n")

    print("\nModel saved as: best_lstm_model.keras")
    print("Model info saved as: lstm_model_info.txt")

    # Write signal file for RL agent to read
    sig_code = 1 if signal == "BUY (LONG)" else (-1 if signal == "SHORT (SELL)" else 0)
    with open('lstm_signal.txt', 'w') as f:
        f.write(f"signal: {sig_code}\n")
        f.write(f"prob: {min(0.5 + abs(expected_move_pct) / 10, 0.95):.4f}\n")
        f.write(f"ensemble_prob: {ensemble_result['ensemble_probability'] if ensemble_result else 50.0:.1f}\n")
    print("Signal file saved as: lstm_signal.txt")

    return model, history, model_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train CNN-1D + LSTM model for stock price prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python train_lstm.py MSFT_daily_data_20260520.csv
  python train_lstm.py MSFT_daily_data_20260520.csv --lookback 90 --epochs 200
  python train_lstm.py MSFT_daily_data_20260520.csv \\
      --cnn1_filters 128 --cnn2_filters 256 --lstm1_units 256 --lstm2_units 128

Architecture parameters:
  CNN block 1  : --cnn1_filters  --cnn1_kernel
  CNN block 2  : --cnn2_filters  --cnn2_kernel  --pool_size
  CNN block 3  : --cnn3_filters  --cnn3_kernel  --cnn_dropout
  LSTM         : --lstm1_units   --lstm2_units   --lstm_dropout
  Dense head   : --dense_units   --dense_dropout
        '''
    )
    parser.add_argument('csv_file', type=str)
    parser.add_argument('--lookback',      type=int,   default=60)
    parser.add_argument('--epochs',        type=int,   default=75)
    parser.add_argument('--batch_size',    type=int,   default=32)
    # CNN
    parser.add_argument('--cnn1_filters',  type=int,   default=64)
    parser.add_argument('--cnn1_kernel',   type=int,   default=3)
    parser.add_argument('--cnn2_filters',  type=int,   default=256)
    parser.add_argument('--cnn2_kernel',   type=int,   default=5)
    parser.add_argument('--cnn3_filters',  type=int,   default=64)
    parser.add_argument('--cnn3_kernel',   type=int,   default=3)
    parser.add_argument('--pool_size',     type=int,   default=2)
    parser.add_argument('--cnn_dropout',   type=float, default=0.25)
    # LSTM
    parser.add_argument('--lstm1_units',   type=int,   default=128)
    parser.add_argument('--lstm2_units',   type=int,   default=64)
    parser.add_argument('--lstm_dropout',  type=float, default=0.3)
    # Dense
    parser.add_argument('--dense_units',   type=int,   default=32)
    parser.add_argument('--dense_dropout', type=float, default=0.2)

    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: File {args.csv_file} not found!")
        exit(1)

    train_lstm_model(
        args.csv_file,
        lookback=args.lookback,
        epochs=args.epochs,
        batch_size=args.batch_size,
        cnn1_filters=args.cnn1_filters,
        cnn1_kernel=args.cnn1_kernel,
        cnn2_filters=args.cnn2_filters,
        cnn2_kernel=args.cnn2_kernel,
        cnn3_filters=args.cnn3_filters,
        cnn3_kernel=args.cnn3_kernel,
        pool_size=args.pool_size,
        cnn_dropout=args.cnn_dropout,
        lstm1_units=args.lstm1_units,
        lstm2_units=args.lstm2_units,
        lstm_dropout=args.lstm_dropout,
        dense_units=args.dense_units,
        dense_dropout=args.dense_dropout,
    )
