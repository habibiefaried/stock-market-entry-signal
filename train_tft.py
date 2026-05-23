"""
CNN-1D + TFT (Temporal Fusion Transformer) Stock Price Prediction Model

Architecture:
  1. Technical Indicator Engine  — same 50+ signals as CNN-LSTM
  2. CNN-1D Feature Extractor    — learns local cross-indicator patterns
  3. Gated Residual Network      — non-linear feature selection per timestep
  4. Variable Selection Network  — learns which indicators matter most
  5. LSTM Encoder                — builds local context sequence
  6. Multi-Head Self-Attention   — captures long-range temporal dependencies
  7. Gated skip connections      — let irrelevant paths be suppressed
  8. Dense Head                  — outputs tomorrow's closing price

Why TFT after CNN?
  - CNN: detects short-range cross-indicator patterns (BB squeeze + RSI divergence)
  - GRN/VSN: gates out noisy indicators, keeps only predictive ones
  - LSTM encoder: builds a local temporal context
  - Multi-head attention: finds non-adjacent similar patterns across the full lookback
  - Gated residuals throughout: model learns how much of each sub-network to trust

Reference: Lim et al. 2021, "Temporal Fusion Transformers for Interpretable
           Multi-horizon Time Series Forecasting" (simplified single-step version)
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
from keras.layers import (Input, Conv1D, MaxPooling1D, Dense, Dropout,
                          BatchNormalization, LSTM, Layer, MultiHeadAttention,
                          LayerNormalization, GlobalAveragePooling1D, Multiply, Add)
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
# TECHNICAL INDICATORS  (identical to train_lstm.py)
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
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()


def compute_stochastic(df, k_period=14, d_period=3):
    low_min  = df['Low'].rolling(k_period).min()
    high_max = df['High'].rolling(k_period).max()
    k = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(d_period).mean()
    return k, d


def compute_cci(df, period):
    tp  = (df['High'] + df['Low'] + df['Close']) / 3
    ma  = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * mad + 1e-10)


def compute_obv(df):
    direction = np.sign(df['Close'].diff()).fillna(0)
    return (direction * df['Volume']).cumsum()


def compute_williams_r(df, period=14):
    high_max = df['High'].rolling(period).max()
    low_min  = df['Low'].rolling(period).min()
    return -100 * (high_max - df['Close']) / (high_max - low_min + 1e-10)


def compute_technical_indicators(df):
    out = df.copy()
    c   = out['Close']

    for p in [5, 10, 20, 50, 100, 200]:
        out[f'SMA_{p}'] = c.rolling(p).mean()
    for p in [9, 21, 50, 100]:
        out[f'EMA_{p}'] = c.ewm(span=p, min_periods=p).mean()

    for p in [20, 50, 200]:
        ma = out[f'SMA_{p}']
        out[f'Close_SMA{p}_ratio'] = (c - ma) / (ma + 1e-10)

    for p in [7, 14, 21]:
        out[f'RSI_{p}'] = compute_rsi(c, p)

    ema12 = c.ewm(span=12, min_periods=12).mean()
    ema26 = c.ewm(span=26, min_periods=26).mean()
    out['MACD_line']   = ema12 - ema26
    out['MACD_signal'] = out['MACD_line'].ewm(span=9, min_periods=9).mean()
    out['MACD_hist']   = out['MACD_line'] - out['MACD_signal']

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    out['BB_upper'] = bb_mid + 2 * bb_std
    out['BB_lower'] = bb_mid - 2 * bb_std
    out['BB_mid']   = bb_mid
    out['BB_pct']   = (c - out['BB_lower']) / (out['BB_upper'] - out['BB_lower'] + 1e-10)
    out['BB_width'] = (out['BB_upper'] - out['BB_lower']) / (bb_mid + 1e-10)

    for p in [7, 14]:
        out[f'ATR_{p}'] = compute_atr(out, p)

    out['STOCH_K'], out['STOCH_D'] = compute_stochastic(out, 14, 3)

    obv_raw     = compute_obv(out)
    out['OBV']  = np.log1p(obv_raw.abs()) * np.sign(obv_raw)

    for p in [14, 20]:
        out[f'CCI_{p}'] = compute_cci(out, p)

    out['WILLR_14'] = compute_williams_r(out, 14)

    for p in [1, 5, 10]:
        out[f'ROC_{p}'] = c.pct_change(p) * 100

    for p in [5, 10]:
        out[f'MOM_{p}'] = c - c.shift(p)

    vol = out['Volume']
    out['Volume_log']        = np.log1p(vol)
    out['Volume_MA20_ratio'] = vol / (vol.rolling(20).mean() + 1e-10)

    out['Price_change_1d'] = c.pct_change(1) * 100
    out['Price_change_3d'] = c.pct_change(3) * 100
    out['Price_change_5d'] = c.pct_change(5) * 100

    ret = c.pct_change()
    for p in [5, 10, 20]:
        out[f'Volatility_{p}d'] = ret.rolling(p).std() * 100

    out['HL_range_pct'] = (out['High'] - out['Low']) / (c + 1e-10) * 100
    out['HL_vs_ATR14']  = (out['High'] - out['Low']) / (out['ATR_14'] + 1e-10)

    return out


FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
    'EMA_9', 'EMA_21', 'EMA_50', 'EMA_100',
    'Close_SMA20_ratio', 'Close_SMA50_ratio', 'Close_SMA200_ratio',
    'RSI_7', 'RSI_14', 'RSI_21',
    'MACD_line', 'MACD_signal', 'MACD_hist',
    'BB_upper', 'BB_lower', 'BB_mid', 'BB_pct', 'BB_width',
    'ATR_7', 'ATR_14',
    'STOCH_K', 'STOCH_D',
    'OBV',
    'CCI_14', 'CCI_20',
    'WILLR_14',
    'ROC_1', 'ROC_5', 'ROC_10',
    'MOM_5', 'MOM_10',
    'Volume_log', 'Volume_MA20_ratio',
    'Price_change_1d', 'Price_change_3d', 'Price_change_5d',
    'Volatility_5d', 'Volatility_10d', 'Volatility_20d',
    'HL_range_pct', 'HL_vs_ATR14',
]


# ============================================================================
# TFT BUILDING BLOCKS
# ============================================================================

class GatedLinearUnit(Layer):
    """
    GLU: splits input in half, one half gates the other via sigmoid.
    Controls how much information flows through — the network learns
    to suppress irrelevant signals completely.

      GLU(x) = x[:, :d] ⊙ sigmoid(x[:, d:])
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = Dense(units * 2)

    def call(self, x):
        projected = self.dense(x)
        x1, x2 = projected[..., :self.units], projected[..., self.units:]
        return x1 * K.sigmoid(x2)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'units': self.units})
        return cfg


class GatedResidualNetwork(Layer):
    """
    GRN: the core non-linear processing block in TFT.

    Applies two dense transformations with ELU activation, a GLU gate,
    dropout, and a residual skip connection with layer norm.

    If the input dimension differs from output units, a linear projection
    aligns them for the skip connection.

    GRN(x) = LayerNorm(x_proj + GLU(ELU(W2 · ELU(W1 · x))))
    """
    def __init__(self, units, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units       = units
        self.dropout_rate = dropout_rate
        self.dense1      = Dense(units, activation='elu')
        self.dense2      = Dense(units)
        self.glu         = GatedLinearUnit(units)
        self.dropout     = Dropout(dropout_rate)
        self.layer_norm  = LayerNormalization()
        self.proj        = None   # built lazily if needed

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.proj = Dense(self.units, use_bias=False)
        super().build(input_shape)

    def call(self, x, training=False):
        residual = self.proj(x) if self.proj else x
        h = self.dense1(x)
        h = self.dense2(h)
        h = self.glu(h)
        h = self.dropout(h, training=training)
        return self.layer_norm(residual + h)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'units': self.units, 'dropout_rate': self.dropout_rate})
        return cfg


class VariableSelectionNetwork(Layer):
    """
    VSN: learns a soft importance weight for each input feature.

    For each of the N input features, applies an individual GRN, then
    combines them via a shared GRN that outputs N softmax weights.
    The final output is the weighted sum of the per-feature GRN outputs.

    This makes the model explicitly learn which indicators (RSI, MACD,
    BB width, …) carry useful information at each timestep — and suppress
    the rest.
    """
    def __init__(self, units, n_features, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units        = units
        self.n_features   = n_features
        self.dropout_rate = dropout_rate
        self.feature_grns = [GatedResidualNetwork(units, dropout_rate)
                             for _ in range(n_features)]
        self.selector_grn = GatedResidualNetwork(n_features, dropout_rate)
        self.softmax      = Dense(n_features, activation='softmax')

    def call(self, x, training=False):
        # x: (batch, timesteps, n_features)  or  (batch, n_features)
        per_feature = K.stack(
            [self.feature_grns[i](x[..., i:i+1], training=training)
             for i in range(self.n_features)],
            axis=-2
        )  # (..., n_features, units)

        # Flatten last two dims to compute selection weights
        flat  = K.reshape(per_feature,
                          (-1,) + (self.n_features * self.units,)
                          if len(K.int_shape(x)) == 2
                          else K.int_shape(x)[:1] + (K.int_shape(x)[1],) + (self.n_features * self.units,))

        # Use the raw input to produce selection weights
        weights = self.softmax(self.selector_grn(x, training=training))  # (..., n_features)
        weights = K.expand_dims(weights, axis=-1)                         # (..., n_features, 1)

        # Weighted combination
        out = K.sum(per_feature * weights, axis=-2)   # (..., units)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'units': self.units, 'n_features': self.n_features,
                    'dropout_rate': self.dropout_rate})
        return cfg


# ============================================================================
# CNN-1D + TFT MODEL
# ============================================================================

def build_cnn_tft_model(
    input_shape,
    # CNN block
    cnn1_filters=64,
    cnn1_kernel=3,
    cnn2_filters=256,
    cnn2_kernel=5,
    cnn3_filters=64,
    cnn3_kernel=3,
    pool_size=2,
    cnn_dropout=0.1,
    # TFT core
    d_model=64,           # hidden size throughout GRN / attention
    n_heads=4,            # multi-head attention heads
    lstm_units=64,        # encoder LSTM size
    grn_dropout=0.1,      # dropout inside GRN blocks
    attn_dropout=0.1,     # dropout on attention weights
    # Dense head
    dense_units=32,
    dense_dropout=0.1,
):
    """
    CNN-1D + TFT hybrid.

    Data flow:
      Input (lookback, n_features)
        → CNN block 1  Conv1D(64,  k=3) + BN
        → CNN block 2  Conv1D(256, k=5) + BN + MaxPool
        → CNN block 3  Conv1D(64,  k=3) + BN + Dropout
        → GRN per timestep          (non-linear feature transform)
        → LSTM encoder              (local temporal context)
        → GRN + skip                (refine encoder output)
        → Multi-Head Self-Attention (long-range dependencies)
        → GRN + skip                (post-attention refinement)
        → GlobalAveragePooling      (collapse timestep axis)
        → Dense(32) + Dropout
        → Dense(1)                  (tomorrow's price)

    VSN is omitted from the per-timestep path here because the CNN already
    performs implicit feature selection; VSN would add O(n_features²)
    parameters on 54 features and overfit on ~1000 rows.
    """
    inp = Input(shape=input_shape)                      # (lookback, n_features)

    # ── CNN feature extraction ──────────────────────────────────────────────
    x = Conv1D(cnn1_filters, cnn1_kernel, activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)

    x = Conv1D(cnn2_filters, cnn2_kernel, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=pool_size)(x)

    x = Conv1D(cnn3_filters, cnn3_kernel, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(cnn_dropout)(x)

    # ── Project CNN output to d_model ───────────────────────────────────────
    x = Dense(d_model)(x)                              # (batch, T', d_model)

    # ── GRN: non-linear per-timestep transform ──────────────────────────────
    grn_in  = GatedResidualNetwork(d_model, grn_dropout, name='grn_pre_lstm')(x)

    # ── LSTM encoder: local temporal context ────────────────────────────────
    lstm_out = LSTM(lstm_units, return_sequences=True, name='lstm_encoder')(grn_in)

    # ── GRN + gated skip after LSTM ─────────────────────────────────────────
    grn_post  = GatedResidualNetwork(d_model, grn_dropout, name='grn_post_lstm')(lstm_out)
    # gate: how much of the LSTM output vs the skip to keep
    gate_lstm = Dense(d_model, activation='sigmoid', name='gate_lstm')(grn_post)
    x         = gate_lstm * grn_post + (1 - gate_lstm) * grn_in
    x         = LayerNormalization(name='ln_after_lstm')(x)

    # ── Multi-Head Self-Attention: long-range temporal dependencies ──────────
    attn_out, attn_weights = MultiHeadAttention(
        num_heads=n_heads,
        key_dim=d_model // n_heads,
        dropout=attn_dropout,
        name='multi_head_attention'
    )(x, x, return_attention_scores=True)

    # ── GRN + gated skip after attention ────────────────────────────────────
    grn_attn  = GatedResidualNetwork(d_model, grn_dropout, name='grn_post_attn')(attn_out)
    gate_attn = Dense(d_model, activation='sigmoid', name='gate_attn')(grn_attn)
    x         = gate_attn * grn_attn + (1 - gate_attn) * x
    x         = LayerNormalization(name='ln_after_attn')(x)

    # ── Collapse timestep axis ───────────────────────────────────────────────
    x = GlobalAveragePooling1D(name='temporal_pool')(x)

    # ── Dense head ───────────────────────────────────────────────────────────
    x   = Dense(dense_units, activation='relu', name='head_dense')(x)
    x   = Dropout(dense_dropout)(x)
    out = Dense(1, name='price_output')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='huber',
        metrics=['mae'],
    )
    return model


# ============================================================================
# DATA HELPERS
# ============================================================================

def load_and_prepare_data(csv_file):
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing  = [c for c in required if c not in df.columns]
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
    train_df  = df[:split_idx]
    test_df   = df[split_idx:]
    print(f"\nData split:")
    print(f"  Train: {len(train_df)} records  ({train_df['Date'].min()} → {train_df['Date'].max()})")
    print(f"  Test:  {len(test_df)} records  ({test_df['Date'].min()} → {test_df['Date'].max()})")
    return train_df, test_df


def create_sequences(X, y, lookback=60):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


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

def train_tft_model(
    csv_file,
    lookback=60,
    epochs=150,
    batch_size=32,
    # CNN
    cnn1_filters=64,
    cnn1_kernel=3,
    cnn2_filters=256,
    cnn2_kernel=5,
    cnn3_filters=64,
    cnn3_kernel=3,
    pool_size=2,
    cnn_dropout=0.1,
    # TFT
    d_model=64,
    n_heads=4,
    lstm_units=64,
    grn_dropout=0.1,
    attn_dropout=0.1,
    # Head
    dense_units=32,
    dense_dropout=0.1,
):
    # ── Load & prepare ───────────────────────────────────────────────────────
    df, feature_cols = load_and_prepare_data(csv_file)
    train_df, test_df = split_train_test(df)

    X_train_raw = train_df[feature_cols].values
    y_train_raw = train_df['Close'].values
    X_test_raw  = test_df[feature_cols].values
    y_test_raw  = test_df['Close'].values

    # ── Scale ────────────────────────────────────────────────────────────────
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1))
    X_test_scaled  = scaler_X.transform(X_test_raw)
    y_test_scaled  = scaler_y.transform(y_test_raw.reshape(-1, 1))

    # ── Sequences ────────────────────────────────────────────────────────────
    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, lookback)
    X_test,  y_test  = create_sequences(X_test_scaled,  y_test_scaled,  lookback)

    print(f"\nSequence shapes:")
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")

    if len(X_train) == 0 or len(X_test) == 0:
        print("ERROR: Not enough data. Reduce --lookback or fetch more data.")
        return

    # ── Build model ──────────────────────────────────────────────────────────
    print(f"\nBuilding CNN-1D + TFT model  (input: {X_train.shape[1:]})")
    model = build_cnn_tft_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        cnn1_filters=cnn1_filters, cnn1_kernel=cnn1_kernel,
        cnn2_filters=cnn2_filters, cnn2_kernel=cnn2_kernel,
        cnn3_filters=cnn3_filters, cnn3_kernel=cnn3_kernel,
        pool_size=pool_size,       cnn_dropout=cnn_dropout,
        d_model=d_model,           n_heads=n_heads,
        lstm_units=lstm_units,     grn_dropout=grn_dropout,
        attn_dropout=attn_dropout,
        dense_units=dense_units,   dense_dropout=dense_dropout,
    )
    print(model.summary())

    # ── Callbacks ────────────────────────────────────────────────────────────
    early_stop = EarlyStopping(
        monitor='val_loss', patience=15,
        restore_best_weights=True, verbose=0
    )
    checkpoint = ModelCheckpoint(
        'best_tft_model.keras', monitor='val_loss',
        save_best_only=True, verbose=0
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=7, min_lr=1e-6, verbose=0
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nTraining CNN-TFT on {_device}...")
    print("This may take 3-8 minutes with GPU, 15-30 minutes with CPU")

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=1,
    )

    model = keras.models.load_model(
        'best_tft_model.keras',
        custom_objects={
            'GatedLinearUnit': GatedLinearUnit,
            'GatedResidualNetwork': GatedResidualNetwork,
            'VariableSelectionNetwork': VariableSelectionNetwork,
        }
    )

    # ── Predict ──────────────────────────────────────────────────────────────
    print("\nMaking predictions...")
    y_train_pred_scaled = model.predict(X_train, verbose=0)
    y_test_pred_scaled  = model.predict(X_test,  verbose=0)

    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled).flatten()
    y_test_pred  = scaler_y.inverse_transform(y_test_pred_scaled).flatten()
    y_train_act  = scaler_y.inverse_transform(y_train).flatten()
    y_test_act   = scaler_y.inverse_transform(y_test).flatten()

    # ── Metrics ──────────────────────────────────────────────────────────────
    train_mae  = mean_absolute_error(y_train_act, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_act, y_train_pred))
    test_mae   = mean_absolute_error(y_test_act,  y_test_pred)
    test_rmse  = np.sqrt(mean_squared_error(y_test_act,  y_test_pred))

    train_acc, train_prec, train_rec, train_f1 = calculate_direction_metrics(y_train_act, y_train_pred)
    test_acc,  test_prec,  test_rec,  test_f1  = calculate_direction_metrics(y_test_act,  y_test_pred)

    print("\n" + "="*60)
    print("CNN-TFT MODEL EVALUATION RESULTS")
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

    # ── Plots ─────────────────────────────────────────────────────────────────
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'],     label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('CNN-TFT Model Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'],     label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('CNN-TFT Model MAE')
    plt.xlabel('Epoch'); plt.ylabel('MAE')
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig('tft_training_history.png')
    print("\nTraining history plot saved as: tft_training_history.png")

    plt.figure(figsize=(15, 6))
    test_dates = test_df['Date'].values[lookback:]
    plt.plot(test_dates, y_test_act,  label='Actual',    color='blue', linewidth=2)
    plt.plot(test_dates, y_test_pred, label='Predicted', color='orange', linewidth=2, alpha=0.8)
    plt.title('CNN-TFT: Actual vs Predicted Prices (Test Set)')
    plt.xlabel('Date'); plt.ylabel('Price')
    plt.legend(); plt.xticks(rotation=45); plt.grid(True); plt.tight_layout()
    plt.savefig('tft_predictions.png')
    print("Predictions plot saved as: tft_predictions.png")

    # ── Trading signal ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TRADING SIGNAL FOR NEXT DAY")
    print("="*60)

    today_price   = df['Close'].iloc[-1]
    recent_data   = df[feature_cols].tail(lookback).values
    recent_scaled = scaler_X.transform(recent_data)
    X_input       = recent_scaled.reshape(1, lookback, len(feature_cols))

    tomorrow_pred_scaled = model.predict(X_input, verbose=0)
    tomorrow_pred        = scaler_y.inverse_transform(tomorrow_pred_scaled)[0][0]

    expected_move     = tomorrow_pred - today_price
    expected_move_pct = (expected_move / today_price) * 100

    if expected_move_pct > 0.5:
        signal       = "BUY (LONG)"
        signal_emoji = "[BUY]"
    elif expected_move_pct < -0.5:
        signal       = "SHORT (SELL)"
        signal_emoji = "[SHORT]"
    else:
        signal       = "HOLD (No clear signal)"
        signal_emoji = "[HOLD]"

    recent_prices = df['Close'].tail(20)
    daily_returns = recent_prices.pct_change().dropna()
    volatility    = daily_returns.std() * today_price

    stop_loss_distance   = 0.6 * volatility
    take_profit_distance = 1.0 * volatility

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

    # ── Multi-approach probability analysis ───────────────────────────────────
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
        model_type='lstm',         # same sequence interface as LSTM
        target_scaler=scaler_y,
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

    # ── Save ─────────────────────────────────────────────────────────────────
    ticker = os.path.basename(csv_file).split('_')[0]
    model_info = {
        'ticker':         ticker,
        'model_type':     f'CNN-1D + TFT (Keras + PyTorch + {_device})',
        'backend':        keras.backend.backend(),
        'gpu_used':       torch.cuda.is_available(),
        'n_features':     len(feature_cols),
        'lookback':       lookback,
        'architecture':   (f'Conv1D({cnn1_filters},{cnn1_kernel}) → '
                           f'Conv1D({cnn2_filters},{cnn2_kernel}) → '
                           f'Conv1D({cnn3_filters},{cnn3_kernel}) → '
                           f'GRN → LSTM({lstm_units}) → GRN+gate → '
                           f'MHA({n_heads}heads) → GRN+gate → '
                           f'GAP → Dense({dense_units})'),
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

    with open('tft_model_info.txt', 'w') as f:
        for key, value in model_info.items():
            f.write(f"{key}: {value}\n")

    print("\nModel saved as: best_tft_model.keras")
    print("Model info saved as: tft_model_info.txt")

    return model, history, model_info


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train CNN-1D + TFT model for stock price prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python train_tft.py MSFT_daily_data_20260520.csv
  python train_tft.py MSFT_daily_data_20260520.csv --d_model 128 --n_heads 8
  python train_tft.py MSFT_daily_data_20260520.csv --lookback 90 --epochs 200

Architecture parameters:
  CNN      : --cnn1_filters --cnn1_kernel --cnn2_filters --cnn2_kernel
             --cnn3_filters --cnn3_kernel --pool_size    --cnn_dropout
  TFT core : --d_model  --n_heads  --lstm_units  --grn_dropout  --attn_dropout
  Head     : --dense_units  --dense_dropout
        '''
    )
    parser.add_argument('csv_file',          type=str)
    parser.add_argument('--lookback',        type=int,   default=60)
    parser.add_argument('--epochs',          type=int,   default=150)
    parser.add_argument('--batch_size',      type=int,   default=32)
    # CNN
    parser.add_argument('--cnn1_filters',    type=int,   default=64)
    parser.add_argument('--cnn1_kernel',     type=int,   default=3)
    parser.add_argument('--cnn2_filters',    type=int,   default=256)
    parser.add_argument('--cnn2_kernel',     type=int,   default=5)
    parser.add_argument('--cnn3_filters',    type=int,   default=64)
    parser.add_argument('--cnn3_kernel',     type=int,   default=3)
    parser.add_argument('--pool_size',       type=int,   default=2)
    parser.add_argument('--cnn_dropout',     type=float, default=0.1)
    # TFT
    parser.add_argument('--d_model',         type=int,   default=64)
    parser.add_argument('--n_heads',         type=int,   default=4)
    parser.add_argument('--lstm_units',      type=int,   default=64)
    parser.add_argument('--grn_dropout',     type=float, default=0.1)
    parser.add_argument('--attn_dropout',    type=float, default=0.1)
    # Head
    parser.add_argument('--dense_units',     type=int,   default=32)
    parser.add_argument('--dense_dropout',   type=float, default=0.1)

    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: File {args.csv_file} not found!")
        exit(1)

    train_tft_model(
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
        d_model=args.d_model,
        n_heads=args.n_heads,
        lstm_units=args.lstm_units,
        grn_dropout=args.grn_dropout,
        attn_dropout=args.attn_dropout,
        dense_units=args.dense_units,
        dense_dropout=args.dense_dropout,
    )
