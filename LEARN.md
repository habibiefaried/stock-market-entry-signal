# Study Guide: Stock Market Entry Signal System

*Written as if your lecturer is sitting across the table from you.*

This document walks through every concept behind this codebase from raw maths to production code decisions. Read it top to bottom the first time, then use it as a reference.

---

## Table of Contents

1. [The Problem We Are Solving](#1-the-problem-we-are-solving)
2. [Time Series Fundamentals](#2-time-series-fundamentals)
3. [Technical Indicators](#3-technical-indicators)
4. [Machine Learning Foundations](#4-machine-learning-foundations)
5. [Tree Models: XGBoost, LightGBM, RandomForest](#5-tree-models-xgboost-lightgbm-randomforest)
6. [Neural Networks from First Principles](#6-neural-networks-from-first-principles)
7. [Convolutional Neural Networks (CNN-1D)](#7-convolutional-neural-networks-cnn-1d)
8. [Recurrent Networks and LSTM](#8-recurrent-networks-and-lstm)
9. [Attention Mechanisms](#9-attention-mechanisms)
10. [The Temporal Fusion Transformer (TFT)](#10-the-temporal-fusion-transformer-tft)
11. [Keras and the PyTorch Backend](#11-keras-and-the-pytorch-backend)
12. [Probability Analysis: Three Approaches](#12-probability-analysis-three-approaches)
13. [Risk Management Mathematics](#13-risk-management-mathematics)
14. [Data Pipeline and Feature Scaling](#14-data-pipeline-and-feature-scaling)
15. [Training Mechanics: Loss, Optimiser, Callbacks](#15-training-mechanics-loss-optimiser-callbacks)
16. [Evaluation Metrics](#16-evaluation-metrics)
17. [GPU Acceleration and Fallback Strategy](#17-gpu-acceleration-and-fallback-strategy)
18. [Reinforcement Learning: The PPO Meta-Agent](#18-reinforcement-learning-the-ppo-meta-agent)
19. [System Architecture: How main.py Orchestrates Everything](#19-system-architecture-how-mainpy-orchestrates-everything)
20. [File-by-File Reference](#20-file-by-file-reference)
21. [Common Pitfalls and How This Code Avoids Them](#21-common-pitfalls-and-how-this-code-avoids-them)
22. [What to Study Next](#22-what-to-study-next)

---

## 1. The Problem We Are Solving

You are given a time series of daily stock prices (Open, High, Low, Close, Volume -- OHLCV). You want to answer one question:

> **"If I enter a trade right now with a given Stop Loss and Take Profit, what is the probability that price hits TP before SL within the next 5 days?"**

This requires:

- **Feature engineering** -- transforming raw prices into signals a model can learn from
- **Sequence modelling** -- prices have memory; yesterday affects today
- **Probabilistic output** -- a single price prediction is not enough; you need a distribution over outcomes
- **Ensemble reasoning** -- no single model is reliable; combine multiple approaches
- **Meta-agent decision** -- a PPO RL agent reads all model outputs and makes a final LONG/SHORT/HOLD call

The system trains 7 models (CNN-LSTM, CNN-TFT, XGBoost, XGBoost-Heavy, LightGBM, LightGBM-Heavy, RandomForest), aggregates their opinions, then passes everything to a PPO reinforcement learning agent that outputs the final trade recommendation.

---

## 2. Time Series Fundamentals

### 2.1 What makes time series different from tabular data

In standard supervised learning, rows are i.i.d. (independently and identically distributed). You can shuffle rows and the model still learns correctly.

In a time series, row t depends on row t-1. Shuffling destroys the signal. This has cascading consequences:

- **No random train/test split.** You must use a chronological split: train on the past, test on the future. This code uses a 90/10 chronological split.
- **No cross-validation in the usual sense.** You need walk-forward validation (section 5.3).
- **Lag features.** Tree models cannot see sequences; you give them the past explicitly by creating `Close_lag_1`, `Close_lag_2`, etc.
- **Look-ahead bias.** A lethal mistake: if any future information leaks into your training features, your model looks good in backtesting but fails in live trading. This code always shifts targets forward with `shift(-1)` and scales only on training data.

### 2.2 The target

All models predict **tomorrow's closing price** (regression). Direction (BUY/SELL/HOLD) is derived from the predicted price vs today:

```
if (predicted - today) / today > 0.005  ->  BUY
if (predicted - today) / today < -0.005 ->  SHORT
else                                     ->  HOLD
```

---

## 3. Technical Indicators

All 52 features in `FEATURE_COLS` (and `INDICATOR_COLS`) are computed in `compute_technical_indicators()`. Every training script computes the same 52 indicators so all models use identical feature sets.

### 3.1 Moving Averages

**Simple Moving Average (SMA):**
```
SMA_p(t) = (1/p) * sum(Close(t-i))  for i = 0..p-1
```
Periods: 5, 10, 20, 50, 100, 200.

**Exponential Moving Average (EMA):**
```
EMA(t) = Close(t) * k + EMA(t-1) * (1-k)    where k = 2 / (p+1)
```
Periods: 9, 21, 50, 100.

**MA Ratios** (dimensionless distance from moving average):
```
Close_SMA20_ratio = (Close - SMA_20) / SMA_20
```

### 3.2 RSI -- Relative Strength Index

```
RS = EMA(gains, p) / EMA(losses, p)
RSI = 100 - (100 / (1 + RS))
```

Bounded [0, 100]. Above 70: overbought. Below 30: oversold. Periods: 7, 14, 21.

### 3.3 MACD

```
MACD_line   = EMA(12) - EMA(26)
MACD_signal = EMA(MACD_line, 9)
MACD_hist   = MACD_line - MACD_signal
```

The histogram measures acceleration of the trend. Zero crossings signal potential trend changes.

### 3.4 Bollinger Bands

```
BB_mid   = SMA(20)
BB_upper = BB_mid + 2 * std(20)
BB_lower = BB_mid - 2 * std(20)
BB_pct   = (Close - BB_lower) / (BB_upper - BB_lower)
BB_width = (BB_upper - BB_lower) / BB_mid
```

BB_pct in [0,1] approximately. BB_width measures volatility -- a squeeze (narrow bands) often precedes a large move.

### 3.5 ATR -- Average True Range

```
TR(t) = max(High-Low, |High-Close(t-1)|, |Low-Close(t-1)|)
ATR_p  = EMA(TR, p)
```

Measures volatility in dollar terms. Periods: 7, 14.

### 3.6 Stochastic Oscillator

```
%K = 100 * (Close - Low_14) / (High_14 - Low_14)
%D = SMA(%K, 3)
```

Bounded [0, 100]. Like RSI but based on price range rather than price changes.

### 3.7 OBV -- On-Balance Volume

```
OBV(t) = OBV(t-1) + sign(Close(t) - Close(t-1)) * Volume(t)
```

Cumulates volume direction. The code log-scales to prevent unbounded growth.

### 3.8 Other Indicators

- **CCI** -- (TP - SMA(TP)) / (0.015 * MAD). Zero-centred, unbounded. Period 14.
- **Williams %R** -- -100 * (High14 - Close) / (High14 - Low14). Range [-100, 0].
- **ROC** -- (Close(t) - Close(t-p)) / Close(t-p) * 100. Periods 1, 5, 10.
- **MOM** -- Close(t) - Close(t-p). Periods 5, 10.
- **Volume_log** -- log1p(Volume). Compresses heavy tail.
- **Volume_MA20_ratio** -- Volume / SMA(Volume,20). Above 1 = above-average volume.
- **Price changes** -- 1d, 3d, 5d pct_change * 100.
- **Volatility** -- rolling std of returns * 100. Periods 5, 10, 20.
- **HL_range_pct** -- (High-Low)/Close * 100.
- **HL_vs_ATR14** -- (High-Low)/ATR_14. Above 1 = unusually wide day.

### 3.9 Heavy-Model Features (XGBoost-Heavy, LightGBM-Heavy)

The heavy models use a **self-contained 30-indicator set** (not a superset of the 52-indicator LSTM/TFT set). The design rule is **max 2 per indicator family** to limit collinearity.

```
-- Price / Volume (5) --
Open, High, Low, Close, Volume

-- Trend (4) --
SMA_20, SMA_50
Close_SMA20_ratio, Close_SMA50_ratio   -- dimensionless; drop SMA_100/200

-- Momentum (2) --
EMA_9, EMA_21                          -- drop EMA_12/26 (redundant with MACD inputs)

-- RSI (2) --
RSI_7, RSI_14                          -- drop RSI_21 (diminishing returns)

-- MACD (3) --
MACD_line, MACD_signal, MACD_hist      -- counted as 1 family

-- Bollinger (2) --
BB_pct, BB_width                       -- drop BB_upper/lower/mid (same unit as Close)

-- Volatility / Range (1) --
ATR_14                                 -- drop ATR_7 (ATR_14 is the standard)

-- Oscillators (2) --
STOCH_K, STOCH_D                       -- drop WILLR (identical concept to STOCH)

-- Volume (2) --
OBV, Volume_MA20_ratio                 -- OBV is cumulative direction; ratio is relative level

-- Momentum (1) --
CCI_14                                 -- drop CCI_20 (1 period sufficient)

-- Price Transforms (4) --
Volume_log, Price_change_1d, Price_change_5d, HL_range_pct

-- Volatility (2) --
Volatility_5d, Volatility_20d          -- drop Volatility_10d/30d
```

On top of these 30, three derived features are added:

```
Close_lag_1, Close_lag_3, Close_lag_5  -- lagged close (3 periods, not 5)
Volume_lag_1, Volume_lag_5             -- lagged volume (2 periods)
RSI14_slope_3d    -- RSI_14.diff(3)  (RSI momentum)
MACD_accel        -- MACD_hist.diff(1)  (MACD acceleration)
BB_squeeze        -- BB_width / BB_width.rolling(20).mean()  (<1 = squeeze)
```

**Total: 38 features** (30 base + 5 lags + 3 derived).

**What was dropped and why**:

| Dropped | Reason |
|---------|--------|
| WILLR | Same concept as Stochastic %K -- both measure Close relative to the High-Low range |
| ROC, MOM | Both are just price changes over a window -- already covered by Price_change_1d/5d |
| HL_vs_ATR14 | ATR_14 already captures typical range; this ratio adds noise without new signal |
| Raw BB_upper/lower/mid | In the same unit as Close -- collinear. BB_pct and BB_width are dimensionless substitutes |
| ATR_7 | ATR_14 is the market-standard period; two ATR periods add minimal independent signal |
| CCI_20 | One CCI period (14) is sufficient; CCI_20 is 86% correlated with CCI_14 |
| RSI_21 | RSI_7 and RSI_14 already span short and medium momentum; RSI_21 adds very little |
| Extra SMA/EMA periods | SMA_20/50 + EMA_9/21 give four trend reference points -- enough |
| Close_lag_2/4, Volume_lag_2/3/4 | Redundant given lags 1, 3, 5 already sample the shape of the lag curve |
| Price_change_10d, Volatility_30d | Replaced by shorter, less redundant periods |

---

## 4. Machine Learning Foundations

### 4.1 Supervised Learning

You have features **X** and a target **y**. Fit `f(X) ~= y`. Here y is tomorrow's close price.

### 4.2 Overfitting vs Underfitting

- **Overfitting**: memorises training data, fails on test. Train loss << Test loss.
- **Underfitting**: too simple for the pattern. Both losses are high.

Regularisation used in this code:
- Dropout (LSTM/TFT): randomly zeros activations during training
- EarlyStopping: stops when validation loss stops improving
- ReduceLROnPlateau: halves learning rate when progress stalls
- max_depth / num_leaves: limits tree complexity
- min_data_in_leaf: prevents leaves on tiny samples

### 4.3 Bias-Variance Tradeoff

High bias = underfitting (too simple). High variance = overfitting (too sensitive to training noise). Ensemble methods reduce variance by averaging many models.

---

## 5. Tree Models: XGBoost, LightGBM, RandomForest

### 5.1 Decision Trees

Split the feature space with axis-aligned cuts. Each leaf is a prediction. Fast, interpretable, but overfit easily if deep.

### 5.2 Gradient Boosting (XGBoost, LightGBM)

Build trees **sequentially**. Each new tree fits the residuals of all previous trees:

```
F_0(x) = mean(y)
F_m(x) = F_{m-1}(x) + lr * h_m(x)
```

Where h_m is fitted to the negative gradient of the loss.

**XGBoost**: adds second-order (Hessian) information, L1/L2 regularisation, level-wise tree growth. Key params: `n_estimators=3000`, `learning_rate=0.005`, `max_depth=8`.

**LightGBM**: leaf-wise growth (always split the leaf with highest loss reduction), GOSS sampling, histogram-based splits. Faster than XGBoost for the same n_estimators. Key param: `num_leaves=63`.

**Why 3000 trees + lr=0.005 in heavy models**: halving the learning rate requires doubling n_estimators to fit the same signal -- but the resulting function is smoother and less overfit. `EarlyStopping(rounds=50)` prevents wasted compute.

**bagging_freq must accompany bagging_fraction in LightGBM**: if `bagging_fraction < 1.0` is set without `bagging_freq`, LightGBM silently ignores bagging. The code sets `bagging_freq=5`.

### 5.3 Random Forest and Walk-Forward Validation

Random Forest trains many trees **in parallel**. Each tree uses a bootstrapped sample and `sqrt(n_features)` random features per split. Final prediction is the average.

**Walk-Forward Validation** (used in `train_randomforest.py`):
```
Fold 1: train [0, 700],   test [700, 760]
Fold 2: train [0, 760],   test [760, 820]
...
Fold 5: train [0, 940],   test [940, 1000]
```
Each fold simulates live trading. Metrics are aggregated across folds.

### 5.4 Lag Features for Tree Models

Trees work on flat vectors, not sequences. Lag features give temporal context:
```
Close_lag_1 = yesterday's close
Close_lag_2 = 2 days ago
...
```
The tree learns patterns like "when price fell 3 consecutive days AND RSI < 30, it tends to bounce."

---

## 6. Neural Networks from First Principles

### 6.1 The Neuron

```
output = activation(W . x + b)
```

Without non-linear activations, stacking layers is a single linear transform. Non-linearities allow networks to approximate any function.

**Activations in this code:**
- `ReLU(x) = max(0, x)` -- CNN layers. Fast, no vanishing gradient for positive inputs.
- `ELU(x) = x if x>0, else (e^x - 1)` -- GRN dense layers. Smooth negative activation.
- `tanh` -- LSTM gates, TemporalAttention.
- `sigmoid` -- LSTM gates, TFT gating. Output in (0,1).

### 6.2 Backpropagation

1. Forward pass: compute prediction
2. Compute loss
3. Backward pass: chain rule for gradients
4. Update: `W <- W - lr * gradient`

The vanishing gradient problem (gradients shrink through deep layers) is solved by LSTM, BatchNorm, and residual connections.

### 6.3 Batch Normalisation

```
x_norm = (x - mean(x)) / (std(x) + epsilon)
output = gamma * x_norm + beta
```

Keeps activations in a healthy range. Dramatically stabilises training of deep networks.

### 6.4 Dropout

During training, randomly zero a fraction `p` of activations. Forces the network to not rely on any single path. Disabled at inference time.

---

## 7. Convolutional Neural Networks (CNN-1D)

### 7.1 Why 1D Convolution for Time Series

A 1D conv slides a small filter across the time axis:
```
output(t) = sum_k sum_f  x(t+k, f) * w(k, f)  + bias
```

Properties:
- **Local receptive field**: each output sees only `kernel_size` consecutive timesteps
- **Weight sharing**: same filter applied at every timestep
- **Translation invariance**: detects a pattern wherever it appears in the sequence

### 7.2 The Three CNN Blocks in This Code

```python
# Block 1: detect fine-grained patterns
Conv1D(64, kernel=3, relu, padding='same') -> BatchNorm

# Block 2: broader patterns + halve time dimension
Conv1D(256, kernel=5, relu, padding='same') -> BatchNorm -> MaxPooling1D(2)

# Block 3: refine
Conv1D(64, kernel=3, relu, padding='same') -> BatchNorm -> Dropout(0.25)
```

256 filters in block 2 means 256 different pattern detectors run in parallel. MaxPool halves the sequence length, reducing compute for the LSTM.

### 7.3 CNN vs LSTM

| | CNN | LSTM |
|---|---|---|
| Scope | Short-range (3-5 timesteps) | Long-range (full lookback) |
| Detects | Local patterns (crossovers, divergences) | How patterns evolve over weeks |

Stacking them: CNN extracts *what is happening now*, LSTM tracks *how it has been evolving*.

---

## 8. Recurrent Networks and LSTM

### 8.1 The Vanishing Gradient Problem

A simple RNN multiplies `W_h` by itself at every timestep during backprop. If its eigenvalue < 1, gradients vanish. If > 1, they explode. Simple RNNs cannot learn dependencies beyond ~10 timesteps.

### 8.2 LSTM Gates

```
f_t = sigmoid(W_f . [h_{t-1}, x_t] + b_f)   -- forget: how much of c_{t-1} to keep
i_t = sigmoid(W_i . [h_{t-1}, x_t] + b_i)   -- input: how much new info to write
g_t = tanh(W_g . [h_{t-1}, x_t] + b_g)       -- candidate cell update
o_t = sigmoid(W_o . [h_{t-1}, x_t] + b_o)   -- output: what to expose as h_t

c_t = f_t * c_{t-1} + i_t * g_t
h_t = o_t * tanh(c_t)
```

The cell state `c_t` flows through with only element-wise operations -- a gradient highway that enables learning dependencies spanning 50-200 timesteps.

### 8.3 Stacked LSTM with return_sequences

```python
x = LSTM(128, return_sequences=True)(x)   # (batch, timesteps, 128)
x = LSTM(64, return_sequences=True)(x)    # (batch, timesteps, 64)
```

`return_sequences=True` outputs h_t at every timestep. Required because: (1) the second LSTM needs a full sequence, (2) the Attention layer needs all timesteps.

---

## 9. Attention Mechanisms

### 9.1 Why Attention

Not every day in the 60-day lookback is equally important. A breakout 3 weeks ago might be far more predictive than yesterday's sideways movement. Attention learns a soft weight per timestep.

### 9.2 Temporal Attention (CNN-LSTM)

```
e_t  = tanh(W^T h_t + b)   -- unnormalised score per timestep
a_t  = softmax(e)_t         -- normalised weight (sum to 1)
c    = sum_t  a_t * h_t     -- context vector (weighted average)
```

One learnable parameter W (64x1). Collapses (batch, timesteps, 64) to (batch, 64).

### 9.3 Multi-Head Self-Attention (CNN-TFT)

```
Q = X @ W_Q,  K = X @ W_K,  V = X @ W_V
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(d_k) ) @ V
```

With 4 heads, each head learns a different attention pattern. The sqrt(d_k) scaling prevents softmax saturation. Unlike LSTM, attention can connect any two timesteps directly -- no need to propagate information step by step.

---

## 10. The Temporal Fusion Transformer (TFT)

### 10.1 Gated Linear Unit (GLU)

```python
projected = Dense(units * 2)(x)
x1 = projected[..., :units]
x2 = projected[..., units:]
output = x1 * sigmoid(x2)   # gate: x2 controls how much of x1 flows
```

The sigmoid output in (0,1) acts as a soft gate. The network learns to suppress irrelevant paths (gate -> 0) or pass them through fully (gate -> 1).

### 10.2 Gated Residual Network (GRN)

```python
residual = Linear(x)           # skip connection
h = ELU(Dense(units)(x))
h = Dense(units)(h)
h = GLU(units)(h)              # gate can suppress entire transformation
h = Dropout(h)
output = LayerNorm(residual + h)
```

Key: if the transformation is unhelpful, the gate closes and the residual dominates. The network degrades gracefully rather than amplifying noise.

### 10.3 Full TFT Data Flow

```
Input (batch, lookback, 52 features)
  -> CNN x3           extract local cross-indicator patterns
  -> Dense(d_model=64) project to uniform width
  -> GRN              non-linear per-timestep transform
  -> LSTM(64, seq)    local temporal context
  -> GRN + gate + skip + LayerNorm
  -> MultiHeadAttn(4) long-range dependencies
  -> GRN + gate + skip + LayerNorm
  -> GlobalAvgPool1D  collapse time axis
  -> Dense(32, relu) -> Dropout -> Dense(1)
```

The gated skip connections around LSTM and attention mean: early in training (when those blocks are noisy), the gate can be near zero and the network bypasses them. As training progresses, the gates open.

---

## 11. Keras and the PyTorch Backend

### 11.1 Setting the Backend

```python
os.environ['KERAS_BACKEND'] = 'torch'  # Must be BEFORE any keras import
```

Keras 3.x supports TensorFlow, PyTorch, and JAX backends. PyTorch is used here for better GPU support on some hardware.

### 11.2 Custom Keras Layers

```python
class TemporalAttention(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(units, 1), trainable=True)

    def call(self, inputs):
        ...

    def get_config(self):
        return super().get_config()   # required for save/load
```

`get_config()` is critical. Without it, `load_model` fails with "Unknown layer".

### 11.3 ModelCheckpoint and EarlyStopping

```python
checkpoint = ModelCheckpoint('best.keras', monitor='val_loss', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
```

After training, the code reloads from checkpoint:
```python
model = keras.models.load_model('best.keras', custom_objects={...})
```
Belt-and-suspenders: guarantees the saved file matches what is used for predictions.

---

## 12. Probability Analysis: Three Approaches

These live in `trade_probability_analyzer.py` and are called by every model script.

### 12.1 Approach 1: Multi-Day Sequential Prediction

The model predicts Day 1. That prediction becomes input for Day 2, and so on for 5 days. At each step, check if price crossed TP or SL.

After appending a predicted row, all 52 indicators must be recomputed -- `recalculate_features()` does this. Error compounds over 5 days (Day 5 prediction depends on 4 prior predictions).

Output: binary (TP or SL hit) -> contributes 100% or 0% to the ensemble.

### 12.2 Approach 2: Monte Carlo Simulation

Run 1000 independent random price walks:
```
daily_return ~ Normal(drift, volatility/current_price)
price(t+1) = price(t) * (1 + daily_return)
drift = predicted_move_pct * 0.3 / n_days
```

The 0.3 (MC_DRIFT_INFLUENCE) means the model's direction biases the walk by 30%, but 70% is pure noise. Win rate = fraction of 1000 runs where price hits TP before SL.

### 12.3 Approach 3: Historical Pattern Matching

Find historical days where RSI, volatility, and trend direction were similar to today (RSI +/- 5 points, volatility within 20%, same trend direction). Top 50 matches are found. For each, check what actually happened in the next 5 days.

### 12.4 Ensemble Combination

```
ensemble_prob = 0.4 * approach1 + 0.35 * approach2 + 0.25 * approach3
```

Weights normalise if any approach is unavailable. Final call:
- >= 60%: TAKE TRADE
- >= 75%: HIGH confidence
- 65-75%: MEDIUM confidence
- < 65%: LOW confidence

---

## 13. Risk Management Mathematics

### 13.1 Stop Loss and Take Profit

```python
volatility = Close.tail(20).pct_change().std() * today_price  # $ per day
stop_loss   = today_price - 0.6 * volatility   # for LONG
take_profit = today_price + 1.0 * volatility   # for LONG
```

Risk/reward = 1.0 / 0.6 = 1.67:1. Break-even win rate = 1 / (1 + 1.67) = 37.5%. The 60% confidence threshold is well above this.

### 13.2 5x Leverage P&L

```
position_pnl% = stock_move% * 5
```

A -1.5% stock move becomes -7.5% position loss with 5x leverage. The report shows these numbers to make the real-money impact clear before entering a trade.

---

## 14. Data Pipeline and Feature Scaling

### 14.1 MinMaxScaler (LSTM/TFT)

```
x_scaled = (x - x_min) / (x_max - x_min)  -> output in [0, 1]
```

**Fit only on training data.** Test data uses the same min/max parameters:
```python
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)  # learns min/max
X_test_scaled  = scaler_X.transform(X_test)        # applies same min/max
```

**Separate scalers for X and y**: the target (Close price) needs its own `scaler_y` so that `scaler_y.inverse_transform(pred)` converts predicted scaled values back to dollars cleanly.

### 14.2 StandardScaler (GBM models)

```
x_scaled = (x - mean) / std   -> mean=0, std=1
```

Tree models are scale-invariant but scaling is applied for consistency.

### 14.3 Sequence Creation (LSTM/TFT)

```python
for i in range(lookback, len(X)):
    Xs.append(X[i-lookback:i])   # (lookback, n_features)
    ys.append(y[i])              # label = day after the window
```

Lookback=60 means 60 days of context to predict day 61. Adjacent windows overlap by 59 days -- this is intentional.

---

## 15. Training Mechanics: Loss, Optimiser, Callbacks

### 15.1 Huber Loss

```
L(y, y_hat) = 0.5*(y-y_hat)^2       if |y-y_hat| <= delta
            = delta*(|y-y_hat| - 0.5*delta)    otherwise
```

Quadratic near zero (like MSE), linear for large errors (like MAE). Robust to overnight price gaps without the instability of MSE.

### 15.2 Adam Optimiser

Maintains per-parameter adaptive learning rates using first moment (gradient mean) and second moment (gradient variance). Default lr=1e-3, beta1=0.9, beta2=0.999. Parameters with large consistent gradients get smaller updates; noisy parameters adapt their own rates.

### 15.3 EarlyStopping and ReduceLROnPlateau

- `patience=15`: stop if val_loss does not improve for 15 epochs
- `ReduceLROnPlateau(factor=0.5, patience=7)`: halve lr after 7 epochs of no improvement
- `restore_best_weights=True`: revert to the best weights, not the final

---

## 16. Evaluation Metrics

### 16.1 Regression Metrics

- **MAE** = mean(|y - y_hat|). In dollars. Interpretable: "off by $X on average."
- **RMSE** = sqrt(mean((y - y_hat)^2)). Penalises large errors more. RMSE > MAE always.

### 16.2 Direction Metrics

```python
y_true_dir = (np.diff(y_true) > 0).astype(int)   # 1=up, 0=down
y_pred_dir = (np.diff(y_pred) > 0).astype(int)
```

- **Accuracy**: correct direction / total. Random baseline = 50%.
- **Precision**: TP / (TP + FP). Of times you predicted "up", how often was it actually up?
- **Recall**: TP / (TP + FN). Of actual "up" days, how many did you catch?
- **F1-Score**: harmonic mean of precision and recall.

For 5x leverage trading, **precision matters more than recall** -- a wrong direction entry loses 7.5% of position.

---

## 17. GPU Acceleration and Fallback Strategy

### 17.1 Tree Model GPU Fallback Pattern

```python
try:
    model = _make_model(use_gpu=True)
    model.fit(X_train, y_train, ...)   # GPU error may appear here, not at init
    _using_gpu = True
except Exception as e:
    model = _make_model(use_gpu=False)
    model.fit(X_train, y_train, ...)
```

Both instantiation AND fit are in the try block because some XGBoost/LightGBM versions raise GPU errors during `.fit()`, not during construction.

### 17.2 Keras/LSTM GPU

Automatic via PyTorch backend. If `torch.cuda.is_available()`, all tensor ops run on GPU. No explicit try/except needed.

---

## 18. Reinforcement Learning: The PPO Meta-Agent

This is the newest component, living in `agent_trader.py`.

### 18.1 Why RL on top of 7 models?

Each of the 7 models produces a signal (BUY/SHORT/HOLD) and a TP win probability. A human trader would look at all 7 and decide whether to trade. The PPO agent learns to do this automatically -- it discovers which combinations of model signals actually lead to profitable trades.

### 18.2 State, Action, Reward

**State vector (16 dimensions):**
```
[xgboost_signal,        xgboost_prob,
 xgboost_heavy_signal,  xgboost_heavy_prob,
 lightgbm_signal,       lightgbm_prob,
 lightgbm_heavy_signal, lightgbm_heavy_prob,
 randomforest_signal,   randomforest_prob,
 lstm_signal,           lstm_prob,      <- reads lstm_signal.txt (written by train_lstm.py)
 tft_signal,            tft_prob,       <- reads tft_signal.txt  (written by train_tft.py)
 (RSI_14 - 50) / 50,                   <- normalised RSI
 trend]                                 <- (Close - SMA20) / SMA20
```

Signals are encoded: LONG=1, SHORT=-1, HOLD=0.

The `prob` for LSTM and TFT uses the **per-trade ensemble probability** (Monte Carlo + pattern match + sequential prediction) written by those model scripts -- not a static test-set accuracy. This is higher quality than the tree model probs, which scale with predicted move magnitude.

**Actions:**
- `LONG (0)`: go long, hold until TP or SL
- `SHORT (1)`: go short, hold until TP or SL
- `HOLD (2)`: skip this trade opportunity

**Rewards:**
```
TP hit first  -> +1.67   (matches 1.5x ATR take-profit vs 1.0x ATR stop-loss)
SL hit first  -> -1.0
Each day held ->  -0.05  (cost of holding without resolution)
Max 5 days    (then episode ends as TIMEOUT)
```

**TP/SL levels** (consistent across all model scripts and the RL environment):
```
Stop Loss   = entry +/- 1.0 * ATR_14
Take Profit = entry +/- 1.5 * ATR_14
Risk/Reward = 1.5:1
```
Using ATR_14 instead of rolling return-std captures intraday gap risk that return-std misses. The RL reward ratio (1.67) approximates the ATR-based R:R of 1.5 -- they are close enough that the agent's learned policy transfers correctly to live levels.

### 18.3 Double Walk-Forward Validation

**Why double?** A single walk-forward would use the same data to generate model predictions AND train the RL agent -- giving the agent access to future data it would not have in live trading.

**Layer 1 (in `load_model_predictions`):**
Load the trained pkl models (XGBoost, LightGBM, etc.) and run them in a walk-forward manner on historical data. The first 60% of data was already used to train those models. For the remaining 40%, generate predictions row by row -- each prediction is genuinely out-of-sample.

**Layer 2 (in `train_ppo`):**
Split those out-of-sample signal rows 80/20. Train the PPO agent on the first 80% of signals. Evaluate (backtest) on the held-out 20%.

This two-layer structure ensures the RL agent has never seen the validation data in any form.

### 18.4 PPO Algorithm

PPO (Proximal Policy Optimization) is the most widely used policy gradient algorithm. It improves the policy while preventing updates that are too large (which would destabilise training).

**Core idea:**
```
ratio = pi_new(a|s) / pi_old(a|s)   -- how much the policy changed
L_CLIP = min(ratio * advantage,
             clip(ratio, 1-eps, 1+eps) * advantage)
```

The clip prevents the new policy from deviating too far from the old one in a single update step. `eps=0.2` is the standard value.

**Advantage** = return - value_estimate. Positive advantage means "this action led to better-than-expected outcome; do it more."

**This implementation:** Pure numpy MLP (no PyTorch/Keras), so it runs without GPU and adds no new dependencies. The gradient update is approximated rather than exact backprop -- sufficient for the low-dimensional state space (12 dims).

### 18.5 Performance Targets

| Metric | Target | Meaning |
|--------|--------|---------|
| Win Rate | >= 60% | More than 60% of LONG/SHORT trades hit TP |
| Profit Factor | >= 1.5 | Gross profit / gross loss |
| Sharpe Ratio | >= 1.0 | Risk-adjusted return (annualised) |
| Max Drawdown | > -20% | Worst peak-to-trough |
| Trades/Month | >= 4 | Enough activity to be useful |

Early in training (with limited data), these targets may not be met. The agent needs 8-15 years of data to see full market cycles (bull, crash, recovery, sideways).

### 18.6 Signal File Loading (LSTM / TFT)

LSTM and TFT cannot be loaded by the RL agent like pkl files (they require Keras with custom layers). Instead, when they finish training they write a small text file:

```
lstm_signal.txt
  signal: 1          (1=LONG, -1=SHORT, 0=HOLD)
  prob: 0.73         (directional probability)
  ensemble_prob: 68.4  (weighted average of 3-method probability analysis)
```

The RL agent reads this file at startup. If the file does not exist (LSTM/TFT not yet trained), those two slots in the state vector default to 0/0.5 (neutral).

### 18.7 Weight Persistence

After each training run, the PPO weights are saved to `rl_agent_weights.npz`. On the next run with the same CSV file (same size and row count), the agent warm-starts from those weights instead of random initialisation. This means the second run is faster and the agent starts from a better policy. A new CSV (different ticker or date range) triggers full retraining.

### 18.8 Fallback When No PKL Models Exist

If the 7 model `.pkl` files have not been trained yet, `_synthetic_signals()` generates proxy signals from the raw indicators (RSI, MACD, trend) for all 7 model slots. This allows `agent_trader.py` to run standalone for testing, but the quality will be lower than when real model outputs are used.

---

## 19. System Architecture: How main.py Orchestrates Everything

### 19.1 Parallel Model Training

```python
with ThreadPoolExecutor(max_workers=7) as executor:
    futures = {executor.submit(run_model, name, script, csv): name
               for name, script in models}
    for future in as_completed(futures):
        results.append(future.result())
```

Each model runs as a **subprocess** (`subprocess.run`), not a thread. Python's GIL prevents true CPU parallelism in threads, but subprocesses are independent OS processes. On a GPU machine, all 7 subprocesses can use the GPU simultaneously via CUDA's internal scheduler.

`as_completed` yields results as they finish (fastest model reports first).

### 19.2 Sequential RL Agent

After all 7 models finish, `run_agent(csv_file)` is called **sequentially** -- it needs the trained `.pkl` files from the model step. It runs as a subprocess with a 600-second timeout.

### 19.3 Output Parsing

Each training script prints in a known format. `main.py` extracts data with regex:
```python
mae_match = re.search(r'Test MAE:\s+\$?([\d.]+)', output)
```

The RL agent prints:
```
AGENT_ACTION:     LONG
AGENT_CONFIDENCE: 67.3%
AGENT_WINRATE:    61.2%
```

Parsed by `parse_agent_output()` in main.py.

### 19.4 HTML Report Structure

The report is built as a list of strings, then joined and written to disk. The RL agent section appears **at the very top** of the report body, above the Executive Summary:

```
[Header: ticker, date, models count]
[RL AGENT DECISION BLOCK]       <- dark card, top of page
[Executive Summary]
[Trading Signals for each model]
[Signal Consensus]
[Detailed Results]
[Recommendations]
```

---

## 20. File-by-File Reference

| File | Purpose | Output Files |
|------|---------|--------------|
| `main.py` | Orchestrator: runs all models in parallel, calls RL agent, generates HTML report | `RESULT-{TICKER}-{DATE}.html` |
| `fetch_stock_data.py` | Standalone data fetcher using yfinance | `{TICKER}_daily_data_{DATE}.csv` |
| `train_lstm.py` | CNN-1D + stacked LSTM + Temporal Attention | `best_lstm_model.keras`, `lstm_model_info.txt` |
| `train_tft.py` | CNN-1D + GRN + LSTM + Multi-Head Attention (TFT-inspired) | `best_tft_model.keras`, `tft_model_info.txt` |
| `train_xgboost.py` | XGBoost baseline (5 OHLCV features) | `xgboost_model.pkl`, `xgboost_scaler.pkl`, `xgboost_features.txt` |
| `train_xgboost_heavy.py` | XGBoost with 38 features (max 2 per indicator family), 3000 trees | `xgboost_heavy_model.pkl`, `xgboost_heavy_scaler.pkl`, `xgboost_heavy_features.txt` |
| `train_lightgbm.py` | LightGBM baseline (5 OHLCV features) | `lightgbm_model.pkl`, `lightgbm_scaler.pkl`, `lightgbm_features.txt` |
| `train_lightgbm_heavy.py` | LightGBM with 38 features (max 2 per indicator family), 3000 trees, num_leaves=63 | `lightgbm_heavy_model.pkl`, `lightgbm_heavy_scaler.pkl`, `lightgbm_heavy_features.txt` |
| `train_randomforest.py` | Random Forest with walk-forward validation | `randomforest_model.pkl`, `randomforest_scaler.pkl`, `randomforest_features.txt` |
| `agent_trader.py` | PPO RL meta-agent: reads all 7 model signals (incl. LSTM/TFT via signal files), outputs LONG/SHORT/HOLD | `rl_agent_weights.npz` (warm-start on next run) |
| `trade_probability_analyzer.py` | Three-approach win probability analysis, called by all model scripts | (no file output -- returns results) |
| `test_gpu.py` | Quick GPU availability check | (stdout only) |

### 20.1 How to Run

```bash
# Install dependencies
pip install -r requirements.txt
# For GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# For CPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Fetch data and run all models + RL agent (default: 96 months = 8 years)
python main.py --ticker MSFT
python main.py --ticker MSFT --months 120   # 10 years (even more data for RL agent)

# Use existing CSV
python main.py MSFT_daily_data_20260524.csv

# Run RL agent standalone (needs pkl files already trained)
python agent_trader.py MSFT_daily_data_20260524.csv

# Check GPU
python test_gpu.py
```

### 20.2 Data Format

The CSV must have these columns (case-sensitive):
```
Date, Open, High, Low, Close, Volume
```

yfinance produces this format automatically.

---

## 21. Common Pitfalls and How This Code Avoids Them

### Look-ahead bias in scaling
**Wrong**: `scaler.fit_transform(all_data)` -- test data's future prices shift the scaler.
**Right**: `scaler.fit_transform(train_data)`, then `scaler.transform(test_data)`.

### Wrong inverse transform for multi-output scalers
**Wrong**: putting the prediction into the Close column of a 52-feature scaled array.
**Right**: separate `scaler_y` fitted only on Close.

### Missing custom_objects on model reload
**Wrong**: `load_model('model.keras')` -- crashes for custom layers.
**Right**: `load_model('model.keras', custom_objects={'TemporalAttention': TemporalAttention})`.

### GPU error only caught at instantiation, not fit
**Wrong**: try/except only around `model = XGBRegressor(device='cuda')`.
**Right**: try/except around both instantiation AND `.fit()`.

### Sequences spanning train/test boundary
**Wrong**: creating sequences that include training data in the test window.
**Right**: create sequences only from `X_test_scaled`. The first `lookback` rows are consumed as context -- so `test_dates = test_df['Date'].values[lookback:]`.

### LightGBM bagging silently ignored
**Wrong**: setting `bagging_fraction=0.8` without `bagging_freq`.
**Right**: always set `bagging_freq=5` alongside `bagging_fraction`.

### Emoji / non-ASCII in print strings
**Wrong**: any emoji, em-dash, bullet, or non-ASCII character in a string that gets printed to stdout.
**Right**: pure ASCII only. All files in this repo are verified clean with a byte-level scan.

### Undefined loop variable when loop body is empty
**Wrong**: `for d, fp in enumerate(futures): ...` then `d + 1` after -- `d` is unbound if `futures` is empty.
**Right**: initialise `days_out = 1` before the loop and set `days_out = d + 1` inside.

### TP/SL inconsistency between model scripts and RL environment
**Wrong**: model scripts use `0.6 * return_std * price` for SL and `1.0 * return_std * price` for TP, while the RL environment uses different multipliers. The agent learns to hit a TP defined differently from what the models use.
**Right**: all model scripts and the RL `TradingEnv.step()` use `SL = 1.0 * ATR_14`, `TP = 1.5 * ATR_14`. ATR_14 captures intraday gap risk that return-std misses and is the industry-standard measure for position sizing.

### RL agent misses the two strongest models
**Wrong**: `load_model_predictions()` only loads `.pkl` files. LSTM and TFT save `.keras` files with custom layers -- trying to load them the same way would require importing Keras + custom objects at agent startup.
**Right**: LSTM and TFT write `lstm_signal.txt` / `tft_signal.txt` at the end of their training runs. The RL agent reads these files. The state vector grows from 12 to 16 dims to include both.

### Static per-model confidence (test accuracy) in RL state
**Wrong**: using `test_acc * 100` as the `prob` fed into the RL state for every single episode. This is a one-time number that never changes -- the agent can't distinguish high-confidence from low-confidence signals.
**Right**: use per-trade probabilities where available (ensemble_prob from trade_probability_analyzer for LSTM/TFT) and move magnitude scaling for tree models. Each row in the training data then has a different prob, which is what the RL agent needs to learn from.

### Fixed signal threshold ignores asset volatility
**Wrong**: `if expected_move_pct > 0.5:` -- a 0.5% threshold means TSLA (3% daily vol) generates signals on nearly every day while JNJ (0.5% daily vol) generates very few.
**Right**: `sig_threshold = max(0.3 * vol_20d_pct, 0.3)`. The threshold scales with each asset's own volatility so signal frequency is consistent across tickers.

### Feature multicollinearity in tree models (max-2-per-family rule)
**Wrong**: including RSI_7, RSI_14, RSI_21 in the same model. All three carry essentially the same signal -- fast/slow RSI. The tree wastes splits arbitrating between them.
**Right**: pick at most 2 periods per indicator family. For RSI: 7 (fast) and 14 (standard). Drop RSI_21. Same logic applies to ATR (only ATR_14), CCI (only CCI_14), Volatility (5d and 20d only). Also drop indicators that are conceptual duplicates of others already present: WILLR (same idea as STOCH), ROC and MOM (same idea as Price_change). See Section 3.9 for the full rationale.

---

## 22. What to Study Next

### Immediate (improves this project directly)

1. **Backtesting frameworks** -- `backtrader` or `vectorbt`. Simulate placing real trades and track P&L, drawdown, and Sharpe over history.

2. **Walk-forward hyperparameter optimisation** -- tune hyperparameters where the search window also advances forward, preventing overfit of params to a specific period.

3. **Kelly criterion** -- optimal position sizing: `f = (bp - q) / b` where b=odds, p=win probability, q=1-p. Use fractional Kelly (e.g. half-Kelly) to limit ruin risk.

### Intermediate

4. **Transformer in depth** -- "Attention Is All You Need" (Vaswani et al., 2017). Original paper is 15 pages. Covers positional encoding, full encoder-decoder, why multi-head attention works.

5. **Stable-Baselines3** -- production-quality RL library. Replaces the hand-rolled PPO here with a well-tested, GPU-accelerated version. Would improve the agent's performance significantly with more data.

6. **N-BEATS and N-HiTS** -- pure MLP forecasting models that rival LSTM/Transformer on time series benchmarks without recurrence. No sequential state, fully parallelisable.

### Advanced

7. **Diffusion models for time series** -- generate realistic future price scenarios (better Monte Carlo), conditioning on current indicators.

8. **Causality** -- Granger causality, PCMCI. Does indicator X actually cause price movement, or merely correlate? The foundational open question in quantitative finance.

### Papers worth reading

- Lim et al. (2021) -- "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
- Hochreiter & Schmidhuber (1997) -- "Long Short-Term Memory"
- Chen & Guestrin (2016) -- "XGBoost: A Scalable Tree Boosting System"
- Ke et al. (2017) -- "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
- Schulman et al. (2017) -- "Proximal Policy Optimization Algorithms" (the PPO paper)
- Dauphin et al. (2017) -- "Language Modeling with Gated Convolutional Networks" (GLU paper)

---

*The best way to learn this is to change one component at a time and observe the effect on val_loss and direction accuracy. Start with the lookback window (try 30, 60, 90) and the CNN filter count. Then try replacing the simple PPO implementation with stable-baselines3 and compare win rates.*
