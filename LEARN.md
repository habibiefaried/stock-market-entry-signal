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
23. [FAQ — Design Decisions & Lessons Learned](#23-faq--design-decisions--lessons-learned)

---

> **Note on Sections 6-11:** The standalone CNN-LSTM and CNN-TFT deep learning
> models were removed from the pipeline after extensive testing showed they
> underperformed tree-based ensembles on this dataset (~45.8% direction accuracy
> vs 50-58% for trees). These sections are kept for educational value and
> because the LSTM architecture is still used inside `train_catboost_bayes.py`
> as a feature generator (see Section 23 FAQ). Keras is no longer a required
> dependency; the only Keras usage is an optional import inside the BO-CatBoost
> script.

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

The system trains 7 models (XGBoost, XGBoost-Heavy, LightGBM, LightGBM-Heavy, RandomForest, RandomForest-Heavy, CatBoost-Bayes), aggregates their opinions, then passes everything to a PPO reinforcement learning agent that outputs the final trade recommendation.

The deep learning models (CNN-LSTM, CNN-TFT) were removed from the pipeline
after extensive testing showed they underperformed tree-based models on this
problem (~45.8% direction accuracy vs 50-58% for trees). The LSTM architecture
lives on inside `train_catboost_bayes.py` as a feature generator (Section 23 FAQ).

---

## 2. Time Series Fundamentals

### 2.1 What makes time series different from tabular data

In standard supervised learning, rows are i.i.d. (independently and identically distributed). You can shuffle rows and the model still learns correctly.

In a time series, row t depends on row t-1. Shuffling destroys the signal. This has cascading consequences:

- **No random train/test split.** You must use a chronological split: train on the past, test on the future. This code uses a 90/10 chronological split.
- **No cross-validation in the usual sense.** You need walk-forward validation (section 5.3).
- **Lag features.** Tree models cannot see sequences; you give them the past explicitly by creating `Close_lag_1`, `Close_lag_2`, etc.
- **Look-ahead bias.** A lethal mistake: if any future information leaks into your training features, your model looks good in backtesting but fails in live trading. This code always shifts targets forward with `shift(-3)` (3-day horizon) and scales only on training data.

### 2.2 The target

All models predict the **next-day return** (percentage). Direction (BUY/SELL/HOLD)
is derived from the predicted return:

```
Target = pct_change().shift(-1) * 100    # (Close[t+1] - Close[t]) / Close[t] * 100
if predicted_return > threshold   ->  BUY (LONG)
if predicted_return < -threshold  ->  SHORT (SELL)
else                              ->  HOLD
```

1-day prediction was chosen over multi-day (3-day, 5-day tested) because:
- Daily error doesn't compound across multiple candles
- Faster feedback loop for live trading
- TP/SL at 1.0/1.5 ATR are achievable in a single session

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

You have features **X** and a target **y**. Fit `f(X) ~= y`. Here y is the 3-day forward return (%).

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

**XGBoost**: adds second-order (Hessian) information, L1/L2 regularisation, level-wise tree growth. Light model: `n_estimators=2000`, `learning_rate=0.01`. Heavy model: `n_estimators=5000`, `learning_rate=0.005`, `max_depth=8`.

**LightGBM**: leaf-wise growth (always split the leaf with highest loss reduction), GOSS sampling, histogram-based splits. Faster than XGBoost for the same n_estimators. Light model: `n_estimators=2000`. Heavy: `n_estimators=5000`, `num_leaves=63`.

**Why 5000 trees + lr=0.005 in heavy models**: halving the learning rate requires doubling n_estimators to fit the same signal -- but the resulting function is smoother and less overfit. `EarlyStopping(rounds=50)` prevents wasted compute.

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
- >= 65%: TAKE TRADE (>= 65% minimum threshold)
- >= 80%: HIGH confidence
- 70-80%: MEDIUM confidence
- < 70%: LOW confidence

---

## 13. Risk Management Mathematics

### 13.1 Stop Loss and Take Profit

```python
# ATR-based TP/SL (consistent across all model scripts + RL environment)
atr = ATR_14  # Average True Range, 14-period
stop_loss   = entry_price ± 1.0 * atr   # tight SL: cut losses fast on wrong calls
take_profit = entry_price ∓ 1.5 * atr   # achievable in a single session
```

Risk/reward = 1.5 / 1.0 = 1.5:1. Break-even win rate = 1.0 / (1.0 + 1.5) = 40%.
With the multi-tier consensus filter (Section 18.10) and regime filter (Section 18.9),
the system achieves 45% avg winrate with 2.86 profit factor across 10 stocks.

**Why 1.0 ATR for SL**: For 1-day predictions, you want to cut losses fast on
wrong calls. 1.0 ATR gives trades breathing room from normal noise while keeping
the stop tight enough for daily trading.

**Why 1.5 ATR for TP**: Achievable in a single session. TP must be > SL
(non-negotiable for positive risk/reward). 1.5:1 ratio means a winning trade
pays for 1.5 losing trades.

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

Each of the 7 models produces a signal (BUY/SHORT/HOLD) and a TP win probability. A human trader would look at all 7 AND the charts to decide whether to trade. The PPO agent learns to do this automatically -- it discovers which combinations of model signals, chart indicators, and market regime actually lead to profitable trades.

### 18.2 State, Action, Reward

**State vector (33 dimensions):**
```
7 model signals       xgboost, xgboost_heavy, lightgbm, lightgbm_heavy,
                      randomforest, randomforest_heavy, catboost_bayes
                      (LONG=1, SHORT=-1, HOLD=0)
7 model probs         (0..1, per-trade direction probability)
RSI_14 normalised     (-1..1 mapped from 0..100)
RSI_7  normalised     (-1..1, shorter-term momentum)
Trend                 (Close - SMA20) / SMA20  ratio
Volatility normalised (capped at 5% daily)
ATR by price          (ATR_14 / Close)
MACD histogram        (normalised by price, clipped)
Bollinger %B          (centred at 0, range ~-1..1)
Stochastic %K         (-1..1 normalised)
Volume ratio          (current vol / 20-day avg, centred at 0)
SMA50 distance        (Close vs SMA50 ratio)
Market regime         (1=BULL, -1=BEAR, 0=RANGING)
ADX_14                (trend strength 0-100, normalised to 0-1)
Choppiness_14         (ranging indicator 0-100, normalised to 0-1)
Awesome Oscillator    (momentum, normalised by price)
DPO_20                (detrended price oscillator, normalised)
Model agreement       (std of model signals)
Avg model confidence  (mean of probs)
Signal consensus      (majority vote direction)
High confidence count (fraction of models with prob > 0.7)
```

The agent sees both **model opinions** (the first 14 dims) and **raw chart context**
(the next 15 dims) -- just like a trader watching both analyst reports and the charts.

**Actions:**
- `LONG (0)`: go long, hold until TP or SL
- `SHORT (1)`: go short, hold until TP or SL
- `HOLD (2)`: skip this trade opportunity

**Rewards:**
```
TP hit first  -> +1.5    (matches 1.5×ATR take-profit)
SL hit first  -> -1.0    (matches 1.0×ATR stop-loss)
Each day held ->  0.0    (no holding penalty)
Timeout       -> -0.05   (tiny nudge: prefer trades that resolve)
Correct dir   -> +0.2    (bonus for matching model consensus)
Counter-regime -> -0.5 penalty (Section 18.9)
Regime-aligned -> +0.3 bonus
Max 15 days   (1-day prediction horizon with generous timeout)
```

**TP/SL levels** (consistent across all model scripts and the RL environment):
```
Stop Loss   = entry ± 1.0 * ATR_14
Take Profit = entry ∓ 1.5 * ATR_14
Risk/Reward = 1.5:1
Break-even  = 40%
```
Using ATR_14 instead of rolling return-std captures intraday gap risk that return-std misses.

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

**This implementation:** Auto-detects PyTorch at import time. If available, uses a 2-layer MLP (128 hidden) with LayerNorm + Dropout, Adam optimiser, proper backpropagation, and gradient clipping. Falls back to a pure numpy MLP (64 hidden) with analytical gradients if PyTorch is not installed. The PyTorch version trains ~3x faster and achieves better convergence due to proper autograd.

**Training scale:** Up to 80k episodes (PyTorch) or 60k (NumPy), batch_size=128,
15 PPO update epochs per batch (PyTorch, 6 for NumPy). The agent warm-starts from
saved weights when re-running on the same CSV file.

### 18.5 Performance Targets

| Metric | Target | Meaning |
|--------|--------|---------|
| Win Rate | >= 36% | Above break-even (40%) is hard with current model accuracy (~55%); profit factor matters more |
| Profit Factor | >= 1.5 | Gross profit / gross loss (> 1.0 = profitable) |
| Sharpe Ratio | >= 1.0 | Risk-adjusted return (annualised) |
| Max Drawdown | > -20% | Worst peak-to-trough (note: can read -100% when equity starts near 0) |
| Trades/Month | >= 10 | Enough activity for swing trading |

**Current TP/SL**: TP = 1.5 x ATR, SL = 1.0 x ATR. Ratio = 1.5:1.
Break-even = 1.0 / (1.0 + 1.5) = 40%.

**Current performance (10 target stocks, 1-day horizon, TP=1.5/SL=1.0)**:
Avg winrate 45.0%, avg profit factor 2.86. All stocks profitable. WMT leads at
55.0% winrate. Winrate ceiling varies by stock (34-55%) based on model accuracy
on each ticker.

### 18.6 Model Predictions for RL Agent

All 7 models save their predictions as `.pkl` files (plus scaler and
feature-name text files). The RL agent's `load_model_predictions()` reads
these files at startup and runs them in walk-forward fashion on the full
dataset to generate out-of-sample signals for each row. If no pkl models
are found, `_synthetic_signals()` generates proxy signals from raw indicators
as a fallback (lower quality, but allows standalone agent testing).

### 18.7 Weight Persistence

After each training run, the PPO weights are saved:
- PyTorch: `rl_agent_torch.pt` + `rl_agent_torch_hash.txt`
- NumPy: `rl_agent_weights.npz` + `rl_agent_csv_hash.txt`

The hash file stores a fingerprint of the CSV (file size + row count). On the next run,
if the fingerprint matches, the agent warm-starts from saved weights. If the CSV
changed (different ticker, date range, or state dimension), training starts fresh.
This means re-runs on the same data are faster and the agent starts from a learned policy.

### 18.8 Fallback Synthetic Signals

If no model `.pkl` files exist (models not yet trained), `_synthetic_signals()`
generates proxy signals for all 7 model slots from the raw indicators (RSI, MACD,
trend) plus random noise. This allows `agent_trader.py` to run standalone for
testing, but the quality will be lower than when real model outputs are used.

### 18.9 Market Regime Filter

The agent includes a hard regime filter that blocks counter-trend trades:

**Regime detection** (`compute_regime()`):
```
Bull score (0-4) based on:
  1 point if price > SMA20
  1 point if price > SMA50
  1 point if SMA50 slope positive (10-day)
  1 point if RSI_14 > 50

BULL   (score >= 3): strong uptrend -- SHORT trades blocked
BEAR   (score <= 1): strong downtrend -- LONG trades blocked
RANGING (score 2):   neutral -- both directions allowed
```

**Effect during training:** Counter-regime actions are converted to HOLD with a
penalty (-0.3), teaching the agent to sit out when the market is trending against
the trade direction.

**Effect during backtest/live:** A hard override flips counter-regime actions to
HOLD before execution. This is the single most impactful improvement in the system:
it cut AAPL backtest trades from 157 to 70 (filtering SHORT signals during a +61%
bull run) while improving profit factor from 1.6 to 2.3.

### 18.10 Winrate Optimization Strategy

**The problem**: With TP=2.05 ATR, SL=1.5 ATR (break-even 42.3%) and model direction
accuracy ~50-55%, the ceiling winrate from consensus alone is ~40-57% depending on
the stock. Getting higher winrate requires stricter trade filtering.

**Strategy — Multi-Tier Consensus Filter**: Instead of a binary trade/skip decision,
trades are graded by how many models agree:

| Agreement | Confidence Scale | Regime Check | Rationale |
|-----------|-----------------|-------------|-----------|
| 5-6/6 | 100% | Always | Near-unanimous, full send |
| 4/6 | 85% | Regime > -0.5 (LONG) or < 0.5 (SHORT) | Good agreement, regime-aware |
| 3/6 | 60% | Regime > 0 (LONG) or < 0 (SHORT) | Must align with trend |
| <3/6 | SKIP | — | Not enough agreement |

This removes "lone wolf" trades and counter-trend trades that killed live performance.
Implemented in both `backtest()` and `get_current_action()`.

**Results across 10 target stocks (1-day horizon, 7 years data)**:

| Stock | Winrate | Profit Factor | Confidence | Notes |
|-------|---------|---------------|------------|-------|
| WMT | 55.0% | 3.33 | 97.9% | Top performer |
| INTC | 51.9% | 3.67 | 80.7% | Strong |
| AMZN | 50.0% | 2.91 | 30% (HOLD) | Models disagree, skip |
| ADBE | 49.5% | 2.75 | 30% (HOLD) | Models disagree, skip |
| MSFT | 45.2% | 3.55 | 91.7% | Solid |
| AMD | 42.1% | 3.24 | 50.5% | Decent |
| NVDA | 40.8% | 1.90 | 90.0% | Profitable |
| META | 36.8% | 2.02 | 95.0% | High conf, lower WR |
| PFE | 34.2% | 2.02 | 30% (HOLD) | Tough stock |

**All 10 stocks profitable** (avg 45.0% winrate, 2.86 profit factor).

---

## 19. System Architecture: How main.py Orchestrates Everything

### 19.1 Parallel Model Training

```python
models = [
    ('XGBoost',              'train_xgboost.py'),
    ('XGBoost-Heavy',        'train_xgboost_heavy.py'),
    ('LightGBM',             'train_lightgbm.py'),
    ('LightGBM-Heavy',       'train_lightgbm_heavy.py'),
    ('RandomForest',         'train_randomforest.py'),
    ('RandomForest-Heavy',   'train_randomforest_heavy.py'),
    ('CatBoost-Bayes',       'train_catboost_bayes.py'),
]

with ThreadPoolExecutor(max_workers=7) as executor:
    ...
```

Each model runs as a **subprocess** (`subprocess.run`), not a thread — Python's GIL prevents true CPU parallelism in threads, but subprocesses are independent OS processes. On a GPU machine, all 7 subprocesses share the GPU via CUDA's internal scheduler.

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
| `main.py` | Orchestrator: runs all 7 models in parallel, calls RL agent, generates HTML report | `RESULT-{TICKER}-{DATE}.html` |
| `fetch_stock_data.py` | Standalone data fetcher using yfinance | `{TICKER}_daily_data_{DATE}.csv` |
| `rank_stocks.py` | Batch runner: ranks all tickers in `target_stock.txt` by agent confidence | `stock-ranking-result.txt` |
| `train_xgboost.py` | XGBoost (5 OHLCV features + lags, 2000 trees) | `xgboost_model.pkl` + scaler + features |
| `train_xgboost_heavy.py` | XGBoost with ~50 indicator features + new indicators (ADX, AO, DPO, etc.), 5000 trees | `xgboost_heavy_model.pkl` + scaler + features |
| `train_lightgbm.py` | LightGBM (5 OHLCV features + lags, 2000 trees) | `lightgbm_model.pkl` + scaler + features |
| `train_lightgbm_heavy.py` | LightGBM with ~50 indicator features + new indicators, 5000 trees | `lightgbm_heavy_model.pkl` + scaler + features |
| `train_randomforest.py` | Random Forest (1000 trees, 5-fold walk-forward) | `randomforest_model.pkl` + scaler + features |
| `train_randomforest_heavy.py` | Random Forest-Heavy (1500 trees, depth 20, 50% bootstrap, 7-fold walk-forward) | `randomforest_heavy_model.pkl` + scaler + features |
| `train_catboost_bayes.py` | LSTM feature generator + Bayesian-optimized CatBoost (Sun & Tian 2023) | `catboost_bayes_model.pkl` + scaler + features |
| `agent_trader.py` | PPO RL meta-agent: reads 7 model pkl files, 33-dim state, consensus + regime filters | `rl_agent_torch.pt` or `rl_agent_weights.npz` (warm-start) |
| `trade_probability_analyzer.py` | Three-approach win probability analysis, called by all model scripts | (no file output — returns results) |
| `recount.py` | Live trading prediction: loads saved models from MODELS/, predicts direction + TP/SL at current price | (stdout only) |
| `model_store.py` | Shared helpers for MODELS/ file naming (`<TICKER>_<YYYYMMDD>_<name>.ext`) | (no file output) |

### 20.1 How to Run

```bash
# Install dependencies
pip install -r requirements.txt
# PyTorch is required (RL agent PPO). GPU recommended but not required.

# Fetch data and run all models + RL agent (default: 84 months = 7 years)
python main.py --ticker MSFT
python main.py --ticker MSFT --months 120   # 10 years

# Use existing CSV
python main.py MSFT_daily_data_20260524.csv

# Live trading with current price override
python main.py --ticker AAPL --current-price 312.50

# Run RL agent standalone (needs pkl files already trained)
python agent_trader.py MSFT_daily_data_20260524.csv

# Batch rank all target stocks
python rank_stocks.py
python rank_stocks.py --top 5 --months 84

# Live recount: predict using pre-trained models + current price
python recount.py --ticker MSFT --current-price 441.31
python recount.py --ticker MSFT --current-price 441.31 --leverage 5
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
**Right**: all model scripts and the RL `TradingEnv.step()` use `SL = 1.0 * ATR_14`, `TP = 1.5 * ATR_14`. ATR_14 captures intraday gap risk that return-std misses and is the industry-standard measure for position sizing. The TP/SL multipliers exist in two places in agent_trader.py (TradingEnv.step() and get_current_action()) -- both must match.

### RL agent loads all 7 models via pkl files
**Right**: `load_model_predictions()` loads all 7 `.pkl` model files directly. Each model saves a pkl + scaler + features.txt triplet during training. The RL agent reads these and runs them in walk-forward fashion to generate out-of-sample signals. No special file handling is needed -- every model uses the same pkl interface.

### Static per-model confidence (test accuracy) in RL state
**Wrong**: using `test_acc * 100` as the `prob` fed into the RL state for every single episode. This is a one-time number that never changes -- the agent can't distinguish high-confidence from low-confidence signals.
**Right**: use per-trade direction probabilities derived from each model's predicted return magnitude. Each row in the training data has a different prob, which is what the RL agent needs to learn from.

### Fixed signal threshold ignores asset volatility
**Wrong**: `if expected_move_pct > 0.5:` -- a 0.5% threshold means TSLA (3% daily vol) generates signals on nearly every day while JNJ (0.5% daily vol) generates very few.
**Right**: `sig_threshold = max(0.15 * vol_20d_pct, 0.1)`. The threshold scales with each asset's own volatility so signal frequency is consistent across tickers.

### Feature multicollinearity in tree models (max-2-per-family rule)
**Wrong**: including RSI_7, RSI_14, RSI_21 in the same model. All three carry essentially the same signal -- fast/slow RSI. The tree wastes splits arbitrating between them.
**Right**: pick at most 2 periods per indicator family. For RSI: 7 (fast) and 14 (standard). Drop RSI_21. Same logic applies to ATR (only ATR_14), CCI (only CCI_14), Volatility (5d and 20d only). Also drop indicators that are conceptual duplicates of others already present: WILLR (same idea as STOCH), ROC and MOM (same idea as Price_change). See Section 3.9 for the full rationale.

---

## 22. What to Study Next

### Immediate (improves this project directly)

1. **Market regime detection** -- already implemented (Section 18.9). Extend with ADX, volatility regime (GARCH), or HMM-based regime switching for more nuanced classification beyond the current 4-score heuristic.

2. **Backtesting frameworks** -- `backtrader` or `vectorbt`. Simulate placing real trades and track P&L, drawdown, and Sharpe over history. The current walk-forward backtest in agent_trader.py is a simplified version.

3. **Kelly criterion** -- optimal position sizing: `f = (bp - q) / b` where b=odds, p=win probability, q=1-p. Use fractional Kelly (e.g. half-Kelly) to limit ruin risk. Combined with the regime filter, risk more in BULL+RANGING, less in BEAR.

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

## 23. FAQ — Design Decisions & Lessons Learned

### Why 1-day prediction instead of multi-day?

We tested 3-day and 5-day. Both improved backtest winrate on paper (AAPL hit 60.5%
at 5-day). But live trading showed all red trades. Why? With daily candles predicting
5 days ahead, there are 5 daily noise events between prediction and resolution.
A correct 5-day call can still get stopped out on day 2 by intra-week noise.
**1-day prediction matches 1-day candles** — less error compounding, faster
feedback, SL/TP are achievable in a single session. Live performance improved
after reverting to 1-day.

### Why 7 models instead of more?

The current 7 models are: XGBoost (light + heavy), LightGBM (light + heavy),
RandomForest (light + heavy), and CatBoost-Bayes (hybrid LSTM+BO). We tested
AdaBoost and vanilla CatBoost but they underperformed (AdaBoost with stumps got
0% F1 on financial data; CatBoost had severe bearish bias with 0% UP predictions).
Adding weak models **dilutes consensus**: 4/9 agreement is less meaningful than
4/7. Only add models that meet a minimum accuracy bar (~50%+ direction F1).

### Why were AdaBoost and vanilla CatBoost removed?

They are NOT in the repo as standalone scripts. AdaBoost with decision stumps
(depth=1) was useless on financial data (0% F1). CatBoost had a severe bearish
bias (0% UP predictions). The only CatBoost variant that survived is
`train_catboost_bayes.py` which uses LSTM features + Bayesian optimization.

### Why was KNN removed?

KNN on daily stock returns always predicted ~0% expected move because similar
market setups lead to opposite outcomes (market efficiency). The weighted average
of 50 neighbors' returns converged to zero — honest but useless for trading.

### Why were LSTM and TFT removed?

LSTM accuracy: 45.8% (below coin-flip). TFT accuracy: 45.8% (same). Both were
deep learning models (~100K parameters) trying to learn from ~1700 data points —
massively overparameterized for the available signal. Tree models (XGBoost,
LightGBM, RandomForest) consistently outperformed them at 50-58% accuracy with
far fewer parameters and faster training.

### Why predict returns (%) instead of absolute prices?

Tree models trained on absolute prices learn price levels ("$150 is normal")
instead of direction. AAPL ranged from $50 to $300 in our data — models memorize
the mean. Predicting `pct_change(3).shift(-3)*100` (3-day % return) makes the
target scale-invariant. MAE dropped from $18 to $3 — 6× improvement in price
prediction accuracy.

### Why TP=1.5, SL=1.0? Why must TP > SL?

**TP must be bigger than SL** — non-negotiable for positive risk/reward. TP=1.5,
SL=1.0 gives 1.5:1 ratio (break-even 40%). We tested symmetric (1.0/1.0 = 50%
break-even, no edge) and wider ratios. 1.5/1.0 was chosen because:
- 1.0 ATR SL: tight enough to cut losers fast, wide enough for normal noise
- 1.5 ATR TP: achievable in a single session for most stocks
- 1.5:1 risk/reward: winning trade pays for 1.5 losing trades

### Why can't we hit 60% winrate consistently?

The 7 models have ~50-55% individual direction accuracy. Even with perfect
agreement filtering, the consensus accuracy is limited by correlated model
errors. Getting above ~57% requires individual model accuracy of 58%+, which
is extremely difficult for single-stock prediction. The system compensates
with **asymmetric rewards** — wins pay more than losses cost (1.5:1 ratio)
— so profit factor stays healthy even at moderate winrates. Some stocks hit
55%+ (TSLA: 56.8%), others don't — it depends on how predictable each ticker is.

### How does `--current-price` work?

When trading live, the CSV has yesterday's close but the market has already moved.
`--current-price` overrides the entry/TP/SL calculations with the live price while
model signals still use the CSV's historical features (no look-ahead):

```bash
python main.py --ticker AAPL --current-price 312.50
```

### Consensus filter vs EV filter vs voting fallback — what's the difference?

- **Voting fallback** (removed): When PPO training was poor, the agent used simple
  model voting. This was worse than PPO — confidence was always 50% with split votes.
- **EV filter** (replaced): Only trade when P(win)×1.5 > P(loss)×1.0. Failed
  because model probabilities are uncalibrated (always ~50%).
- **Multi-tier consensus** (current): Grades trades by how many models agree.
  Calibrated by actual winrate per agreement level. Simple, robust, effective.

### Why does LSTM-BO-CatBoost use LSTM as a "feature generator" instead of a predictor?

From Sun & Tian (2023). The LSTM doesn't make the final trade call. It reads
20 days of High/Low/Close and predicts what today's price "should be" based on
the temporal pattern. The difference between prediction and reality (LSTM_dir,
LSTM_High, LSTM_Low, LSTM_Close) becomes 4 extra features for CatBoost. Think
of it as CatBoost getting a second opinion: "based on the last 20 days' pattern,
the close should be $X, but it's actually $Y — that's bullish/bearish." The LSTM
adds pattern-recognition context that raw indicators miss. Unlike our old
standalone LSTM (45.8% accuracy), this LSTM doesn't need to be right — it just
needs to provide useful temporal features for CatBoost to learn from.

### What about AdaBoost and vanilla CatBoost — are they permanently gone?

They were removed from the repo entirely after underperforming. AdaBoost with
decision stumps got 0% F1 on financial data. Vanilla CatBoost had severe bearish
bias (0% UP predictions). The only CatBoost variant that survived is the
LSTM-BO-CatBoost hybrid (`train_catboost_bayes.py`) which uses Bayesian
optimization and LSTM features to overcome these issues. If you want to
experiment, add a new model script that meets the ~50%+ direction F1 bar.

---

*The best way to learn this is to change one component at a time and observe the effect. Start with the consensus threshold (try 3/7, 4/7, 5/7), then try tuning the regime filter thresholds, then try adding a new model type.*
