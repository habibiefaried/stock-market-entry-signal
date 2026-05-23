# Study Guide: Stock Market Entry Signal System

*Written as if your lecturer is sitting across the table from you.*

This document walks through every concept behind this codebase — from raw maths to production code decisions. Read it top to bottom the first time, then use it as a reference. Nothing here is hand-wavy; where there is an equation, it is the real one used in the code.

---

## Table of Contents

1. [The Problem We Are Solving](#1-the-problem-we-are-solving)
2. [Time Series Fundamentals](#2-time-series-fundamentals)
3. [Technical Indicators — The Feature Engineering Layer](#3-technical-indicators--the-feature-engineering-layer)
4. [Machine Learning Foundations](#4-machine-learning-foundations)
5. [Tree-Based Models: XGBoost, LightGBM, RandomForest](#5-tree-based-models-xgboost-lightgbm-randomforest)
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
16. [Evaluation Metrics — What They Actually Mean](#16-evaluation-metrics--what-they-actually-mean)
17. [GPU Acceleration and Fallback Strategy](#17-gpu-acceleration-and-fallback-strategy)
18. [System Architecture: How main.py Orchestrates Everything](#18-system-architecture-how-mainpy-orchestrates-everything)
19. [Common Pitfalls and How This Code Avoids Them](#19-common-pitfalls-and-how-this-code-avoids-them)
20. [What to Study Next](#20-what-to-study-next)

---

## 1. The Problem We Are Solving

You are given a time series of daily stock prices (Open, High, Low, Close, Volume — OHLCV). You want to answer one question:

> **"If I enter a trade right now with a given Stop Loss and Take Profit, what is the probability that price hits TP before SL within the next 5 days?"**

This is not a trivial regression problem. It requires:

- **Feature engineering** — transforming raw prices into signals a model can learn from
- **Sequence modelling** — prices have memory; yesterday affects today
- **Probabilistic output** — a single price prediction is not enough; you need a distribution over outcomes
- **Ensemble reasoning** — no single model is reliable; combine multiple approaches

The system trains five models (CNN-LSTM, CNN-TFT, XGBoost, LightGBM, RandomForest), each answering the same question from a different angle, then aggregates their opinions into one trading recommendation.

---

## 2. Time Series Fundamentals

### 2.1 What makes time series different from tabular data

In standard supervised learning, rows are i.i.d. (independently and identically distributed). You can shuffle rows and the model still learns correctly.

In a time series, row *t* depends on row *t-1*. Shuffling destroys the signal. This has cascading consequences:

- **No random train/test split.** You must use a chronological split: train on the past, test on the future. This code uses a 90/10 chronological split.
- **No cross-validation in the usual sense.** You need walk-forward validation (explained in section 5.3).
- **Lag features.** Tree models cannot see sequences; you give them the past explicitly by creating `Close_lag_1`, `Close_lag_2`, etc.
- **Look-ahead bias.** A lethal mistake: if any future information leaks into your training features, your model looks good in backtesting but fails in live trading. This code always shifts targets forward with `shift(-1)` and scales only on training data.

### 2.2 Stationarity

A stationary series has constant mean and variance over time. Raw prices are non-stationary (they trend). Returns (`pct_change`) are approximately stationary. Many indicators (RSI, MACD, BB %B) are engineered to be bounded or zero-centred, making them more stationary and easier for models to learn from.

### 2.3 The target

All five models predict **tomorrow's closing price** (regression). Direction (BUY/SELL/HOLD) is derived from the predicted price vs today's price, with a 0.5% threshold to filter noise:

```
if (predicted - today) / today > 0.005  →  BUY
if (predicted - today) / today < -0.005 →  SHORT
else                                     →  HOLD
```

---

## 3. Technical Indicators — The Feature Engineering Layer

This is where domain knowledge enters. Raw OHLCV data carries limited signal; technical indicators re-express the same data in forms that highlight momentum, trend, volatility, and volume pressure. All 52 features in `FEATURE_COLS` are computed in `compute_technical_indicators()`.

### 3.1 Moving Averages

**Simple Moving Average (SMA):**
```
SMA_p(t) = (1/p) * Σ Close(t-i)  for i = 0..p-1
```
Smooths noise. Slow to react. Used for trend identification. This code computes SMA for periods 5, 10, 20, 50, 100, 200.

**Exponential Moving Average (EMA):**
```
EMA(t) = Close(t) * k + EMA(t-1) * (1-k)    where k = 2 / (p+1)
```
Gives more weight to recent prices. Faster than SMA. Used for MACD, and standalone for periods 9, 21, 50, 100.

**MA Ratios:**
```
Close_SMA20_ratio = (Close - SMA_20) / SMA_20
```
Normalises price distance from its moving average. A value of +0.05 means price is 5% above the 20-day average — a useful dimensionless feature.

### 3.2 RSI — Relative Strength Index

```
RS = EMA(gains, p) / EMA(losses, p)
RSI = 100 - (100 / (1 + RS))
```

RSI is bounded [0, 100]. Above 70: overbought (potential reversal down). Below 30: oversold (potential reversal up). The code uses the Wilder EMA variant (`ewm(com=p-1)`). Three periods are computed: 7 (short-term), 14 (standard), 21 (longer-term).

Key insight: RSI does not predict direction by itself. It tells you *how extreme* the current momentum is relative to recent history, which helps the CNN extract divergence patterns.

### 3.3 MACD — Moving Average Convergence Divergence

```
MACD_line   = EMA(12) - EMA(26)
MACD_signal = EMA(MACD_line, 9)
MACD_hist   = MACD_line - MACD_signal
```

Three features from one calculation. The histogram measures acceleration of the trend. When `MACD_hist` crosses zero, it signals a potential trend change. The CNN can learn to detect these crossovers across the time axis.

### 3.4 Bollinger Bands

```
BB_mid   = SMA(20)
BB_upper = BB_mid + 2 * std(20)
BB_lower = BB_mid - 2 * std(20)
BB_pct   = (Close - BB_lower) / (BB_upper - BB_lower)    ∈ [0, 1] approximately
BB_width = (BB_upper - BB_lower) / BB_mid
```

BB_pct tells you where the price sits within its recent volatility range. BB_width measures volatility itself — a squeeze (narrow bands) often precedes a large move. Five features here give the model rich information about the volatility regime.

### 3.5 ATR — Average True Range

```
TR(t) = max(High-Low, |High-Close(t-1)|, |Low-Close(t-1)|)
ATR_p  = EMA(TR, p)
```

ATR measures volatility in price units (dollars), not percentage. Used in this code for two things: (1) as a feature for the model, and (2) as the denominator in HL_vs_ATR14 to normalise daily range against recent volatility.

### 3.6 Stochastic Oscillator

```
%K = 100 * (Close - Low_14) / (High_14 - Low_14)
%D = SMA(%K, 3)
```

Where High_14 and Low_14 are the 14-period rolling high and low. Like RSI, bounded [0, 100]. Captures where the close sits within the recent price range. The %D line smooths %K.

### 3.7 OBV — On-Balance Volume

```
OBV(t) = OBV(t-1) + sign(Close(t) - Close(t-1)) * Volume(t)
```

Cumulates volume with a +1/-1 direction multiplier. The idea: volume drives price. If price is rising on high volume, the trend is strong. The code log-scales OBV because it grows cumulatively and can become very large.

### 3.8 CCI — Commodity Channel Index

```
TP  = (High + Low + Close) / 3
CCI = (TP - SMA(TP, p)) / (0.015 * MAD(TP, p))
```

Where MAD is the mean absolute deviation. CCI is zero-centred and unbounded. Values above +100 or below -100 indicate extreme conditions. Two periods: 14, 20.

### 3.9 Williams %R

```
WILLR = -100 * (High_14 - Close) / (High_14 - Low_14)
```

Range: [-100, 0]. Values near 0 mean price is near its 14-day high (potentially overbought). Values near -100 mean near its low. Essentially the inverse of %K stochastic.

### 3.10 Rate of Change (ROC) and Momentum

```
ROC_p   = (Close(t) - Close(t-p)) / Close(t-p) * 100
MOM_p   = Close(t) - Close(t-p)
```

ROC is percentage momentum; MOM is absolute momentum. Both quantify how fast prices are moving. The CNN can detect when momentum is accelerating or decelerating.

### 3.11 Volume Features

```
Volume_log        = log1p(Volume)
Volume_MA20_ratio = Volume / rolling_mean(Volume, 20)
```

Log-volume compresses the heavy tail. The ratio tells you whether today's volume is above or below the recent average — a spike in volume on a price breakout is highly significant.

### 3.12 Price Changes and Volatility

```
Price_change_1d  = pct_change(1) * 100
Price_change_3d  = pct_change(3) * 100
Price_change_5d  = pct_change(5) * 100
Volatility_5d    = rolling_std(returns, 5) * 100
Volatility_10d   = rolling_std(returns, 10) * 100
Volatility_20d   = rolling_std(returns, 20) * 100
HL_range_pct     = (High - Low) / Close * 100
HL_vs_ATR14      = (High - Low) / ATR_14
```

These capture short-term momentum and the current volatility regime. `HL_vs_ATR14` is particularly useful: when the daily range is much larger than the average, something unusual is happening.

---

## 4. Machine Learning Foundations

### 4.1 Supervised Learning

You have input features **X** and a target **y**. You fit a function `f(X) ≈ y`. In this project, `y` is tomorrow's close price and **X** is a vector (GBM models) or a sequence matrix (LSTM/TFT models).

### 4.2 Train/Test Split

The training set is used to fit model parameters. The test set is held out and used only to evaluate. If you evaluate on training data, you measure how well the model memorised the data, not how well it generalises.

In this code: 90% of chronological data is training, 10% is test. This is not random.

### 4.3 Overfitting vs Underfitting

- **Overfitting**: model memorises training data, fails on test data. Train loss << Test loss.
- **Underfitting**: model is too simple to capture patterns. Both losses are high.

Regularisation techniques used in this code:
- Dropout (LSTM/TFT): randomly zeros activations during training
- EarlyStopping: stops training when validation loss stops improving
- ReduceLROnPlateau: reduces learning rate when progress stalls
- max_depth / num_leaves (GBM): limits tree complexity
- min_data_in_leaf (LightGBM): prevents leaves fitted on very few samples

### 4.4 Bias-Variance Tradeoff

Every model makes a tradeoff:
- High bias (underfitting): model assumptions are too simple
- High variance (overfitting): model is too sensitive to training data noise

Ensemble methods (RandomForest, XGBoost, LightGBM) reduce variance by averaging many models. Dropout in neural networks adds noise during training, forcing the network to learn robust, distributed representations.

---

## 5. Tree-Based Models: XGBoost, LightGBM, RandomForest

### 5.1 Decision Trees

A decision tree splits the feature space with axis-aligned cuts:
```
if Close_lag_1 > 150.0:
    if RSI_14 > 70:
        predict 148.0
    else:
        predict 153.0
else:
    predict 145.0
```

Each internal node is a feature + threshold. Each leaf is a prediction. Trees are fast to predict, interpretable, but overfit easily if deep.

### 5.2 Gradient Boosting (XGBoost, LightGBM)

Gradient boosting builds an ensemble of weak trees **sequentially**. Each new tree fits the *residuals* (errors) of all previous trees:

```
F_0(x) = initial prediction (e.g. mean of y)
F_m(x) = F_{m-1}(x) + η * h_m(x)
```

Where `h_m` is the m-th tree fitted to the negative gradient of the loss, and `η` is the learning rate.

For MSE loss, the negative gradient is simply the residual `y - F_{m-1}(x)`. So each tree literally corrects what the previous ensemble got wrong.

**XGBoost** adds second-order (Hessian) information and L1/L2 regularisation. It uses level-wise tree growth (all nodes at depth d are split before depth d+1).

**LightGBM** uses leaf-wise growth: always split the leaf with the highest loss reduction, regardless of depth. This finds better splits faster but can overfit without `min_data_in_leaf`. LightGBM also uses histogram-based algorithms for speed and gradient-based one-side sampling (GOSS) to further reduce computation.

**Key hyperparameters in this code:**
- `n_estimators`: number of trees. More = potentially better, but slower and more overfit risk. EarlyStopping mitigates this.
- `learning_rate`: how much each tree contributes. Lower = more trees needed but more robust.
- `num_leaves` (LightGBM): maximum leaves per tree. Higher = more expressive.
- `max_depth` (XGBoost): maximum tree depth.
- `subsample`, `colsample_bytree`: fraction of rows/columns to sample per tree — like random forests, this adds diversity.

### 5.3 Random Forest and Walk-Forward Validation

Random Forest trains many trees **in parallel** (not sequentially). Each tree is trained on a bootstrapped sample (random rows with replacement) and uses only `sqrt(n_features)` randomly chosen features per split. The final prediction is the average across all trees.

Random Forest is more robust than a single tree and parallelises perfectly, but it cannot extrapolate beyond the training range — if test prices are higher than any price seen in training, the model predicts the maximum training price.

**Walk-Forward Validation** (used only for RandomForest in this code):

Instead of a single train/test split, roll a training window forward:
```
Fold 1: train [0, 700],   test [700, 760]
Fold 2: train [0, 760],   test [760, 820]
Fold 3: train [0, 820],   test [820, 880]
Fold 4: train [0, 880],   test [880, 940]
Fold 5: train [0, 940],   test [940, 1000]
```
Each fold simulates live trading. The final reported metrics are aggregated across all folds.

### 5.4 Lag Features — Why Tree Models Need Them

Trees operate on flat feature vectors, not sequences. To give them temporal context, the code creates:
```
Close_lag_1  = yesterday's close
Close_lag_2  = 2 days ago close
...
Close_lag_10 = 10 days ago close
Volume_lag_1..10
Price_change_1d, 5d, 10d
Volatility_5d, 10d
```

This turns the time series prediction into a standard tabular regression. The model learns patterns like "when price fell for 3 consecutive days and RSI is below 30, it tends to bounce."

---

## 6. Neural Networks from First Principles

### 6.1 The Neuron

A single neuron computes:
```
output = activation(W · x + b)
```
Where `W` are learnable weights, `x` is the input vector, `b` is a bias scalar, and `activation` is a non-linear function.

Without activation functions, stacking layers is equivalent to a single linear transformation — no expressive power. Non-linearities let networks approximate any function (Universal Approximation Theorem).

**Activation functions used in this code:**
- `ReLU(x) = max(0, x)` — default for CNN layers. Fast, no vanishing gradient for positive inputs.
- `ELU(x) = x if x>0, else α*(e^x - 1)` — used in GRN dense layers. Smooth for negative inputs, helps gradient flow.
- `tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})` — used in LSTM gates and TemporalAttention. Output ∈ (-1, 1).
- `sigmoid(x) = 1 / (1 + e^{-x})` — used in LSTM gates and TFT gating. Output ∈ (0, 1), ideal for "how much to let through".

### 6.2 Dense (Fully Connected) Layer

Every input is connected to every output:
```python
Dense(32, activation='relu')
```
This creates a weight matrix of shape `(input_dim, 32)`. The output is `relu(X @ W + b)`.

The Dense head at the end of both CNN-LSTM and CNN-TFT:
```python
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
out = Dense(1)(x)  # Linear output — predicts price
```
The final Dense(1) has no activation because we are doing regression (predicting a continuous value).

### 6.3 Backpropagation

Training works by:
1. Forward pass: compute prediction from input
2. Compute loss: measure error between prediction and true target
3. Backward pass: use chain rule to compute gradient of loss w.r.t. every weight
4. Update: `W ← W - η * ∇W`

The chain rule for a layer is:
```
∂L/∂W_i = (∂L/∂output) * (∂output/∂W_i)
```

Deeper networks multiply many gradients together. If those gradients are < 1, they vanish (deep layers learn nothing). If > 1, they explode (training becomes unstable). LSTM, BatchNorm, and residual connections all exist to solve this problem.

### 6.4 Batch Normalisation

After a layer, normalise the activations across the batch:
```
x_norm = (x - mean(x)) / (std(x) + ε)
output = γ * x_norm + β
```
Where `γ` and `β` are learnable. This keeps activations in a healthy range throughout training, dramatically stabilises and speeds up training of deep networks.

### 6.5 Dropout

During training, randomly set a fraction `p` of activations to zero:
```python
Dropout(0.25)  # 25% of activations zeroed at each step
```
This forces the network to not rely on any single neuron — each path must carry the signal independently. At inference time, dropout is disabled and activations are scaled by `(1-p)`.

---

## 7. Convolutional Neural Networks (CNN-1D)

### 7.1 Why Convolution for Time Series?

A 1D convolution slides a small filter (kernel) across the time axis, computing a dot product at each position. Given input `x` of shape `(timesteps, features)` and a filter `w` of shape `(kernel_size, features)`:

```
output(t) = Σ_k Σ_f  x(t+k, f) * w(k, f)    + bias
```

Key properties:
- **Local receptive field**: each output sees only `kernel_size` consecutive timesteps — captures local patterns (crossovers, divergences)
- **Weight sharing**: the same filter is applied at every timestep — the model learns "what does RSI divergence look like" once, not separately for each timestep
- **Translation invariance**: the pattern is detected wherever it appears in the sequence

### 7.2 The Three CNN Blocks in This Code

```python
# Block 1: narrow filter, detect fine-grained patterns
Conv1D(64, kernel=3, activation='relu', padding='same')
BatchNormalization()

# Block 2: wider filter, detect broader patterns (e.g. multi-day formations)
Conv1D(256, kernel=5, activation='relu', padding='same')
BatchNormalization()
MaxPooling1D(pool_size=2)   # ← halves the time dimension

# Block 3: narrow filter again, refine features
Conv1D(64, kernel=3, activation='relu', padding='same')
BatchNormalization()
Dropout(0.25)
```

`padding='same'` keeps the output length equal to the input length. Without padding, convolution shrinks the sequence by `(kernel_size - 1)`.

`MaxPooling1D(2)` takes the maximum over each non-overlapping window of 2. This halves the time dimension, making the sequence shorter (fewer timesteps for LSTM to process) while keeping the most prominent features. It also adds a small degree of translation invariance.

256 filters in block 2 means the model learns 256 different pattern detectors simultaneously. Each filter specialises in a different cross-indicator pattern.

### 7.3 What CNN Learns vs What LSTM Learns

| | CNN | LSTM |
|---|---|---|
| **Scope** | Short-range (3-5 timesteps) | Long-range (full lookback window) |
| **What it detects** | Local patterns (crossovers, divergences at a moment in time) | How patterns evolve over weeks |
| **Analogy** | "Is RSI crossing 30 right now?" | "Has RSI been recovering from oversold for 2 weeks?" |

By stacking them, the CNN extracts *what the indicators are doing now*, and LSTM tracks *how that has been evolving over time*.

---

## 8. Recurrent Networks and LSTM

### 8.1 The Vanishing Gradient Problem in Simple RNNs

A simple RNN computes:
```
h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)
```

To train, gradients flow backwards through time (BPTT). The gradient of the loss with respect to `h_{t-k}` involves multiplying `W_h` by itself k times. If the largest eigenvalue of `W_h` is < 1, the gradient vanishes exponentially. If > 1, it explodes. Either way, simple RNNs cannot learn dependencies beyond ~10 timesteps.

### 8.2 LSTM Architecture

LSTM (Long Short-Term Memory) solves this with a gated architecture. At each timestep it maintains two vectors:
- `h_t`: the hidden state (short-term memory, output)
- `c_t`: the cell state (long-term memory, flows mostly unchanged)

The four gates:
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)   ← forget gate: how much of c_{t-1} to keep
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)   ← input gate: how much new info to write
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g) ← candidate cell update
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)   ← output gate: what to expose as h_t

c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
```

`⊙` is element-wise multiplication. The cell state `c_t` flows through the network with only element-wise operations (no matrix multiply), creating a gradient highway. This is why LSTMs can learn dependencies spanning 50-200 timesteps.

### 8.3 Stacked LSTM with return_sequences

```python
x = LSTM(128, return_sequences=True)(x)   # output: (batch, timesteps, 128)
x = Dropout(0.3)(x)
x = LSTM(64, return_sequences=True)(x)    # output: (batch, timesteps, 64)
```

`return_sequences=True` means the LSTM outputs `h_t` at every timestep, not just the last one. This is required because:
1. The second LSTM needs a full sequence as input
2. The Temporal Attention layer needs all timesteps to compute weights over them

Without `return_sequences=True`, the LSTM would only output the final `h_T`, discarding all intermediate states.

### 8.4 Why Two LSTM Layers?

The first LSTM (128 units) learns broad temporal patterns. The second LSTM (64 units) refines those patterns, building higher-level abstractions on top of the first layer's output. This is analogous to why deep CNNs work better than shallow ones: each layer learns increasingly abstract representations.

A third LSTM was considered and rejected: with ~900 training sequences (from 4 years of daily data minus warmup), the risk of overfitting outweighs the capacity gain.

---

## 9. Attention Mechanisms

### 9.1 Why Attention?

Not every day in the 60-day lookback window is equally important for predicting tomorrow. A significant breakout 3 weeks ago might be far more predictive than yesterday's sideways movement. A fixed LSTM hidden state gives equal "structural" weight to all timesteps (though it can learn to emphasise some). Explicit attention lets the model learn a soft weight for each timestep.

### 9.2 Temporal Attention (used in CNN-LSTM)

```python
class TemporalAttention(Layer):
    def build(self, input_shape):
        units = input_shape[-1]  # = lstm2_units = 64
        self.W = self.add_weight(shape=(units, 1), ...)
        self.b = self.add_weight(shape=(1,), ...)

    def call(self, inputs):
        # inputs: (batch, timesteps, 64)
        score = tanh(inputs @ W + b)       # (batch, timesteps, 1)
        score = squeeze(score, axis=-1)     # (batch, timesteps)
        weights = softmax(score)            # (batch, timesteps)  — sum to 1
        weights = expand_dims(weights, -1)  # (batch, timesteps, 1)
        context = sum(inputs * weights, axis=1)  # (batch, 64)
        return context
```

The weight vector `W` (64×1) is the only learned parameter. It maps each 64-dim hidden state to a scalar score. After softmax, the scores become a probability distribution over timesteps. The context vector is the weighted average of hidden states.

Mathematically:
```
e_t  = tanh(W^T h_t + b)       — unnormalised importance of timestep t
a_t  = softmax(e)_t             — normalised attention weight
c    = Σ_t  a_t * h_t           — context vector
```

This is **Bahdanau-style** (additive) attention simplified to a single query.

### 9.3 Multi-Head Self-Attention (used in CNN-TFT)

Multi-head attention computes multiple attention patterns in parallel:

```
Q = X @ W_Q,  K = X @ W_K,  V = X @ W_V

Attention(Q, K, V) = softmax( Q @ K^T / sqrt(d_k) ) @ V
```

Where `d_k` is the key dimension (`d_model / n_heads = 64/4 = 16`). The `sqrt(d_k)` scaling prevents the dot products from growing so large that softmax saturates.

With 4 heads, each head learns a different attention pattern:
- Head 1 might focus on weekly periodicity
- Head 2 might focus on high-volatility days
- Head 3 might focus on trend reversals
- Head 4 might focus on the most recent days

All heads run in parallel, their outputs are concatenated and projected back to `d_model`.

**Why self-attention?** The query and key both come from the same sequence (X). This means the model asks "which other timesteps in this window are most relevant to each timestep?" — it learns long-range dependencies regardless of distance, unlike LSTM which must propagate information step by step.

---

## 10. The Temporal Fusion Transformer (TFT)

### 10.1 Gated Linear Unit (GLU)

```python
class GatedLinearUnit(Layer):
    def call(self, x):
        projected = Dense(units * 2)(x)       # project to double size
        x1 = projected[..., :units]           # first half
        x2 = projected[..., units:]           # second half
        return x1 * sigmoid(x2)               # gate: x2 controls how much of x1 flows
```

The sigmoid output acts as a soft gate ∈ (0,1). The network learns to suppress irrelevant information completely (gate → 0) or pass it through fully (gate → 1). This is the fundamental information-routing primitive in TFT.

### 10.2 Gated Residual Network (GRN)

```python
class GatedResidualNetwork(Layer):
    def call(self, x, training=False):
        residual = self.proj(x) if self.proj else x    # linear skip connection
        h = ELU( Dense(units)(x) )                     # first transformation
        h = Dense(units)(h)                             # second transformation
        h = GLU(units)(h)                               # gate: can suppress entire path
        h = Dropout(rate)(h)
        return LayerNorm(residual + h)                  # residual + normalise
```

The GRN is the workhorse of TFT. Key design decisions:

1. **ELU activation**: smooth at zero, no dying neuron problem, allows negative activations
2. **GLU gating**: if the transformation is unhelpful, the gate closes and the residual dominates
3. **Residual connection**: if the whole transformation block is useless, the network can learn to pass the input unchanged (gate → 0, output ≈ residual)
4. **LayerNorm after residual add**: stabilises training without the batch-size dependency of BatchNorm

The `proj` (linear projection) exists only if `input_dim ≠ units` — it aligns dimensions so the residual add is valid.

### 10.3 Full TFT Data Flow

```
Input (batch, lookback, 52 features)
  ↓ CNN block ×3 → extract local cross-indicator patterns
  ↓ Dense(d_model=64) → project CNN output to uniform width
  ↓ GRN(pre-LSTM) → non-linear feature transform per timestep
  ↓ LSTM(64, return_seq=True) → build local temporal context
  ↓ GRN(post-LSTM) + sigmoid gate + skip + LayerNorm
       gate = σ(Dense(d_model)(grn_out))
       x = gate * grn_out + (1-gate) * grn_pre_lstm_input
       x = LayerNorm(x)
  ↓ MultiHeadAttention(4 heads, d_k=16) → long-range dependencies
  ↓ GRN(post-attn) + sigmoid gate + skip + LayerNorm
  ↓ GlobalAveragePooling1D → collapse time axis (mean of all timesteps)
  ↓ Dense(32, relu)
  ↓ Dropout(0.1)
  ↓ Dense(1) → tomorrow's price
```

The gated skip connections around LSTM and attention are critical: if the LSTM output is noisy (early in training), the gate can be near zero and the network effectively bypasses the LSTM. As training progresses, the gate opens as the LSTM becomes useful.

### 10.4 Why VSN is Defined but Not Used

The `VariableSelectionNetwork` class exists in the code but is not wired into `build_cnn_tft_model`. The reason is parameter count. VSN applies one GRN per feature, then another GRN for the selection weights. With 52 features and `d_model=64`:

```
params ≈ 52 * GRN(52→64) + GRN(64→52) ≈ 52 * (52*64 + 64*64) + (64*52 + 52*52)
       ≈ 52 * (3328 + 4096) + (3328 + 2704)
       ≈ 52 * 7424 + 6032
       ≈ 392,480 parameters
```

With only ~900 training sequences, adding ~400K parameters for a feature-selection mechanism that the CNN already handles implicitly would cause severe overfitting. Good engineering is knowing when *not* to add a component.

---

## 11. Keras and the PyTorch Backend

### 11.1 Keras as a Frontend

Keras is a high-level neural network API. Historically it ran on TensorFlow. Since Keras 3.0, it supports multiple backends: TensorFlow, PyTorch, and JAX. This code uses:

```python
os.environ['KERAS_BACKEND'] = 'torch'
```

This must be set **before** importing Keras. The backend determines how tensors are stored and how gradients are computed. PyTorch is used here because it has better native GPU support on some hardware.

### 11.2 Custom Keras Layers

Any class inheriting from `keras.layers.Layer` becomes a Keras layer. Required methods:

```python
def build(self, input_shape):
    # Create weights here. Called once on first forward pass.
    self.W = self.add_weight(name='W', shape=(...), trainable=True)
    super().build(input_shape)

def call(self, inputs, training=False):
    # Forward pass logic. training=True during .fit(), False during .predict().
    ...
    return output

def get_config(self):
    # Required for model serialisation (save/load).
    cfg = super().get_config()
    cfg.update({'my_param': self.my_param})
    return cfg
```

`get_config()` is critical. When you save a model with custom layers and reload it, Keras calls `get_config()` to reconstruct the layer. Without it, `load_model` fails.

### 11.3 custom_objects in load_model

When you call `keras.models.load_model('model.keras')`, Keras reads the config and reconstructs each layer by name. For built-in layers (`Dense`, `LSTM`, etc.) it knows the class. For custom layers, you must provide the mapping:

```python
model = keras.models.load_model(
    'best_lstm_model.keras',
    custom_objects={'TemporalAttention': TemporalAttention}
)
```

Without this, Keras raises `ValueError: Unknown layer: TemporalAttention`. This is one of the most common bugs when deploying custom models.

### 11.4 ModelCheckpoint and EarlyStopping

```python
checkpoint = ModelCheckpoint(
    'best_lstm_model.keras',
    monitor='val_loss',
    save_best_only=True,
)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
)
```

`ModelCheckpoint` saves the model whenever `val_loss` improves. After training, the code explicitly reloads the checkpoint:
```python
model = keras.models.load_model('best_lstm_model.keras', custom_objects={...})
```

Why reload? With `restore_best_weights=True`, EarlyStopping restores weights in memory. But the checkpoint on disk is also the best version. Reloading from disk is belt-and-suspenders — it guarantees the saved file matches the in-memory model used for predictions.

`ReduceLROnPlateau` halves the learning rate after 7 epochs of no val_loss improvement. This allows the optimiser to make finer adjustments when stuck.

---

## 12. Probability Analysis: Three Approaches

### 12.1 Approach 1: Multi-Day Sequential Prediction

The model predicts Day 1. That prediction becomes part of the input for Day 2. And so on for 5 days. At each step, check if the predicted price crossed TP or SL:

```python
for day in range(1, 6):
    pred = model.predict(recent_window)
    if pred >= take_profit:  return {'hit_tp': True, 'hit_day': day}
    if pred <= stop_loss:    return {'hit_tp': False, 'hit_day': day}
    # Append pred to df_sim, recalculate all indicators
    df_sim = recalculate_features(df_sim)
```

When a predicted row is appended, all 52 technical indicators must be recomputed — the CNN/LSTM models need all features, not just Close. The `recalculate_features` function does this from scratch using the (now extended) price history.

**Limitation**: error compounds over 5 days. The Day 2 prediction is based on the (already imperfect) Day 1 prediction. By Day 5, uncertainty is high. This is by design — the result is treated as one of three inputs to the ensemble, not the sole answer.

**Output**: binary (TP or SL hit) → contributes 100% or 0% to the ensemble.

### 12.2 Approach 2: Monte Carlo Simulation

Run 1000 independent random price walks. Each walk uses:
```
daily_return ~ Normal(drift, volatility/current_price)
price(t+1) = price(t) * (1 + daily_return)
```

Where:
```
drift = (predicted_move_pct / 100) * MC_DRIFT_INFLUENCE / n_days
```

`MC_DRIFT_INFLUENCE = 0.3` means the model's predicted direction biases the random walk by 30%, but 70% is pure noise. This reflects the reality that stock price movements are partly random.

The win rate is the fraction of 1000 simulations where price hit TP before SL. With 1000 runs, the standard error is `sqrt(p*(1-p)/1000)` ≈ 1.6% for p=0.5.

**Why this matters**: even if the model predicts a 2% move toward TP, if daily volatility is 2.5%, many random paths will hit SL first. Monte Carlo captures this geometric reality that a deterministic prediction cannot.

### 12.3 Approach 3: Historical Pattern Matching

Find historical dates where market conditions were similar to today:

```python
rsi_diff    = |current_RSI - historical_RSI|       < 5
vol_ratio   = historical_vol / current_vol          ∈ [0.8, 1.2]
trend_same  = (current_5d_return * historical_5d_return) > 0
```

The top 50 most similar dates are found. For each, scale the SL/TP levels proportionally:
```
hist_tp = take_profit / current_price * hist_price
hist_sl = stop_loss   / current_price * hist_price
```

Then check actual historical outcomes: did the price hit hist_tp or hist_sl in the next 5 days? The win rate is `tp_hits / total_matches`.

**Why this matters**: this approach is non-parametric and model-free. It asks "what actually happened in the past when the market looked like this?" It captures regime-specific behaviour that models may have generalised away.

### 12.4 Ensemble Combination

```python
weights = [W_prediction=0.4, W_monte_carlo=0.35, W_pattern=0.25]

ensemble_prob = Σ (prob_i * weight_i) / Σ weight_i
```

Weights are normalised if any approach is unavailable (e.g. insufficient historical data for pattern matching). The ensemble probability drives the TAKE/SKIP recommendation:
- ≥ 60%: TAKE TRADE
- < 60%: SKIP TRADE
- ≥ 75%: HIGH confidence
- 65-75%: MEDIUM confidence
- < 65%: LOW confidence

---

## 13. Risk Management Mathematics

### 13.1 Stop Loss and Take Profit Placement

The code uses volatility-based position sizing. Daily volatility is computed as the standard deviation of returns over the last 20 days, then converted to a dollar amount:

```python
daily_returns = Close.tail(20).pct_change().dropna()
volatility    = daily_returns.std() * today_price   # $ per day
```

Stop loss and take profit distances:
```python
stop_loss_distance   = 0.6 * volatility
take_profit_distance = 1.0 * volatility
```

For a BUY signal:
```
stop_loss   = today_price - 0.6σ
take_profit = today_price + 1.0σ
```

The risk/reward ratio is `1.0σ / 0.6σ = 1.67:1`. This means you need to win more than 37.5% of trades to be profitable at break-even (before costs):
```
P(win) * 1.67 > P(loss) * 1.0
P(win) > 1 / (1 + 1.67) = 0.375
```

The minimum confidence threshold of 60% is set well above this break-even point.

### 13.2 5x Leverage P&L

With 5x leverage on a CFD/options platform (like IQ Option):
```
position_pnl% = stock_move% * 5
```

So a stop loss at -1.5% stock move becomes -7.5% position loss. A take profit at +2.5% stock move becomes +12.5% position gain.

The 5x P&L numbers in the report help you immediately see the real-money impact before entering a trade.

---

## 14. Data Pipeline and Feature Scaling

### 14.1 Why Scale?

Neural networks are sensitive to input scale. If one feature ranges 0-200 (price) and another ranges 0-1 (BB_pct), the gradients will be dominated by the large-scale feature. Scaling puts all features on the same range.

### 14.2 MinMaxScaler (used for LSTM/TFT)

```
x_scaled = (x - x_min) / (x_max - x_min)    → output ∈ [0, 1]
```

Critical rule: **fit only on training data**. The scaler learns `x_min` and `x_max` from training. Test data is transformed using the same parameters:

```python
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)   # learns min/max
X_test_scaled  = scaler_X.transform(X_test)         # applies same min/max
```

If you fit on the full dataset, the test set's future prices leak into the scaler parameters — a subtle form of look-ahead bias.

### 14.3 Separate Scalers for Features and Target

```python
scaler_X = MinMaxScaler()   # scales the 52-feature matrix
scaler_y = MinMaxScaler()   # scales only the Close price target
```

The target (Close price) needs its own scaler because when making predictions, you need to invert **only the target scale**, not the feature scale. The model outputs a scaled Close price; `scaler_y.inverse_transform(pred)` converts it back to actual dollars.

Without separate scalers, you would need to create a dummy feature vector, put the prediction in the Close column, and inverse-transform the whole thing — messy and error-prone.

### 14.4 StandardScaler (used for GBM models)

```
x_scaled = (x - mean) / std    → output: mean=0, std=1
```

Tree models are scale-invariant (splits are threshold-based, not distance-based), but standard scaling is applied anyway for consistency and in case future additions require it.

### 14.5 Sequence Creation

For LSTM/TFT, the flat feature matrix is converted into overlapping windows:

```python
def create_sequences(X, y, lookback=60):
    for i in range(lookback, len(X)):
        Xs.append(X[i-lookback:i])   # shape: (lookback, n_features)
        ys.append(y[i])              # the label is the day AFTER the window
```

A lookback of 60 means: to predict the price on day 61, the model sees days 1-60. Window advances by one day each time, creating highly correlated training examples. This is fine — the model learns to use the full temporal context.

---

## 15. Training Mechanics: Loss, Optimiser, Callbacks

### 15.1 Huber Loss

```
L_δ(y, ŷ) = 0.5 * (y-ŷ)²           if |y-ŷ| ≤ δ
           = δ * (|y-ŷ| - 0.5*δ)    otherwise
```

Huber loss is quadratic near zero (like MSE) but linear for large errors (like MAE). This makes it robust to outliers — a price spike that makes the error 10x normal won't dominate the gradient update. Default δ=1.0 in Keras.

**Why not MSE?** Stock prices occasionally gap 5-10% overnight. MSE squares these, making the gradient 100x larger and destabilising training. Huber caps the gradient linearly for large errors.

**Why not MAE?** MAE's gradient is constant (always ±1 per sample), which makes it harder to find precise minima. Huber gets the best of both: precise gradients near the solution, robust gradients for outliers.

### 15.2 Adam Optimiser

Adam (Adaptive Moment Estimation) maintains per-parameter learning rates:

```
m_t = β₁ * m_{t-1} + (1-β₁) * g_t          # first moment (gradient mean)
v_t = β₂ * v_{t-1} + (1-β₂) * g_t²         # second moment (gradient variance)
m̂_t = m_t / (1 - β₁^t)                       # bias-corrected
v̂_t = v_t / (1 - β₂^t)

W ← W - η * m̂_t / (sqrt(v̂_t) + ε)
```

Default: `β₁=0.9, β₂=0.999, ε=1e-7, η=1e-3`. The division by `sqrt(v̂_t)` normalises the update: parameters that receive large, consistent gradients get smaller updates (already converging), while parameters with noisy gradients adapt their own rates. This is why Adam works well without extensive learning rate tuning.

### 15.3 Validation Split

```python
model.fit(X_train, y_train, validation_split=0.1)
```

10% of training data is held out as a validation set. The model never trains on this data, but callbacks monitor `val_loss` to detect overfitting and trigger EarlyStopping. This is a second holdout within training — distinct from the test set.

---

## 16. Evaluation Metrics — What They Actually Mean

### 16.1 Regression Metrics (for price prediction)

**MAE (Mean Absolute Error):**
```
MAE = (1/n) * Σ |y_i - ŷ_i|
```
In dollars. If MAE = $3.50 for a $150 stock, the model is off by about 2.3% on average.

**RMSE (Root Mean Squared Error):**
```
RMSE = sqrt( (1/n) * Σ (y_i - ŷ_i)² )
```
Penalises large errors more than MAE. RMSE > MAE always (unless all errors are equal). A large gap between RMSE and MAE indicates a few large errors are dominating.

### 16.2 Direction Metrics (for trading usefulness)

The model predicts prices (regression), but traders care about direction. The code converts predictions to direction:

```python
y_true_dir = (np.diff(y_true) > 0).astype(int)   # 1=up, 0=down
y_pred_dir = (np.diff(y_pred) > 0).astype(int)
```

`np.diff([100, 102, 101])` = `[2, -1]` → `[True, False]` → `[1, 0]`.

**Accuracy**: `correct_directions / total`. A baseline of 50% is random (coin flip). Anything consistently above 55% is meaningful for trading.

**Precision**: `TP / (TP + FP)`. Of the times the model predicted "up", how often was it actually up? High precision = few false positives.

**Recall**: `TP / (TP + FN)`. Of the actual "up" days, how many did the model catch? High recall = few missed opportunities.

**F1-Score**: harmonic mean of precision and recall. Balances the two — a model that always predicts "up" has high recall but low precision; F1 captures this.

For trading with 5x leverage, **precision matters more than recall** — a false positive (entering a wrong direction trade) loses 7.5% of position.

---

## 17. GPU Acceleration and Fallback Strategy

### 17.1 Why GPU?

Modern GPUs have thousands of CUDA cores designed for parallel floating-point operations. Training a neural network on batches of data is embarrassingly parallel — every sample in a batch is independent. A GPU can compute a batch 10-50x faster than a CPU.

For tree models (XGBoost, LightGBM), GPU acceleration is also available for the histogram-building step, typically 2-5x speedup.

### 17.2 The Fallback Pattern

```python
_using_gpu = False
try:
    model = _make_model(use_gpu=True)
    model.fit(X_train, y_train, ...)
    _using_gpu = True
    print("Using GPU acceleration (CUDA)")
except Exception as e:
    print(f"GPU not available, falling back to CPU: {e}")
    model = _make_model(use_gpu=False)
    model.fit(X_train, y_train, ...)
```

The entire training attempt (instantiation + fit) is wrapped in the try block. This is important: some libraries (e.g. newer XGBoost versions) raise GPU errors during `.fit()`, not during instantiation.

For Keras/LSTM, GPU is automatic — PyTorch's CUDA support means any tensor operation runs on GPU if `torch.cuda.is_available()` is True. No explicit try/except needed; the backend handles it.

---

## 18. System Architecture: How main.py Orchestrates Everything

### 18.1 Parallel Execution

```python
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(run_model, name, script, csv): name
               for name, script in models}
    for future in as_completed(futures):
        results.append(future.result())
```

Five models train simultaneously in separate threads. Each model runs as a subprocess (`subprocess.run`), not as a thread — this is critical. Python's GIL prevents true parallelism for CPU-bound Python code in threads, but subprocesses run as independent OS processes, each with their own Python interpreter and GIL. On a GPU machine, all five subprocesses can use the GPU simultaneously (via CUDA's internal scheduling).

`as_completed` yields results as they finish, rather than waiting for all to complete in order. The model that finishes fastest (usually XGBoost or LightGBM) reports first.

### 18.2 Output Parsing

Each training script's stdout is captured as a string. `main.py` extracts data using regex:

```python
# Example: extract Test MAE
mae_match = re.search(r'Test MAE:\s+\$?([\d.]+)', output)
if mae_match:
    metrics['test_mae'] = float(mae_match.group(1))
```

This is a form of **structured logging** — training scripts print in a known format, and the orchestrator parses it. The alternative (passing data through files or pipes) is more complex. The downside is that format changes in print statements break parsing silently.

### 18.3 HTML Report Generation

The report is assembled as a list of strings, then joined and written:

```python
html = []
html.append('<div class="model-card">')
...
with open(output_file, 'w') as f:
    f.write('\n'.join(html))
```

This is more memory-efficient than string concatenation (O(n²) copies vs O(n)) and avoids a template engine dependency.

---

## 19. Common Pitfalls and How This Code Avoids Them

### Pitfall 1: Look-ahead bias in scaling
**Wrong**: `scaler.fit_transform(all_data)` — test data's future prices change the scaler parameters.
**Right**: `scaler.fit_transform(train_data)`, then `scaler.transform(test_data)`.

### Pitfall 2: Wrong inverse transform for multi-output scalers
**Wrong**: putting the prediction into the Close column of a 52-feature scaled array.
**Right**: separate `scaler_y` fitted only on Close, so `scaler_y.inverse_transform([[pred]])` works cleanly.

### Pitfall 3: Hardcoded sequence length in multi-step prediction
**Wrong**: `lookback = 60` hardcoded inside the prediction loop.
**Right**: `lookback` passed as a parameter matching what the model was trained on.

### Pitfall 4: Missing custom_objects on model reload
**Wrong**: `load_model('model.keras')` — crashes for custom layers.
**Right**: `load_model('model.keras', custom_objects={'TemporalAttention': TemporalAttention})`.

### Pitfall 5: GPU error only caught at instantiation, not fit
**Wrong**: try/except only around `model = XGBRegressor(device='cuda')`.
**Right**: try/except around both instantiation AND `.fit()`.

### Pitfall 6: Sequences from test data starting before test data
**Wrong**: creating sequences that span train/test boundary.
**Right**: create sequences from `X_test_scaled` only. The first `lookback` rows of the test set are consumed as context — that is why `test_dates = test_df['Date'].values[lookback:]`.

### Pitfall 7: Regex capturing across newlines
**Wrong**: `re.search(r'SIGNAL:\s+([A-Z\s()]+)', output)` — `\s` matches `\n`, captures multiple lines.
**Right**: `re.search(r'SIGNAL:\s+([A-Za-z ()\t]+?)(?:\n|$)', output)` — stops at line end.

---

## 20. What to Study Next

You now understand everything in this codebase. Here is what to learn next, in order of immediate impact:

### Immediate (will directly improve this project)

1. **Backtesting frameworks** — `backtrader` or `vectorbt`. The current system only predicts; a backtest actually simulates placing trades, tracking P&L, drawdown, and Sharpe ratio over historical data.

2. **Walk-forward optimisation** — hyperparameter tuning where the search window also advances forward in time. Prevents overfitting of hyperparameters to a specific period.

3. **Position sizing** — Kelly criterion and fractional Kelly. Rather than always trading the same amount, size positions proportionally to the estimated edge: `f = (bp - q) / b` where b=odds, p=win probability, q=1-p.

### Intermediate (will open new model architectures)

4. **Transformer architecture in depth** — "Attention Is All You Need" (Vaswani et al., 2017). The original paper is readable and short (15 pages). Multi-head attention, positional encoding, and the full encoder-decoder architecture.

5. **Normalising flows and probabilistic outputs** — instead of predicting a single price, predict a distribution over prices. This gives calibrated uncertainty and is the foundation of modern forecasting (e.g. N-BEATS, Temporal Fusion Transformer in its full form).

6. **Reinforcement learning for trading** — frame the problem as an RL agent that decides to buy/hold/sell at each step, maximising cumulative reward. Libraries: `stable-baselines3`, `gym-anytrading`.

### Advanced (research frontier)

7. **Graph Neural Networks for market regime detection** — represent inter-stock correlations as a graph, use GNN to detect regime changes.

8. **Diffusion models for time series** — DDPM and score-based models are being adapted for unconditional and conditional time series generation, enabling much richer Monte Carlo scenarios.

9. **Causality in time series** — Granger causality, PCMCI. Understanding whether indicator X *causes* price movement or merely *correlates* with it is the open research question in quantitative finance.

### Papers worth reading

- Lim et al. (2021) — "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
- Hochreiter & Schmidhuber (1997) — "Long Short-Term Memory" (the original LSTM paper)
- Chen & Guestrin (2016) — "XGBoost: A Scalable Tree Boosting System"
- Ke et al. (2017) — "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
- Dauphin et al. (2017) — "Language Modeling with Gated Convolutional Networks" (the GLU paper)

---

*End of study guide. The best way to learn this is to change one component at a time and observe the effect on val_loss and direction accuracy. Start with the lookback window (try 30, 60, 90) and the CNN filter count. Trust the metrics, not the theory alone.*
