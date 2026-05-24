# How to Improve PPO Performance

## Current Issues

The current PPO implementation has **15.3% win rate** and poor performance because:

1. **Crude gradient approximation** - Uses `np.sign()` finite differences instead of proper backpropagation
2. **Limited state information** - Only 16 dimensions, missing important context
3. **Simple reward structure** - Doesn't guide learning effectively
4. **No experience replay** - Limited training data not used efficiently
5. **Random curriculum** - Learns from hard and easy cases simultaneously

---

## Improvement Roadmap (Ordered by Impact)

### 🔥 **1. Switch to PyTorch (Biggest Impact)**

**Current**: Crude numpy gradients with `np.sign()`
**Improvement**: Proper automatic differentiation

**Files Created**:
- `agent_trader_torch.py` - Full PyTorch implementation

**Expected Improvement**: Win rate 15% → 35-50%

**How to Use**:
```bash
# Install PyTorch first
pip install torch

# Then modify agent_trader.py to use PyTorch policy
```

**Key Benefits**:
- ✅ Real gradient descent instead of finite differences
- ✅ Deeper networks (128-256 hidden units)
- ✅ Layer normalization for stability
- ✅ Proper gradient clipping
- ✅ Multiple update epochs per batch

---

### 🎯 **2. Enhanced State Representation (Already Applied)**

**Current**: 16 dimensions
**Improvement**: 24 dimensions with model consensus features

**New Features Added**:
```python
# Market indicators
- Volatility (normalized)
- ATR/price ratio

# Model agreement metrics
- Signal disagreement (std)
- Average confidence
- Consensus vote
- High-confidence model count
```

**Expected Improvement**: +5-10% win rate

**Status**: ✅ Already implemented in `agent_trader.py`

---

### 💰 **3. Improved Reward Structure (Already Applied)**

**Changes Made**:
```python
REWARD_TP = 2.0            # Increased from 1.67
REWARD_TIMEOUT = -0.3      # New penalty
REWARD_CORRECT_DIR = 0.1   # Bonus for matching consensus
MAX_DAYS = 10              # Increased from 5
```

**Benefits**:
- ✅ Stronger signal for winning trades
- ✅ Discourages timeouts
- ✅ Guides agent toward model consensus
- ✅ More time for trades to develop

**Expected Improvement**: +5-10% win rate

**Status**: ✅ Already implemented in `agent_trader.py`

---

### 🔄 **4. Experience Replay (High Impact)**

**File**: `experience_replay.py`

**Two Options**:

#### A. Basic Replay Buffer
```python
from experience_replay import ReplayBuffer

buffer = ReplayBuffer(max_size=10000)

# During training
buffer.add(state, action, reward, next_state, done, log_prob, value)

# Sample for training
batch = buffer.sample(batch_size=128)
```

#### B. Prioritized Replay (Better)
```python
from experience_replay import PrioritizedReplayBuffer

buffer = PrioritizedReplayBuffer(max_size=10000)

# Prioritizes rare TP/SL outcomes
buffer.add(state, action, reward, next_state, done, log_prob, value, outcome='TP')
```

**Benefits**:
- ✅ Reuse experiences multiple times
- ✅ Learn from rare events (TP hits)
- ✅ Better sample efficiency with limited data

**Expected Improvement**: +10-15% win rate

**Integration Effort**: Medium (need to modify training loop)

---

### 📚 **5. Curriculum Learning (Medium Impact)**

**File**: `curriculum_learning.py`

**How it Works**:
1. Stage 1: Train on cases where 6-7 models agree (easy)
2. Stage 2: Add cases where 4-5 models agree (medium)
3. Stage 3: Add split decisions (hard)
4. Stage 4: All scenarios

**Usage**:
```python
from curriculum_learning import CurriculumScheduler, train_with_curriculum

curriculum = CurriculumScheduler(signals_df)
train_with_curriculum(env, policy, curriculum, episodes_per_stage=1500)
```

**Benefits**:
- ✅ Faster learning by starting simple
- ✅ Better convergence
- ✅ More stable training

**Expected Improvement**: +5-10% win rate, faster convergence

**Integration Effort**: Medium

---

### ⚙️ **6. Hyperparameter Tuning**

**Current Settings**:
```python
lr = 3e-4                  # Learning rate
hidden = 64                # Hidden layer size
clip_eps = 0.2            # PPO clip epsilon
entropy_coef = 0.01       # Entropy coefficient
n_epochs = 4              # Update epochs per batch
```

**Recommended Tuning**:
```python
# For PyTorch version
lr = 1e-4                 # Lower for stability
hidden = 128              # Larger network
clip_eps = 0.15           # Slightly tighter
entropy_coef = 0.02       # More exploration
n_epochs = 10             # More updates per batch
batch_size = 128          # Larger batches
```

**Use Grid Search**:
```python
for lr in [1e-4, 3e-4, 1e-3]:
    for hidden in [64, 128, 256]:
        policy = PPOPolicyTorch(state_dim=24, hidden=hidden, lr=lr)
        # Train and evaluate
```

**Expected Improvement**: +5-10% win rate

**Effort**: Low (just parameter changes)

---

## 🎯 Recommended Implementation Order

### Phase 1: Quick Wins (Week 1)
1. ✅ Enhanced state representation (DONE)
2. ✅ Improved rewards (DONE)
3. Hyperparameter tuning

**Expected Result**: Win rate 15% → 25%

### Phase 2: Architecture Upgrade (Week 2)
4. Switch to PyTorch
5. Deeper network (128-256 hidden)
6. Tune learning rate and batch size

**Expected Result**: Win rate 25% → 40%

### Phase 3: Sample Efficiency (Week 3)
7. Add Prioritized Replay Buffer
8. Increase training episodes (5K → 10K)

**Expected Result**: Win rate 40% → 50%

### Phase 4: Advanced Techniques (Week 4)
9. Implement curriculum learning
10. Add model ensembling (train 3 policies, vote)

**Expected Result**: Win rate 50% → 60%+

---

## 🧪 Quick Test Command

After improvements, test with:
```bash
# Delete old weights
rm -f rl_agent_weights.npz rl_agent_csv_hash.txt

# Train and test
python agent_trader.py ADBE_daily_data_20260524.csv

# Look for these metrics:
# - Win Rate > 40%
# - Profit Factor > 1.5
# - Sharpe Ratio > 1.0
# - Max Drawdown > -30%
```

---

## 📊 Performance Targets

| Metric | Current | Target | Excellent |
|--------|---------|--------|-----------|
| Win Rate | 15% | 40% | 60% |
| Profit Factor | 0.54 | 1.5 | 2.5 |
| Sharpe Ratio | -4.64 | 1.0 | 2.0 |
| Max Drawdown | -100% | -30% | -15% |

---

## 🔧 Alternative: Simpler Meta-Strategy

If PPO remains difficult to train, consider these alternatives:

### Option A: Weighted Voting
```python
def meta_decision(signals, probs):
    """Simple weighted vote by model confidence"""
    weighted_long = sum(p for s, p in zip(signals, probs) if s == 1)
    weighted_short = sum(p for s, p in zip(signals, probs) if s == -1)

    if weighted_long > weighted_short * 1.2:  # 20% confidence margin
        return 'LONG'
    elif weighted_short > weighted_long * 1.2:
        return 'SHORT'
    else:
        return 'HOLD'
```

### Option B: Stacking (Train XGBoost on Model Outputs)
```python
import xgboost as xgb

# Use model signals/probs as features
X = signals_df[['xgboost_signal', 'xgboost_prob', 'lstm_signal', ...]]
y = actual_outcomes  # 1=TP, 0=TIMEOUT, -1=SL

meta_model = xgb.XGBClassifier()
meta_model.fit(X, y)
```

**These might perform better than PPO with limited data!**

---

## 📝 Final Notes

**The #1 issue**: Limited training data (784 examples) makes RL challenging.

**Solutions**:
1. Use PyTorch for better gradients
2. Add experience replay to reuse data
3. Consider simpler meta-strategies (voting, XGBoost stacking)

The voting fallback you currently have is actually quite sensible and might outperform PPO until you implement the improvements above.
