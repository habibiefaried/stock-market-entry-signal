# Changelog

## 2026-05-20 - Major Update v2

### Removed Features

#### ARIMA Model Removed
- **Reason**: Training takes too long with 48 months of data
- **Action**: Moved `train_arima.py` to `train_arima.py.bak`
- **Impact**: Now running 3 models instead of 4 (LSTM, XGBoost, LightGBM)
- **Files updated**: 
  - `main.py` - Removed ARIMA from model list, updated documentation
  - `.gitignore` - Added `train_arima.py` and `*.bak` to ignore list
  - `README.md` - Removed all ARIMA documentation and examples

### Configuration Changes

#### Train/Test Split Updated
- **Previous**: 5/6 split (83% train, 17% test)
- **New**: 9/10 split (90% train, 10% test)
- **Reason**: With 48 months of data, we have more samples, so 10% test is sufficient
- **Details**:
  - 48 months total data
  - ~43 months (90%) for training
  - ~5 months (10%) for testing
  - More training data = better model performance
- **Files affected**:
  - `train_lstm.py`
  - `train_xgboost.py`
  - `train_lightgbm.py`

### Summary of Changes

**Models Active:**
- LSTM (Deep Learning)
- XGBoost (Gradient Boosting)
- LightGBM (Gradient Boosting)

**Training Configuration:**
- Default data: 48 months (4 years)
- Train/test: 90/10 split
- Parallel execution: 3 models (was 4)

## 2026-05-20 - Major Update v1

### Fixed Issues

#### 1. Unicode Encoding Errors
- **Issue**: Arrow character `→` caused `UnicodeEncodeError: 'charmap' codec can't encode character` on Windows
- **Solution**: Replaced all Unicode arrows with ASCII arrows `->` across all training scripts
- **Files affected**: 
  - `train_arima.py` (8 occurrences)
  - `train_lightgbm.py` (4 occurrences)
  - `train_xgboost.py` (4 occurrences)
  - `train_lstm.py` (3 occurrences)

#### 2. Warning Suppression
- **Issue**: Warnings from matplotlib, statsmodels, and other libraries cluttering output and HTML reports
- **Solution**: Comprehensive warning suppression added to all scripts
- **Implementation**:
  - `warnings.filterwarnings('ignore')` - General warnings
  - `matplotlib.use('Agg')` - Non-interactive backend
  - Logger suppression for: matplotlib, PIL, lightgbm, xgboost, keras, torch
- **Files affected**: All training scripts and `main.py`

#### 3. Emojis Removed
- **Issue**: Emoji characters could cause encoding issues
- **Solution**: Replaced all emojis with plain text equivalents
- **Changes**:
  - `✓` → `[OK]`
  - `✗` → `[FAIL]`
  - `📈` → `[BUY]` or `[UP]`
  - `📉` → `[SHORT]` or `[DOWN]`
  - `➡️` → `[HOLD]`
  - `📊` → removed from report title
- **Files affected**: `main.py`

### New Features

#### 1. Integrated Data Fetching in main.py
- **Feature**: Can now fetch stock data directly from main.py
- **Usage**: 
  ```bash
  python main.py --ticker MSFT
  python main.py --ticker MSFT --months 60
  python main.py --ticker BTC-USD --months 48
  ```
- **Benefits**: Streamlined workflow - fetch and train in one command

#### 2. Updated Default Historical Data Period
- **Change**: Default changed from 36 months (3 years) to 48 months (4 years)
- **Rationale**: More historical data improves model training
- **Files affected**: `fetch_stock_data.py`, `main.py`

### Technical Improvements

#### Warning Suppression Details

**train_arima.py**:
```python
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
```

**train_lightgbm.py**:
```python
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('lightgbm').setLevel(logging.ERROR)
```

**train_xgboost.py**:
```python
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('xgboost').setLevel(logging.ERROR)
```

**train_lstm.py**:
```python
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('keras').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
```

**main.py**:
```python
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
```

### Documentation Updates

- Updated README.md with "Quick Start" section
- Added Option 1 (Recommended): Streamlined approach using `main.py --ticker`
- Added Option 2: Manual step-by-step approach
- Clarified default historical data period (48 months = 4 years)

### Testing Recommendations

After these changes, test the following:

1. **ARIMA training** (previously failed with Unicode error):
   ```bash
   python main.py --ticker MSFT
   ```

2. **Verify no warnings in output**:
   - Check console output is clean
   - Check HTML report has no warning messages

3. **Test with different tickers**:
   ```bash
   python main.py --ticker AAPL --months 48
   python main.py --ticker BTC-USD --months 36
   ```

4. **Test backward compatibility**:
   ```bash
   python main.py MSFT_daily_data_20260520.csv
   ```

### Breaking Changes

None. All changes are backward compatible.

### Notes

- All text output now uses plain ASCII characters
- No more encoding issues on Windows systems
- Cleaner console and HTML report output
- Default data period increased for better model accuracy
