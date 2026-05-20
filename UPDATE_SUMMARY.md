# Update Summary - 2026-05-20

## Changes Made

### 1. Train/Test Split Changed to 9/10
**Previous**: 5/6 split (83.3% train, 16.7% test)  
**New**: 9/10 split (90% train, 10% test)

**Reason**: With 48 months (4 years) of historical data by default, we have more samples, so 10% test data is sufficient while giving more data for training.

**Details**:
- 48 months total data
- ~43 months (90%) for training
- ~5 months (10%) for testing
- More training data improves model accuracy

**Files Updated**:
- `train_lstm.py` - Updated `split_train_test()` function
- `train_xgboost.py` - Updated `split_train_test()` function
- `train_lightgbm.py` - Updated `split_train_test()` function

### 2. ARIMA Model Removed
**Reason**: ARIMA training takes too long with 48 months of data

**Actions Taken**:
- Moved `train_arima.py` to `train_arima.py.bak` (backup)
- Removed ARIMA from `main.py` model list
- Updated `ThreadPoolExecutor` max_workers from 4 to 3
- Added `train_arima.py` and `*.bak` to `.gitignore`
- Removed all ARIMA documentation from README.md

**Current Active Models**:
1. LSTM (Deep Learning)
2. XGBoost (Gradient Boosting)
3. LightGBM (Gradient Boosting)

**Files Updated**:
- `main.py` - Removed ARIMA from models list, updated docs
- `.gitignore` - Added ARIMA file to ignore list
- `README.md` - Removed ARIMA sections and examples
- `CHANGELOG.md` - Documented changes

## Summary

### Before
- 4 models: ARIMA, LSTM, XGBoost, LightGBM
- Train/test split: 5/6 (83%/17%)
- Default data: 48 months

### After
- 3 models: LSTM, XGBoost, LightGBM
- Train/test split: 9/10 (90%/10%)
- Default data: 48 months

### Benefits
✓ Faster training (3 models instead of 4)  
✓ More training data per model (90% vs 83%)  
✓ Adequate test data (10% = ~5 months with 48-month dataset)  
✓ Better model accuracy with more training data

## Usage

**Quick start (recommended)**:
```bash
python main.py --ticker MSFT
```

This will:
1. Fetch 48 months of MSFT data
2. Train 3 models in parallel (LSTM, XGBoost, LightGBM)
3. Generate HTML comparison report

**Custom data period**:
```bash
# 5 years of data
python main.py --ticker MSFT --months 60

# 2 years of data
python main.py --ticker BTC-USD --months 24
```

## Testing

To verify the changes work correctly:

1. **Test training with new split**:
   ```bash
   python main.py --ticker MSFT
   ```

2. **Check that output shows 90/10 split**:
   - Look for "Training set: X records" and "Test set: Y records"
   - Verify ratio is approximately 9:1

3. **Verify 3 models run** (not 4):
   - Check console output shows: LSTM, XGBoost, LightGBM
   - No ARIMA mentioned

4. **Check HTML report**:
   - Open `RESULT-MSFT-{DATE}.html`
   - Should show 3 models
   - Should have comparison and consensus

## Rollback (if needed)

If you need to restore ARIMA:

```bash
# Restore ARIMA script
mv train_arima.py.bak train_arima.py

# Revert main.py changes manually:
# - Add ('ARIMA', 'train_arima.py') to models list
# - Change ThreadPoolExecutor max_workers back to 4
```

## Files Modified

### Python Scripts
- `train_lstm.py` - Train/test split 5/6 → 9/10
- `train_xgboost.py` - Train/test split 5/6 → 9/10
- `train_lightgbm.py` - Train/test split 5/6 → 9/10
- `main.py` - Removed ARIMA, updated model count

### Documentation
- `README.md` - Removed ARIMA docs, updated split info
- `CHANGELOG.md` - Added update details
- `.gitignore` - Added ARIMA and backup files

### Backups
- `train_arima.py.bak` - Original ARIMA script (backed up)

## Notes

- The 9/10 split is standard for time series with large datasets
- 10% test data (~5 months with 48-month dataset) is sufficient for evaluation
- More training data (90% vs 83%) helps capture longer-term patterns
- ARIMA can be restored from backup if needed in the future
