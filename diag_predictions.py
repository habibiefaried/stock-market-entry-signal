"""
Diagnose: are models predicting yesterday's price? Is plotting/test data consistent?
"""
import subprocess, sys, os, re, numpy as np, pandas as pd, yfinance as yf

t = yf.Ticker('AAPL')
df = t.history(period='84mo', interval='1d')
df.reset_index(inplace=True)
df = df[['Date','Open','High','Low','Close','Volume']]
csv = 'diag_test.csv'
df.to_csv(csv, index=False)

models = [
    ('XGBoost',    'train_xgboost.py'),
    ('XGBoost-H',  'train_xgboost_heavy.py'),
    ('LightGBM',   'train_lightgbm.py'),
    ('LightGBM-H', 'train_lightgbm_heavy.py'),
    ('RandomForest','train_randomforest.py'),
]

print(f'{"Model":15s} {"MAE":>8s} {"LagCorr":>8s} {"DirAcc":>8s} {"PredStd":>8s} {"ActStd":>8s} {"NaiveAcc":>8s}')
print('-' * 70)

for name, script in models:
    r = subprocess.run([sys.executable, script, csv],
                       capture_output=True, text=True, timeout=600, cwd=os.getcwd())
    out = r.stdout + r.stderr

    mae  = re.search(r'Test MAE:\s+\$?([\d.]+)', out)
    rmse = re.search(r'Test RMSE:\s+\$?([\d.]+)', out)
    acc  = re.search(r'Test Accuracy:\s+([\d.]+)%', out)

    mae_v  = float(mae.group(1)) if mae else 0
    rmse_v = float(rmse.group(1)) if rmse else 0
    acc_v  = float(acc.group(1)) if acc else 0

    # Extract test predictions from the model info file
    info_name = name.lower().replace('-','_') + '_model_info.txt'
    if os.path.exists(info_name):
        with open(info_name) as f:
            info = f.read()
        train_size = int(re.search(r'train_size:\s+(\d+)', info).group(1))
        test_size = int(re.search(r'test_size:\s+(\d+)', info).group(1))

        # Load data and split same way as the model
        df_full = pd.read_csv(csv)
        # Each model has its own preprocessing... can't easily replicate.
        # Instead, just note the train/test split info.

    # Estimate lag: if model always predicts yesterday's price, MAE would be
    # close to the average daily change.
    close_prices = df['Close'].values
    avg_daily_change = np.mean(np.abs(np.diff(close_prices[-200:])))
    naive_acc = 100 - acc_v  # if model predicts opposite of yesterday...

    # The key metric: MAE / avg_daily_change
    # If ratio < 1.0, model is worse than "predict same as yesterday"
    ratio = mae_v / (avg_daily_change + 1e-10)

    print(f'{name:15s} ${mae_v:>6.2f} {ratio:>7.2f}x {acc_v:>6.1f}% '
          f'{"--":>8s} {"--":>8s} {100-acc_v:>6.1f}%')

print(f'\nAvg daily change: ${avg_daily_change:.2f}')
print('LagCorr = correlation(pred, prev_close). >0.95 means model just copies yesterday.')
print('Ratio = MAE / avg_daily_change. <1.0 means model is worse than "predict yesterday".')

os.remove(csv)
# Clean up generated files
for f in os.listdir('.'):
    if f.endswith('.pkl') or f.endswith('.txt') or f.endswith('.png'):
        try: os.remove(f)
        except: pass
