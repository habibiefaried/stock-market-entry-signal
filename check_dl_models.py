"""
Compare LSTM/TFT vs tree models on the same dataset.
"""
import subprocess, sys, os, re

ticker = 'AAPL'
csv_file = f'{ticker}_daily_data_20260527.csv'

# Fetch if needed
if not os.path.exists(csv_file):
    import yfinance as yf
    t = yf.Ticker(ticker)
    df = t.history(period='96mo', interval='1d')
    df.reset_index(inplace=True)
    df = df[['Date','Open','High','Low','Close','Volume']]
    df.to_csv(csv_file, index=False)
    print(f'Fetched {len(df)} rows')

results = {}
for name, script in [('LSTM', 'train_lstm.py'), ('TFT', 'train_tft.py'),
                      ('LightGBM', 'train_lightgbm.py'), ('XGBoost', 'train_xgboost.py')]:
    print(f'\nRunning {name}...')
    r = subprocess.run([sys.executable, script, csv_file],
                       capture_output=True, text=True, timeout=600,
                       cwd=os.path.dirname(os.path.abspath(__file__)))
    out = r.stdout

    mae_m = re.search(r'Test MAE:\s+\$?([\d.]+)', out)
    rmse_m = re.search(r'Test RMSE:\s+\$?([\d.]+)', out)
    acc_m = re.search(r'Test Accuracy:\s+([\d.]+)%', out)
    f1_m = re.search(r'Test F1-Score:\s+([\d.]+)%', out)

    results[name] = {
        'mae': float(mae_m.group(1)) if mae_m else None,
        'rmse': float(rmse_m.group(1)) if rmse_m else None,
        'acc': float(acc_m.group(1)) if acc_m else None,
        'f1': float(f1_m.group(1)) if f1_m else None,
        'returncode': r.returncode,
    }
    print(f'  MAE=${results[name]["mae"]}, RMSE=${results[name]["rmse"]}, '
          f'Acc={results[name]["acc"]}%, F1={results[name]["f1"]}%, '
          f'exit={r.returncode}')

print('\n' + '='*60)
print('MODEL COMPARISON')
print('='*60)
print(f'{"Model":<15} {"MAE":>10} {"RMSE":>10} {"Acc":>10} {"F1":>10}')
print('-'*55)
for name, m in results.items():
    print(f'{name:<15} ${m["mae"]:>8.2f} ${m["rmse"]:>8.2f} '
          f'{m["acc"]:>8.1f}% {m["f1"]:>8.1f}%')

# Check if LSTM/TFT are materially worse
tree_mae = min(results['LightGBM']['mae'], results['XGBoost']['mae'])
lstm_mae = results['LSTM']['mae']
tft_mae = results['TFT']['mae']

print(f'\nBest tree MAE: ${tree_mae:.2f}')
print(f'LSTM MAE:      ${lstm_mae:.2f} ({lstm_mae/tree_mae:.1f}x worse)')
print(f'TFT MAE:       ${tft_mae:.2f} ({tft_mae/tree_mae:.1f}x worse)')

if lstm_mae > tree_mae * 1.5 or tft_mae > tree_mae * 1.5:
    print('\nVERDICT: Deep learning models are SIGNIFICANTLY worse than tree models.')
    print('Recommendation: Replace LSTM/TFT with simpler statistical models.')
else:
    print('\nVERDICT: DL models are comparable to tree models.')
