import pandas as pd
from train_statistical import *

df = pd.read_csv('AAPL_daily_data_20260527.csv')
df = compute_indicators(df)
print(f'Rows: {len(df)}')

# Current prediction
s, c, m = knn_predict(df, FEATURES, k=50, calibrate=False)
print(f'Current: Signal={s} Conf={c:.1f}% Move={m:.4f}%')

# Walk-back predictions
for i in [len(df)-200, len(df)-100, len(df)-50, len(df)-20]:
    sub = df.iloc[:i+1]
    _, _, move = knn_predict(sub, FEATURES, k=50)
    actual_ret = df['Close'].pct_change().iloc[i+1] * 100 if i+1 < len(df) else 0
    print(f'  idx={i}: pred_move={move:.4f}%  actual_ret={actual_ret:.4f}%')
