"""
RandomForest-Heavy — bagging ensemble with heavy computation.

Same proven feature set as light RandomForest (lags + simple derived),
but with 2500 trees, deeper splits, aggressive bootstrapping, and
7-fold walk-forward validation for more realistic evaluation.

Key differences from light RF:
- 2500 trees (vs 1000) — smoother predictions, lower variance
- Depth 22 (vs 15) — learns more complex patterns
- max_samples=0.6 — each tree sees only 60% of data → higher diversity
- 7 walk-forward folds (vs 5) — tests on more market regimes
- min_samples_leaf=5 — prevents tiny leaf overfitting with deep trees
- max_features=0.5 — uses half the features per split for decorrelation

Usage:
    python train_randomforest_heavy.py MSFT_daily_data_20260520.csv
    python train_randomforest_heavy.py MSFT_daily_data_20260520.csv --n_estimators 3000 --max_depth 25
"""

import argparse, os, sys, warnings, logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
)

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trade_probability_analyzer import (
    predict_multi_day_path,
    monte_carlo_simulation,
    find_similar_patterns,
    calculate_ensemble_probability,
    format_analysis_report,
)

# ===========================================================================
# FEATURE ENGINEERING — same proven simple features as light RandomForest
# ===========================================================================

def create_features(df):
    """Simple lag-based features + price changes + volatility + MAs."""
    out = df.copy()
    c = out['Close']
    for lag in [1, 2, 3, 5, 10]:
        out[f'Close_lag_{lag}'] = c.shift(lag)
        out[f'Volume_lag_{lag}'] = out['Volume'].shift(lag)
    out['Price_change_1d'] = c.pct_change(1) * 100
    out['Price_change_5d'] = c.pct_change(5) * 100
    out['Price_change_10d'] = c.pct_change(10) * 100
    out['Volatility_5d'] = c.rolling(5).std()
    out['Volatility_10d'] = c.rolling(10).std()
    out['MA_5'] = c.rolling(5).mean()
    out['MA_10'] = c.rolling(10).mean()
    out['MA_20'] = c.rolling(20).mean()
    out['Target'] = c.pct_change().shift(-1) * 100
    return out.dropna().reset_index(drop=True)


# ===========================================================================
# WALK-FORWARD SPLITS
# ===========================================================================

def walk_forward_splits(n_samples, n_splits=5):
    initial = int(n_samples * 0.7)
    chunk = (n_samples - initial) // n_splits
    splits = []
    for i in range(n_splits):
        end = initial + i * chunk
        start = end
        stop = min(start + chunk, n_samples)
        if stop <= start:
            break
        splits.append((list(range(0, end)), list(range(start, stop))))
    return splits


# ===========================================================================
# MAIN
# ===========================================================================

def run_randomforest_heavy(
    csv_file,
    n_estimators=1500,
    max_depth=20,
    max_features=0.5,
    min_samples_split=7,
    min_samples_leaf=4,
    max_samples=0.5,
    max_leaf_nodes=150,
):
    print("=" * 60)
    print("RANDOMFOREST-HEAVY MODEL")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(csv_file)
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    if any(c not in df.columns for c in required):
        raise ValueError("CSV missing OHLCV columns")

    df = create_features(df)
    df['Date'] = pd.to_datetime(df['Date'], utc=True) if 'Date' in df.columns else None

    all_features = [c for c in df.columns
                    if c not in ['Date', 'Target']]
    n = len(df)
    print(f"Records: {n}  Features: {len(all_features)}")
    print(f"Trees: {n_estimators}  Depth: {max_depth}  "
          f"MinSplit: {min_samples_split}  MinLeaf: {min_samples_leaf}")
    print(f"MaxSamples: {max_samples}  MaxLeafNodes: {max_leaf_nodes}")

    # Walk-forward validation
    splits = walk_forward_splits(n, n_splits=7)
    print(f"\nWalk-forward: {len(splits)} folds")

    all_train_preds, all_train_actuals = [], []
    all_test_preds,  all_test_actuals  = [], []
    all_train_dirs,  all_test_dirs     = [], []
    fold_feature_votes = {}

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        print(f"\nFold {fold_i+1}/{len(splits)}: "
              f"train={len(train_idx)} test={len(test_idx)}")

        # Per-fold feature selection on TRAINING data only
        fold_features = all_features
        n_feat_wanted = min(35, len(all_features))
        if n_feat_wanted < len(all_features):
            train_df = df.iloc[train_idx]
            corrs = train_df[all_features].corrwith(
                train_df['Target']).abs().fillna(0)
            fold_features = corrs.nlargest(n_feat_wanted).index.tolist()
            for feat in fold_features:
                fold_feature_votes[feat] = fold_feature_votes.get(feat, 0) + 1

        close_col = fold_features.index('Close')
        X_fold = df[fold_features].values
        y_fold = df['Target'].values

        X_tr, X_te = X_fold[train_idx], X_fold[test_idx]
        y_tr, y_te = y_fold[train_idx], y_fold[test_idx]

        # Scale
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        # Train
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_samples=max_samples,
            max_leaf_nodes=max_leaf_nodes,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        rf.fit(X_tr_s, y_tr)

        tr_pred = rf.predict(X_tr_s)
        te_pred = rf.predict(X_te_s)

        # Convert returns to prices for storage
        prev_tr = X_tr[:, close_col]
        prev_te = X_te[:, close_col]
        tr_pred_price = prev_tr * (1 + tr_pred / 100)
        te_pred_price = prev_te * (1 + te_pred / 100)
        y_tr_price = prev_tr * (1 + y_tr / 100)
        y_te_price = prev_te * (1 + y_te / 100)

        all_train_preds.extend(tr_pred_price)
        all_train_actuals.extend(y_tr_price)
        all_test_preds.extend(te_pred_price)
        all_test_actuals.extend(y_te_price)

        tr_dir_pred   = (tr_pred > 0).astype(int)
        tr_dir_actual = (y_tr > 0).astype(int)
        te_dir_pred   = (te_pred > 0).astype(int)
        te_dir_actual = (y_te > 0).astype(int)
        all_train_dirs.extend(zip(tr_dir_actual, tr_dir_pred))
        all_test_dirs.extend(zip(te_dir_actual, te_dir_pred))

    # Aggregate metrics
    all_train_preds = np.array(all_train_preds)
    all_train_actuals = np.array(all_train_actuals)
    all_test_preds = np.array(all_test_preds)
    all_test_actuals = np.array(all_test_actuals)

    train_mae = mean_absolute_error(all_train_actuals, all_train_preds)
    train_rmse = np.sqrt(mean_squared_error(all_train_actuals, all_train_preds))
    test_mae = mean_absolute_error(all_test_actuals, all_test_preds)
    test_rmse = np.sqrt(mean_squared_error(all_test_actuals, all_test_preds))

    tr_dir_actual = np.array([d[0] for d in all_train_dirs])
    tr_dir_pred   = np.array([d[1] for d in all_train_dirs])
    te_dir_actual = np.array([d[0] for d in all_test_dirs])
    te_dir_pred   = np.array([d[1] for d in all_test_dirs])

    train_acc  = accuracy_score(tr_dir_actual, tr_dir_pred)
    train_prec = precision_score(tr_dir_actual, tr_dir_pred, zero_division=0)
    train_rec  = recall_score(tr_dir_actual, tr_dir_pred, zero_division=0)
    train_f1   = f1_score(tr_dir_actual, tr_dir_pred, zero_division=0)
    test_acc  = accuracy_score(te_dir_actual, te_dir_pred)
    test_prec = precision_score(te_dir_actual, te_dir_pred, zero_division=0)
    test_rec  = recall_score(te_dir_actual, te_dir_pred, zero_division=0)
    test_f1   = f1_score(te_dir_actual, te_dir_pred, zero_division=0)

    print(f"\n{'='*60}")
    print("RANDOMFOREST-HEAVY EVALUATION (Walk-Forward)")
    print("=" * 60)
    print(f"\nREGRESSION METRICS:")
    print(f"  Training MAE:  ${train_mae:.2f}")
    print(f"  Training RMSE: ${train_rmse:.2f}")
    print(f"  Test MAE:      ${test_mae:.2f}")
    print(f"  Test RMSE:     ${test_rmse:.2f}")
    print(f"\nCLASSIFICATION METRICS:")
    print(f"  Training Accuracy:  {train_acc*100:.2f}%")
    print(f"  Test Accuracy:      {test_acc*100:.2f}%")
    print(f"  Test Precision:     {test_prec*100:.2f}%")
    print(f"  Test Recall:        {test_rec*100:.2f}%")
    print(f"  Test F1-Score:      {test_f1*100:.2f}%")

    # ---- Final model on all data ----
    print(f"\nTraining final model on all data...")
    if fold_feature_votes:
        min_votes = max(1, len(splits) // 2)
        final_features = [f for f, v in fold_feature_votes.items()
                         if v >= min_votes]
        if len(final_features) < 8:
            final_features = all_features
    else:
        final_features = all_features

    X_final = df[final_features].values
    y_final = df['Target'].values
    close_final = final_features.index('Close')
    sc_final = StandardScaler()
    X_final_s = sc_final.fit_transform(X_final)

    final_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_samples=max_samples,
        max_leaf_nodes=max_leaf_nodes,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    final_model.fit(X_final_s, y_final)

    # ---- Plots ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fi = pd.DataFrame({
        'feature': final_features,
        'importance': final_model.feature_importances_,
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    top = fi.head(20)
    plt.barh(range(len(top)), top['importance'])
    plt.yticks(range(len(top)), top['feature'])
    plt.xlabel('Importance')
    plt.title('RandomForest-Heavy: Feature Importance (Top 20)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('randomforest_heavy_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

    plot_n = min(200, len(all_test_actuals))
    plt.figure(figsize=(15, 6))
    plt.plot(range(plot_n), all_test_actuals[-plot_n:],
             label='Actual', color='blue', linewidth=2)
    plt.plot(range(plot_n), all_test_preds[-plot_n:],
             label='Predicted', color='red', linewidth=2, alpha=0.7)
    plt.title(f'RandomForest-Heavy: Predictions (Last {plot_n} Test Samples)')
    plt.xlabel('Test Sample')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('randomforest_heavy_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plots saved.")

    # ---- Trading signal ----
    today_price = float(df['Close'].iloc[-1])
    recent = df[final_features].iloc[-1:].values
    recent_s = sc_final.transform(recent)
    tomorrow_return = final_model.predict(recent_s)[0]
    expected_move_pct = tomorrow_return
    tomorrow_price = today_price * (1 + tomorrow_return / 100)
    expected_move = tomorrow_price - today_price

    vol_20d_pct = float(df['Close'].pct_change().tail(20).std() * 100)
    sig_threshold = max(0.15 * vol_20d_pct, 0.1)

    if expected_move_pct > sig_threshold:
        signal = "BUY (LONG)"; signal_int = 1
    elif expected_move_pct < -sig_threshold:
        signal = "SHORT (SELL)"; signal_int = -1
    else:
        signal = "HOLD (No clear signal)"; signal_int = 0

    # Compute ATR inline (not in the simple feature set)
    h, l, c_raw = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c_raw.shift()).abs(), (l - c_raw.shift()).abs()], axis=1).max(axis=1)
    atr_val = float(tr.ewm(span=14, min_periods=14).mean().iloc[-1])
    if pd.isna(atr_val) or atr_val <= 0:
        atr_val = today_price * 0.02

    sl_dist = 1.5 * atr_val; tp_dist = 2.0 * atr_val
    volatility = float(df['Close'].tail(20).pct_change().dropna().std() * today_price)

    if signal_int == 1:
        stop_loss = today_price - sl_dist; take_profit = today_price + tp_dist
    elif signal_int == -1:
        stop_loss = today_price + sl_dist; take_profit = today_price - tp_dist
    else:
        stop_loss = today_price - sl_dist; take_profit = today_price + tp_dist

    confidence = test_acc * 100

    emoji = "[BUY]" if signal_int == 1 else ("[SHORT]" if signal_int == -1 else "[HOLD]")
    print(f"\n{emoji} SIGNAL: {signal}")
    print(f"\nCurrent Price (Today):      ${today_price:.2f}")
    print(f"Predicted Price (Tomorrow): ${tomorrow_price:.2f}")
    print(f"Expected Move:              ${expected_move:+.2f} ({expected_move_pct:+.2f}%)")
    print(f"\nRisk Management (Stock Price Levels):")
    print(f"  Stop Loss:    ${stop_loss:.2f} ({((stop_loss-today_price)/today_price*100):+.2f}%)")
    print(f"  Take Profit:  ${take_profit:.2f} ({((take_profit-today_price)/today_price*100):+.2f}%)")
    print(f"  Risk/Reward:   1.33:1")
    print(f"\nModel Confidence: {confidence:.1f}%")

    # ---- Probability analysis ----
    print("\n" + "="*70)
    print("Running Multi-Approach Win Probability Analysis...")
    print("="*70)

    pred_result = predict_multi_day_path(
        model=final_model, scaler=sc_final, df=df,
        feature_cols=final_features, current_price=today_price,
        stop_loss=stop_loss, take_profit=take_profit, model_type='gbm',
    )
    mc_result = monte_carlo_simulation(
        current_price=today_price, stop_loss=stop_loss,
        take_profit=take_profit, volatility=volatility,
        predicted_move_pct=expected_move_pct,
    )
    pattern_result = find_similar_patterns(
        df=df, current_price=today_price,
        stop_loss=stop_loss, take_profit=take_profit,
    )
    ensemble = calculate_ensemble_probability(
        prediction_result=pred_result,
        monte_carlo_result=mc_result,
        pattern_result=pattern_result,
    )
    report = format_analysis_report(
        prediction_result=pred_result, monte_carlo_result=mc_result,
        pattern_result=pattern_result, ensemble_result=ensemble,
        signal=signal, current_price=today_price,
        stop_loss=stop_loss, take_profit=take_profit,
    )
    print(report)
    if ensemble:
        print(f"ENSEMBLE_PROBABILITY: {ensemble['ensemble_probability']:.1f}%")
        print(f"CONFIDENCE_LEVEL: {ensemble['confidence_level']}")
        print(f"RECOMMENDATION: {ensemble['recommendation']}")

    # ---- Save ----
    joblib.dump(final_model, os.path.join(base_dir, 'randomforest_heavy_model.pkl'))
    joblib.dump(sc_final, os.path.join(base_dir, 'randomforest_heavy_scaler.pkl'))
    with open(os.path.join(base_dir, 'randomforest_heavy_features.txt'), 'w') as f:
        f.write('\n'.join(final_features))
    with open(os.path.join(base_dir, 'randomforest_heavy_model_info.txt'), 'w') as f:
        f.write(f"ticker: {os.path.basename(csv_file).split('_')[0]}\n")
        f.write(f"model_type: RandomForest-Heavy (38 indicators + walk-forward)\n")
        f.write(f"n_features: {len(final_features)}\n")
        f.write(f"n_estimators: {n_estimators}\n")
        f.write(f"max_depth: {max_depth}\n")
        f.write(f"max_features: {max_features}\n")
        f.write(f"min_samples_split: {min_samples_split}\n")
        f.write(f"min_samples_leaf: {min_samples_leaf}\n")
        f.write(f"max_samples: {max_samples}\n")
        f.write(f"max_leaf_nodes: {max_leaf_nodes}\n")
        f.write(f"train_size: {len(all_train_actuals)}\n")
        f.write(f"test_size: {len(all_test_actuals)}\n")
        f.write(f"test_mae: {test_mae}\n")
        f.write(f"test_rmse: {test_rmse}\n")
        f.write(f"test_accuracy: {test_acc}\n")
        f.write(f"test_precision: {test_prec}\n")
        f.write(f"test_recall: {test_rec}\n")
        f.write(f"test_f1: {test_f1}\n")
        f.write(f"timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\nModel saved.")
    print("=" * 60)
    print("RANDOMFOREST-HEAVY COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RandomForest-Heavy Model')
    parser.add_argument('csv_file', type=str)
    parser.add_argument('--n_estimators',      type=int,   default=1500)
    parser.add_argument('--max_depth',         type=int,   default=20)
    parser.add_argument('--max_features',      type=str,   default='0.5')
    parser.add_argument('--min_samples_split', type=int,   default=7)
    parser.add_argument('--min_samples_leaf',  type=int,   default=4)
    parser.add_argument('--max_samples',       type=float, default=0.5)
    parser.add_argument('--max_leaf_nodes',    type=int,   default=150)
    args = parser.parse_args()

    max_feat = args.max_features
    if max_feat not in ('sqrt', 'log2', None):
        try:
            max_feat = int(max_feat)
        except ValueError:
            max_feat = 'sqrt'

    run_randomforest_heavy(
        args.csv_file,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features=max_feat,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_samples=args.max_samples,
        max_leaf_nodes=args.max_leaf_nodes,
    )
