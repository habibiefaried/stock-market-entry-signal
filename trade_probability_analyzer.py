"""
Trade Probability Analyzer - Multi-Approach Win Rate Calculator

This module implements three different approaches to calculate the probability
of a trade hitting Take Profit before Stop Loss:

1. Multi-Day Sequential Prediction: Uses the trained model to predict next N days
2. Monte Carlo Simulation: Runs thousands of random price path simulations
3. Historical Pattern Matching: Finds similar past setups and their outcomes

The ensemble combines all three to give a final recommendation with confidence level.
"""

import numpy as np
import pandas as pd
from scipy import stats

# ============================================================================
# CONFIGURABLE PARAMETERS
# ============================================================================

# Multi-Day Prediction Settings
PREDICTION_DAYS = 5  # Number of days to predict ahead (5 = 1 trading week)
MIN_CONFIDENCE_THRESHOLD = 60.0  # Minimum win probability to recommend trade (%)

# Monte Carlo Simulation Settings
MONTE_CARLO_SIMULATIONS = 1000  # Number of random simulations (more = accurate but slower)
MC_DRIFT_INFLUENCE = 0.3  # How much predicted trend affects random walk (0-1)

# Historical Pattern Matching Settings
PATTERN_LOOKBACK = 200  # How many days to search for similar patterns
PATTERN_MATCH_COUNT = 50  # Number of similar patterns to find
RSI_TOLERANCE = 5  # RSI similarity tolerance (±)
VOLATILITY_TOLERANCE = 0.2  # Volatility similarity tolerance (±20%)

# Ensemble Weights (must sum to 1.0)
WEIGHT_PREDICTION = 0.4  # Weight for multi-day prediction approach
WEIGHT_MONTE_CARLO = 0.35  # Weight for Monte Carlo simulation
WEIGHT_PATTERN = 0.25  # Weight for historical pattern matching

# Confidence Levels
CONFIDENCE_HIGH = 75.0  # Above this = HIGH confidence
CONFIDENCE_MEDIUM = 65.0  # Between medium and high = MEDIUM confidence
# Below CONFIDENCE_MEDIUM = LOW confidence


# ============================================================================
# APPROACH 1: MULTI-DAY SEQUENTIAL PREDICTION
# ============================================================================

def predict_multi_day_path(model, scaler, df, feature_cols, current_price,
                           stop_loss, take_profit, n_days=PREDICTION_DAYS,
                           model_type='gbm'):
    """
    Predict next N days sequentially and check if TP or SL is hit.

    This approach uses the trained model to predict tomorrow's price, then uses
    that prediction as input to predict the next day, and so on. It simulates
    the most likely price path according to the model.

    Args:
        model: Trained model (LightGBM, XGBoost, or LSTM)
        scaler: Fitted StandardScaler for feature normalization
        df: Historical price dataframe
        feature_cols: List of feature column names
        current_price: Today's closing price
        stop_loss: Stop loss price level
        take_profit: Take profit price level
        n_days: Number of days to predict (default: 5)
        model_type: 'gbm' for LightGBM/XGBoost, 'lstm' for LSTM

    Returns:
        dict with:
            - hit_tp: Boolean, True if TP hit before SL
            - hit_day: Which day (1-5) the target was hit
            - predicted_path: List of predicted prices for each day
            - reason: 'TP' or 'SL' or 'NONE'
    """
    predicted_path = []
    df_sim = df.copy()

    # Verify all feature columns exist in the dataframe
    missing_cols = [col for col in feature_cols if col not in df_sim.columns]
    if missing_cols:
        print(f"ERROR: DataFrame missing required features: {missing_cols}")
        print(f"Available columns: {list(df_sim.columns)}")
        return None

    for day in range(1, n_days + 1):
        if model_type == 'lstm':
            # For LSTM, need sequence of recent prices
            recent_data = df_sim[feature_cols].tail(60).values
            recent_data_scaled = scaler.transform(recent_data)
            X_input = recent_data_scaled.reshape(1, 60, len(feature_cols))
            pred_scaled = model.predict(X_input, verbose=0)[0][0]
            # Inverse transform prediction
            dummy = np.zeros((1, len(feature_cols)))
            dummy[0, 0] = pred_scaled
            pred_price = scaler.inverse_transform(dummy)[0, 0]
        else:
            # For GBM models (LightGBM/XGBoost)
            # Make sure all feature columns exist
            missing_cols = [col for col in feature_cols if col not in df_sim.columns]
            if missing_cols:
                print(f"Warning: Missing columns: {missing_cols}")
                # Fill missing columns with last valid value
                for col in missing_cols:
                    df_sim[col] = df_sim['Close']

            last_row = df_sim[feature_cols].iloc[-1:].values
            last_row_scaled = scaler.transform(last_row)
            pred_price = model.predict(last_row_scaled)[0]

        predicted_path.append(pred_price)

        # Check if TP or SL is hit
        if stop_loss < take_profit:  # LONG position
            if pred_price >= take_profit:
                return {
                    'hit_tp': True,
                    'hit_day': day,
                    'predicted_path': predicted_path,
                    'reason': 'TP'
                }
            elif pred_price <= stop_loss:
                return {
                    'hit_tp': False,
                    'hit_day': day,
                    'predicted_path': predicted_path,
                    'reason': 'SL'
                }
        else:  # SHORT position
            if pred_price <= take_profit:
                return {
                    'hit_tp': True,
                    'hit_day': day,
                    'predicted_path': predicted_path,
                    'reason': 'TP'
                }
            elif pred_price >= stop_loss:
                return {
                    'hit_tp': False,
                    'hit_day': day,
                    'predicted_path': predicted_path,
                    'reason': 'SL'
                }

        # Add predicted price to dataframe for next iteration
        new_row = df_sim.iloc[-1:].copy()
        new_row['Close'] = pred_price
        new_row['Open'] = pred_price
        new_row['High'] = pred_price * 1.01  # Estimate
        new_row['Low'] = pred_price * 0.99  # Estimate
        # Keep volume from last day
        new_row['Volume'] = df_sim.iloc[-1]['Volume']
        df_sim = pd.concat([df_sim, new_row], ignore_index=True)

        # Recalculate ALL features for next prediction
        df_sim = recalculate_features(df_sim, feature_cols)

        # Make sure we have all required features
        for col in feature_cols:
            if col not in df_sim.columns:
                df_sim[col] = df_sim['Close']  # Fallback

    # Neither TP nor SL hit in n_days
    return {
        'hit_tp': None,
        'hit_day': None,
        'predicted_path': predicted_path,
        'reason': 'NONE'
    }


def recalculate_features(df, feature_cols):
    """
    Recalculate technical indicators after adding new predicted price.
    This ensures features stay consistent for multi-day prediction.
    """
    # Recalculate price change features
    if 'Price_change_1d' in feature_cols:
        df['Price_change_1d'] = df['Close'].pct_change(1) * 100
    if 'Price_change_5d' in feature_cols:
        df['Price_change_5d'] = df['Close'].pct_change(5) * 100
    if 'Price_change_10d' in feature_cols:
        df['Price_change_10d'] = df['Close'].pct_change(10) * 100

    # Volatility features
    if 'Volatility_5d' in feature_cols:
        df['Volatility_5d'] = df['Close'].rolling(window=5).std()
    if 'Volatility_10d' in feature_cols:
        df['Volatility_10d'] = df['Close'].rolling(window=10).std()

    # Lag features for Close
    for lag in [1, 2, 3, 5, 10]:
        col_name = f'Close_lag_{lag}'
        if col_name in feature_cols:
            df[col_name] = df['Close'].shift(lag)

    # Lag features for Volume
    for lag in [1, 2, 3, 5, 10]:
        col_name = f'Volume_lag_{lag}'
        if col_name in feature_cols:
            df[col_name] = df['Volume'].shift(lag)

    # Fill NaN with forward fill then backward fill
    df = df.ffill().bfill()

    return df


# ============================================================================
# APPROACH 2: MONTE CARLO SIMULATION
# ============================================================================

def monte_carlo_simulation(current_price, stop_loss, take_profit, volatility,
                           predicted_move_pct, n_days=PREDICTION_DAYS,
                           n_simulations=MONTE_CARLO_SIMULATIONS):
    """
    Run Monte Carlo simulations to estimate probability of hitting TP before SL.

    This approach generates thousands of random price paths based on historical
    volatility, with a slight bias toward the predicted direction. It's like
    asking "given the market's typical randomness, how often does this trade work?"

    Algorithm:
    1. Each day, price changes by: (drift + random_shock)
    2. Drift = predicted_move_pct * MC_DRIFT_INFLUENCE (trend bias)
    3. Random shock = normal(0, daily_volatility) (market randomness)
    4. Stop when TP or SL is hit, or n_days reached

    Args:
        current_price: Starting price
        stop_loss: Stop loss price level
        take_profit: Take profit price level
        volatility: Daily volatility (standard deviation of returns)
        predicted_move_pct: Model's predicted move direction (%)
        n_days: Max days to simulate (default: 5)
        n_simulations: Number of simulations to run (default: 1000)

    Returns:
        dict with:
            - win_rate: Percentage of sims that hit TP before SL
            - tp_count: Number of sims that hit TP
            - sl_count: Number of sims that hit SL
            - no_hit_count: Number of sims that hit neither
            - avg_days_to_tp: Average days to hit TP
            - avg_days_to_sl: Average days to hit SL
    """
    tp_count = 0
    sl_count = 0
    no_hit_count = 0
    tp_days = []
    sl_days = []

    # Calculate drift (trend bias)
    daily_drift = (predicted_move_pct / 100) * MC_DRIFT_INFLUENCE / n_days
    daily_volatility = volatility / current_price  # Convert to percentage

    is_long = stop_loss < take_profit

    for _ in range(n_simulations):
        price = current_price

        for day in range(1, n_days + 1):
            # Generate random price change
            random_return = np.random.normal(daily_drift, daily_volatility)
            price = price * (1 + random_return)

            # Check if TP or SL is hit
            if is_long:
                if price >= take_profit:
                    tp_count += 1
                    tp_days.append(day)
                    break
                elif price <= stop_loss:
                    sl_count += 1
                    sl_days.append(day)
                    break
            else:  # SHORT
                if price <= take_profit:
                    tp_count += 1
                    tp_days.append(day)
                    break
                elif price >= stop_loss:
                    sl_count += 1
                    sl_days.append(day)
                    break
        else:
            # Neither hit
            no_hit_count += 1

    win_rate = (tp_count / n_simulations) * 100

    return {
        'win_rate': win_rate,
        'tp_count': tp_count,
        'sl_count': sl_count,
        'no_hit_count': no_hit_count,
        'avg_days_to_tp': np.mean(tp_days) if tp_days else None,
        'avg_days_to_sl': np.mean(sl_days) if sl_days else None
    }


# ============================================================================
# APPROACH 3: HISTORICAL PATTERN MATCHING
# ============================================================================

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def find_similar_patterns(df, current_price, stop_loss, take_profit,
                         lookback=PATTERN_LOOKBACK, n_matches=PATTERN_MATCH_COUNT):
    """
    Find historical patterns similar to current market conditions.

    This approach searches the past to find situations that looked like today:
    - Similar RSI (momentum)
    - Similar volatility (market chaos level)
    - Similar trend direction

    Then checks: "What happened in the next 5 days? Did TP or SL get hit?"

    Similarity scoring:
    - RSI difference (±5 points is "similar")
    - Volatility ratio (within ±20% is "similar")
    - Recent trend direction (up/down/sideways)

    Args:
        df: Historical price dataframe
        current_price: Today's price
        stop_loss: Stop loss level
        take_profit: Take profit level
        lookback: How many days to search back (default: 200)
        n_matches: How many similar patterns to find (default: 50)

    Returns:
        dict with:
            - win_rate: % of similar patterns that hit TP before SL
            - tp_count: Number that hit TP
            - sl_count: Number that hit SL
            - no_hit_count: Number that hit neither
            - matches_found: Actual number of patterns found
    """
    if len(df) < lookback + 10:
        return None

    # Calculate current market conditions
    current_rsi = calculate_rsi(df['Close']).iloc[-1]
    recent_volatility = df['Close'].pct_change().tail(20).std()
    recent_trend = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]

    is_long = stop_loss < take_profit
    tp_count = 0
    sl_count = 0
    no_hit_count = 0
    matches_found = 0

    # Search through historical data
    search_start = max(50, len(df) - lookback)  # Need enough history for RSI
    search_end = len(df) - PREDICTION_DAYS - 1  # Need future data to check

    similarities = []

    for i in range(search_start, search_end):
        # Calculate similarity to current conditions
        hist_rsi = calculate_rsi(df['Close'].iloc[:i+1]).iloc[-1]
        hist_volatility = df['Close'].iloc[i-19:i+1].pct_change().std()
        hist_trend = (df['Close'].iloc[i] - df['Close'].iloc[i-5]) / df['Close'].iloc[i-5]

        # Skip if NaN
        if pd.isna(hist_rsi) or pd.isna(hist_volatility):
            continue

        # Calculate similarity score
        rsi_diff = abs(hist_rsi - current_rsi)
        vol_ratio = hist_volatility / recent_volatility if recent_volatility > 0 else 0
        trend_same = (hist_trend * recent_trend) > 0  # Same direction

        # Filter by similarity
        if rsi_diff <= RSI_TOLERANCE and \
           (1 - VOLATILITY_TOLERANCE) <= vol_ratio <= (1 + VOLATILITY_TOLERANCE) and \
           trend_same:

            # Calculate similarity score (lower is better)
            score = rsi_diff + abs(1 - vol_ratio) * 10
            similarities.append((i, score))

    # Get top N most similar patterns
    similarities.sort(key=lambda x: x[1])
    top_matches = similarities[:n_matches]
    matches_found = len(top_matches)

    if matches_found == 0:
        return None

    # Check outcomes of similar patterns
    for idx, _ in top_matches:
        hist_price = df['Close'].iloc[idx]
        hist_sl = stop_loss / current_price * hist_price
        hist_tp = take_profit / current_price * hist_price

        # Check next PREDICTION_DAYS days
        hit_tp = False
        hit_sl = False

        for day in range(1, PREDICTION_DAYS + 1):
            future_price = df['Close'].iloc[idx + day]

            if is_long:
                if future_price >= hist_tp:
                    hit_tp = True
                    break
                elif future_price <= hist_sl:
                    hit_sl = True
                    break
            else:  # SHORT
                if future_price <= hist_tp:
                    hit_tp = True
                    break
                elif future_price >= hist_sl:
                    hit_sl = True
                    break

        if hit_tp:
            tp_count += 1
        elif hit_sl:
            sl_count += 1
        else:
            no_hit_count += 1

    win_rate = (tp_count / matches_found) * 100 if matches_found > 0 else 0

    return {
        'win_rate': win_rate,
        'tp_count': tp_count,
        'sl_count': sl_count,
        'no_hit_count': no_hit_count,
        'matches_found': matches_found
    }


# ============================================================================
# ENSEMBLE: COMBINE ALL APPROACHES
# ============================================================================

def calculate_ensemble_probability(prediction_result, monte_carlo_result,
                                   pattern_result):
    """
    Combine results from all three approaches into final probability.

    Uses weighted average of all three methods. If a method fails or returns
    None, it's excluded and weights are redistributed.

    Args:
        prediction_result: Result from multi-day prediction
        monte_carlo_result: Result from Monte Carlo simulation
        pattern_result: Result from historical pattern matching

    Returns:
        dict with:
            - ensemble_probability: Final weighted probability (%)
            - confidence_level: 'HIGH', 'MEDIUM', or 'LOW'
            - recommendation: 'TAKE TRADE' or 'SKIP TRADE'
            - contributing_methods: List of methods used
    """
    probabilities = []
    weights = []
    methods = []

    # Add prediction probability
    if prediction_result and prediction_result.get('hit_tp') is not None:
        prob = 100.0 if prediction_result['hit_tp'] else 0.0
        probabilities.append(prob)
        weights.append(WEIGHT_PREDICTION)
        methods.append('Multi-Day Prediction')

    # Add Monte Carlo probability
    if monte_carlo_result:
        probabilities.append(monte_carlo_result['win_rate'])
        weights.append(WEIGHT_MONTE_CARLO)
        methods.append('Monte Carlo')

    # Add pattern matching probability
    if pattern_result:
        probabilities.append(pattern_result['win_rate'])
        weights.append(WEIGHT_PATTERN)
        methods.append('Historical Patterns')

    if not probabilities:
        return None

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Calculate weighted average
    ensemble_prob = sum(p * w for p, w in zip(probabilities, weights))

    # Determine confidence level
    if ensemble_prob >= CONFIDENCE_HIGH:
        confidence = 'HIGH'
    elif ensemble_prob >= CONFIDENCE_MEDIUM:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'

    # Make recommendation
    recommendation = 'TAKE TRADE' if ensemble_prob >= MIN_CONFIDENCE_THRESHOLD else 'SKIP TRADE'

    return {
        'ensemble_probability': ensemble_prob,
        'confidence_level': confidence,
        'recommendation': recommendation,
        'contributing_methods': methods,
        'individual_probabilities': dict(zip(methods, probabilities))
    }


def format_analysis_report(prediction_result, monte_carlo_result, pattern_result,
                           ensemble_result, signal, current_price, stop_loss,
                           take_profit):
    """
    Format a detailed text report of the probability analysis.

    This creates the text output that will be printed to console and
    included in the HTML report.

    Returns:
        str: Formatted report text
    """
    report = []
    report.append("\n" + "="*70)
    report.append("TRADE PROBABILITY ANALYSIS")
    report.append("="*70)

    # Current setup
    report.append(f"\nCurrent Price: ${current_price:.2f}")
    report.append(f"Signal: {signal}")
    report.append(f"Stop Loss: ${stop_loss:.2f} ({((stop_loss - current_price) / current_price * 100):+.2f}%)")
    report.append(f"Take Profit: ${take_profit:.2f} ({((take_profit - current_price) / current_price * 100):+.2f}%)")

    report.append(f"\n{'-'*70}")
    report.append("APPROACH 1: Multi-Day Sequential Prediction")
    report.append(f"{'-'*70}")

    if prediction_result:
        if prediction_result['reason'] == 'TP':
            report.append(f"[WIN] Predicts TAKE PROFIT hit on Day {prediction_result['hit_day']}")
            report.append(f"   Win Probability: 100%")
        elif prediction_result['reason'] == 'SL':
            report.append(f"[LOSS] Predicts STOP LOSS hit on Day {prediction_result['hit_day']}")
            report.append(f"   Win Probability: 0%")
        else:
            report.append(f"[NEUTRAL] Neither TP nor SL hit in {PREDICTION_DAYS} days")
            report.append(f"   Win Probability: N/A")

        report.append(f"\n   Predicted Path ({PREDICTION_DAYS}-day):")
        for i, price in enumerate(prediction_result['predicted_path'], 1):
            marker = ""
            if prediction_result['reason'] == 'TP' and i == prediction_result['hit_day']:
                marker = " <- HIT TAKE PROFIT"
            elif prediction_result['reason'] == 'SL' and i == prediction_result['hit_day']:
                marker = " <- HIT STOP LOSS"
            report.append(f"   Day {i}: ${price:.2f}{marker}")
    else:
        report.append("[ERROR] Prediction failed")

    report.append(f"\n{'-'*70}")
    report.append(f"APPROACH 2: Monte Carlo Simulation ({MONTE_CARLO_SIMULATIONS} runs)")
    report.append(f"{'-'*70}")

    if monte_carlo_result:
        report.append(f"Win Probability: {monte_carlo_result['win_rate']:.1f}%")
        report.append(f"  • {monte_carlo_result['tp_count']} simulations hit Take Profit")
        report.append(f"  • {monte_carlo_result['sl_count']} simulations hit Stop Loss")
        report.append(f"  • {monte_carlo_result['no_hit_count']} simulations hit neither")

        if monte_carlo_result['avg_days_to_tp']:
            report.append(f"  • Avg days to TP: {monte_carlo_result['avg_days_to_tp']:.1f}")
        if monte_carlo_result['avg_days_to_sl']:
            report.append(f"  • Avg days to SL: {monte_carlo_result['avg_days_to_sl']:.1f}")
    else:
        report.append("[ERROR] Simulation failed")

    report.append(f"\n{'-'*70}")
    report.append("APPROACH 3: Historical Pattern Matching")
    report.append(f"{'-'*70}")

    if pattern_result:
        report.append(f"Win Probability: {pattern_result['win_rate']:.1f}%")
        report.append(f"  • Found {pattern_result['matches_found']} similar historical setups")
        report.append(f"  • {pattern_result['tp_count']} times TP was hit first")
        report.append(f"  • {pattern_result['sl_count']} times SL was hit first")
        report.append(f"  • {pattern_result['no_hit_count']} times neither was hit")
    else:
        report.append("[ERROR] No similar patterns found (insufficient historical data)")

    report.append(f"\n{'='*70}")
    report.append("ENSEMBLE DECISION")
    report.append(f"{'='*70}")

    if ensemble_result:
        report.append(f"\nENSEMBLE WIN PROBABILITY: {ensemble_result['ensemble_probability']:.1f}%")
        report.append(f"   Confidence Level: {ensemble_result['confidence_level']}")
        report.append(f"   Recommendation: {ensemble_result['recommendation']}")

        report.append(f"\n   Contributing Methods:")
        for method, prob in ensemble_result['individual_probabilities'].items():
            report.append(f"   • {method}: {prob:.1f}%")

        if ensemble_result['recommendation'] == 'TAKE TRADE':
            report.append(f"\n[RECOMMENDED] {signal}")
            report.append(f"   This trade has {ensemble_result['ensemble_probability']:.1f}% probability of hitting")
            report.append(f"   Take Profit before Stop Loss in the next {PREDICTION_DAYS} days.")
        else:
            report.append(f"\n[WARNING] SKIP THIS TRADE")
            report.append(f"   Probability ({ensemble_result['ensemble_probability']:.1f}%) is below threshold ({MIN_CONFIDENCE_THRESHOLD}%).")
            report.append(f"   Wait for a better setup.")
    else:
        report.append("\n[ERROR] Ensemble analysis failed - not enough data")

    report.append(f"\n{'='*70}\n")

    return "\n".join(report)
