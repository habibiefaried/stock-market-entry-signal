"""
Main script to train all models and generate comparison report

This script runs all 3 prediction models (LSTM, XGBoost, LightGBM) in parallel
and generates a comprehensive HTML report comparing their performance and trading signals.

Usage:
    # Fetch data and train models
    python main.py --ticker MSFT

    # Train with existing CSV
    python main.py MSFT_daily_data_20260520.csv
"""

import subprocess
import argparse
import os
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import warnings
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)

def fetch_stock_data(ticker, months=48):
    """
    Fetch stock data using yfinance

    Args:
        ticker (str): Stock ticker symbol (e.g., "MSFT", "BTC-USD")
        months (int): Number of months of historical data (default: 48 = 4 years)

    Returns:
        str: Path to saved CSV file
    """
    try:
        import yfinance as yf
        import pandas as pd

        print(f"Fetching {ticker} data ({months} months = {months/12:.1f} years)...")

        # Create ticker object
        ticker_obj = yf.Ticker(ticker)

        # Get historical data with daily interval
        df = ticker_obj.history(period=f"{months}mo", interval="1d")

        if df.empty:
            print(f"Error: No data found for {ticker}")
            sys.exit(1)

        # Reset index to make Date a column
        df.reset_index(inplace=True)

        # Keep only OHLCV columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Save to CSV
        filename = f"{ticker.replace('-', '_')}_daily_data_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(filename, index=False)

        print(f"Data saved to: {filename}")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

        return filename
    except ImportError:
        print("Error: yfinance not installed. Install with: pip install yfinance")
        sys.exit(1)
    except Exception as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)

def run_model(model_name, script_name, csv_file):
    """
    Run a single model training script and capture output

    Args:
        model_name (str): Display name of the model (e.g., "ARIMA")
        script_name (str): Python script filename (e.g., "train_arima.py")
        csv_file (str): Path to CSV data file

    Returns:
        dict: Results containing model name, success status, and output
    """
    print(f"[{model_name}] Starting training...")

    try:
        # Get absolute path to script
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)

        if not os.path.exists(script_path):
            print(f"[{model_name}] [FAIL] Script not found: {script_path}")
            return {
                'name': model_name,
                'success': False,
                'output': None,
                'error': f"Script not found: {script_path}"
            }

        # Run the training script and capture output
        result = subprocess.run(
            [sys.executable, script_path, csv_file],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=os.path.dirname(os.path.abspath(__file__))  # Set working directory
        )

        if result.returncode == 0:
            print(f"[{model_name}] [OK] Training completed successfully")
            return {
                'name': model_name,
                'success': True,
                'output': result.stdout,
                'error': None
            }
        else:
            print(f"[{model_name}] [FAIL] Training failed (exit code: {result.returncode})")
            # Print first few lines of error for debugging
            error_preview = result.stderr[:500] if result.stderr else "No error output"
            print(f"[{model_name}] Error preview: {error_preview}")
            return {
                'name': model_name,
                'success': False,
                'output': result.stdout,
                'error': result.stderr if result.stderr else "Unknown error - check output"
            }
    except subprocess.TimeoutExpired:
        print(f"[{model_name}] [FAIL] Training timed out (>10 minutes)")
        return {
            'name': model_name,
            'success': False,
            'output': None,
            'error': "Training timed out after 10 minutes"
        }
    except Exception as e:
        print(f"[{model_name}] [FAIL] Error: {e}")
        import traceback
        return {
            'name': model_name,
            'success': False,
            'output': None,
            'error': f"{str(e)}\n{traceback.format_exc()}"
        }

def parse_metrics(output):
    """
    Parse metrics from model output

    Args:
        output (str): Captured stdout from training script

    Returns:
        dict: Parsed metrics (MAE, RMSE, accuracy, etc.)
    """
    metrics = {}

    # Parse test metrics
    mae_match = re.search(r'Test MAE:\s+\$?([\d.]+)', output)
    if mae_match:
        metrics['test_mae'] = float(mae_match.group(1))

    rmse_match = re.search(r'Test RMSE:\s+\$?([\d.]+)', output)
    if rmse_match:
        metrics['test_rmse'] = float(rmse_match.group(1))

    acc_match = re.search(r'Test Accuracy:\s+([\d.]+)%', output)
    if acc_match:
        metrics['test_accuracy'] = float(acc_match.group(1))

    prec_match = re.search(r'Test Precision:\s+([\d.]+)%', output)
    if prec_match:
        metrics['test_precision'] = float(prec_match.group(1))

    rec_match = re.search(r'Test Recall:\s+([\d.]+)%', output)
    if rec_match:
        metrics['test_recall'] = float(rec_match.group(1))

    f1_match = re.search(r'Test F1-Score:\s+([\d.]+)%', output)
    if f1_match:
        metrics['test_f1'] = float(f1_match.group(1))

    return metrics

def parse_trading_signal(output):
    """
    Parse trading signal from model output

    Args:
        output (str): Captured stdout from training script

    Returns:
        dict: Parsed trading signal information
    """
    signal = {}

    # Parse signal
    signal_match = re.search(r'SIGNAL:\s+([A-Za-z ()\t]+?)(?:\n|$)', output)
    if signal_match:
        signal['signal'] = signal_match.group(1).strip()

    # Parse current price
    current_match = re.search(r'Current Price \(Today\):\s+\$?([\d,]+\.?\d*)', output)
    if current_match:
        signal['current_price'] = float(current_match.group(1).replace(',', ''))

    # Parse predicted price
    pred_match = re.search(r'Predicted Price \(Tomorrow\):\s+\$?([\d,]+\.?\d*)', output)
    if pred_match:
        signal['predicted_price'] = float(pred_match.group(1).replace(',', ''))

    # Parse expected move
    move_match = re.search(r'Expected Move:\s+\$?([+-]?[\d,]+\.?\d*)\s+\(([+-]?[\d.]+)%\)', output)
    if move_match:
        signal['expected_move'] = float(move_match.group(1).replace(',', ''))
        signal['expected_move_pct'] = float(move_match.group(2))

    # Parse stop loss
    sl_match = re.search(r'Stop Loss:\s+\$?([\d,]+\.?\d*)\s+\(([+-]?[\d.]+)%\)', output)
    if sl_match:
        signal['stop_loss'] = float(sl_match.group(1).replace(',', ''))
        signal['stop_loss_pct'] = float(sl_match.group(2))

    # Parse take profit
    tp_match = re.search(r'Take Profit:\s+\$?([\d,]+\.?\d*)\s+\(([+-]?[\d.]+)%\)', output)
    if tp_match:
        signal['take_profit'] = float(tp_match.group(1).replace(',', ''))
        signal['take_profit_pct'] = float(tp_match.group(2))

    # Parse confidence
    conf_match = re.search(r'Model Confidence:\s+([\d.]+)%', output)
    if conf_match:
        signal['confidence'] = float(conf_match.group(1))

    # Parse volatility
    vol_match = re.search(r'Recent Volatility:\s+\$?([\d,]+\.?\d*)', output)
    if vol_match:
        signal['volatility'] = float(vol_match.group(1).replace(',', ''))

    # Parse probability analysis results
    ensemble_prob_match = re.search(r'ENSEMBLE_PROBABILITY:\s+([\d.]+)%', output)
    if ensemble_prob_match:
        signal['ensemble_probability'] = float(ensemble_prob_match.group(1))

    confidence_level_match = re.search(r'CONFIDENCE_LEVEL:\s+(\w+)', output)
    if confidence_level_match:
        signal['confidence_level'] = confidence_level_match.group(1)

    recommendation_match = re.search(r'RECOMMENDATION:\s+(.+?)(?:\n|$)', output)
    if recommendation_match:
        signal['recommendation'] = recommendation_match.group(1).strip()

    # Parse individual approach results from the detailed analysis
    # Approach 1: Multi-Day Prediction
    pred_win_match = re.search(r'\[WIN\] Predicts TAKE PROFIT hit on Day (\d+)', output)
    pred_loss_match = re.search(r'\[LOSS\] Predicts STOP LOSS hit on Day (\d+)', output)
    pred_neutral_match = re.search(r'\[NEUTRAL\] Neither TP nor SL hit', output)

    if pred_win_match:
        signal['approach1_result'] = 'WIN'
        signal['approach1_probability'] = 100.0
        signal['approach1_day'] = int(pred_win_match.group(1))
    elif pred_loss_match:
        signal['approach1_result'] = 'LOSS'
        signal['approach1_probability'] = 0.0
        signal['approach1_day'] = int(pred_loss_match.group(1))
    elif pred_neutral_match:
        signal['approach1_result'] = 'NEUTRAL'
        signal['approach1_probability'] = None

    # Parse predicted path
    path_match = re.findall(r'Day (\d+): \$(\d+\.?\d*)', output)
    if path_match:
        signal['predicted_path'] = [(int(day), float(price)) for day, price in path_match]

    # Approach 2: Monte Carlo
    mc_win_match = re.search(r'Win Probability: ([\d.]+)%.*?(\d+) simulations hit Take Profit.*?(\d+) simulations hit Stop Loss', output, re.DOTALL)
    if mc_win_match:
        signal['approach2_probability'] = float(mc_win_match.group(1))
        signal['approach2_tp_count'] = int(mc_win_match.group(2))
        signal['approach2_sl_count'] = int(mc_win_match.group(3))

    # Approach 3: Historical Patterns
    pattern_win_match = re.search(r'Found (\d+) similar historical setups.*?(\d+) times TP was hit first.*?(\d+) times SL was hit first', output, re.DOTALL)
    if pattern_win_match:
        signal['approach3_matches'] = int(pattern_win_match.group(1))
        signal['approach3_tp_count'] = int(pattern_win_match.group(2))
        signal['approach3_sl_count'] = int(pattern_win_match.group(3))
        total = signal['approach3_tp_count'] + signal['approach3_sl_count']
        if total > 0:
            signal['approach3_probability'] = (signal['approach3_tp_count'] / total) * 100

    return signal

def generate_probability_html(signal):
    """Generate beautiful HTML for probability analysis section"""
    if 'ensemble_probability' not in signal:
        return ""

    prob = signal['ensemble_probability']
    conf_level = signal.get('confidence_level', 'LOW')
    recommendation = signal.get('recommendation', 'N/A')

    # Determine probability circle class
    if prob >= 75:
        prob_class = 'prob-high'
    elif prob >= 65:
        prob_class = 'prob-medium'
    else:
        prob_class = 'prob-low'

    # Determine confidence badge class
    if conf_level == 'HIGH':
        conf_class = 'conf-high'
    elif conf_level == 'MEDIUM':
        conf_class = 'conf-medium'
    else:
        conf_class = 'conf-low'

    # Determine recommendation badge class
    rec_class = 'rec-take' if 'TAKE' in recommendation else 'rec-skip'
    rec_text = '✓ TAKE TRADE' if 'TAKE' in recommendation else '✗ SKIP TRADE'

    html = []
    html.append('<div class="probability-section">')
    html.append('    <div class="probability-header">')
    html.append('        <h3 class="probability-title">🎯 Multi-Approach Win Probability Analysis</h3>')
    html.append('    </div>')
    html.append('    <div class="ensemble-result">')
    html.append(f'        <div class="probability-circle {prob_class}">')
    html.append(f'            <div class="probability-value">{prob:.1f}%</div>')
    html.append('            <div class="probability-label">Win Probability</div>')
    html.append('        </div>')
    html.append(f'        <div class="confidence-badge {conf_class}">{conf_level} CONFIDENCE</div>')
    html.append(f'        <div class="recommendation-badge {rec_class}">{rec_text}</div>')
    html.append('    </div>')

    # Generate individual approach cards
    html.append('    <div class="approaches-grid">')

    # Approach 1: Multi-Day Prediction
    html.append('        <div class="approach-card">')
    html.append('            <div class="approach-header">📈 Approach 1: Multi-Day Prediction</div>')

    if 'approach1_probability' in signal and signal['approach1_probability'] is not None:
        html.append(f'            <div class="approach-probability">{signal["approach1_probability"]:.1f}%</div>')
        html.append('            <div class="approach-details">')

        if signal.get('approach1_result') == 'WIN':
            html.append(f'                <strong>Result:</strong> Predicts TP hit on Day {signal.get("approach1_day", "?")}<br>')
        elif signal.get('approach1_result') == 'LOSS':
            html.append(f'                <strong>Result:</strong> Predicts SL hit on Day {signal.get("approach1_day", "?")}<br>')
        else:
            html.append('                <strong>Result:</strong> Neither TP nor SL hit in 5 days<br>')

        # Show predicted path if available
        if 'predicted_path' in signal and signal['predicted_path']:
            html.append('                <br><strong>Predicted Path (5-day):</strong>')
            html.append('                <table class="path-table">')
            for day, price in signal['predicted_path'][:5]:  # Show max 5 days
                marker = ''
                if signal.get('approach1_result') == 'WIN' and day == signal.get('approach1_day'):
                    marker = '<span class="hit-marker">TP HIT</span>'
                elif signal.get('approach1_result') == 'LOSS' and day == signal.get('approach1_day'):
                    marker = '<span class="hit-marker" style="background:#ee0979;">SL HIT</span>'
                html.append(f'                    <tr><td>Day {day}:</td><td>${price:.2f}</td><td>{marker}</td></tr>')
            html.append('                </table>')
        html.append('            </div>')
    else:
        html.append('            <div class="approach-details">No prediction data available</div>')

    html.append('        </div>')

    # Approach 2: Monte Carlo
    html.append('        <div class="approach-card">')
    html.append('            <div class="approach-header">🎲 Approach 2: Monte Carlo (1000 runs)</div>')

    if 'approach2_probability' in signal:
        html.append(f'            <div class="approach-probability">{signal["approach2_probability"]:.1f}%</div>')
        html.append('            <div class="approach-details">')
        html.append(f'                <strong>{signal.get("approach2_tp_count", 0)}</strong> simulations hit Take Profit<br>')
        html.append(f'                <strong>{signal.get("approach2_sl_count", 0)}</strong> simulations hit Stop Loss<br>')
        html.append(f'                <strong>{1000 - signal.get("approach2_tp_count", 0) - signal.get("approach2_sl_count", 0)}</strong> simulations hit neither<br>')
        html.append('                <br>Uses historical volatility + predicted trend to generate random price paths')
        html.append('            </div>')
    else:
        html.append('            <div class="approach-details">No Monte Carlo data available</div>')

    html.append('        </div>')

    # Approach 3: Historical Patterns
    html.append('        <div class="approach-card">')
    html.append('            <div class="approach-header">📊 Approach 3: Historical Patterns</div>')

    if 'approach3_probability' in signal:
        html.append(f'            <div class="approach-probability">{signal["approach3_probability"]:.1f}%</div>')
        html.append('            <div class="approach-details">')
        html.append(f'                Found <strong>{signal.get("approach3_matches", 0)}</strong> similar historical setups<br>')
        html.append(f'                <strong>{signal.get("approach3_tp_count", 0)}</strong> times TP was hit first<br>')
        html.append(f'                <strong>{signal.get("approach3_sl_count", 0)}</strong> times SL was hit first<br>')
        html.append('                <br>Matches current RSI, volatility, and trend direction to past setups')
        html.append('            </div>')
    else:
        html.append('            <div class="approach-details">No historical pattern data available</div>')

    html.append('        </div>')

    html.append('    </div>')  # Close approaches-grid
    html.append('</div>')  # Close probability-section

    return '\n'.join(html)

def generate_html_report(results, csv_file, output_file):
    """
    Generate HTML report comparing all models

    Args:
        results (list): List of result dictionaries from each model
        csv_file (str): Path to CSV data file
        output_file (str): Path to output HTML file
    """
    # Extract stock name and date from CSV filename
    # Format: TICKER_daily_data_YYYYMMDD.csv
    basename = os.path.basename(csv_file)
    parts = basename.split('_')
    ticker = parts[0]

    # Start HTML content
    html = []
    html.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Price Prediction Report: """ + ticker + """</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .header .ticker-badge {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 8px 20px;
            border-radius: 20px;
            font-size: 1.2em;
            font-weight: 600;
            margin-top: 10px;
        }

        .content {
            padding: 40px;
        }

        .info-bar {
            display: flex;
            justify-content: space-around;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }

        .info-item {
            text-align: center;
            padding: 10px;
        }

        .info-label {
            font-size: 0.9em;
            color: #6c757d;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .info-value {
            font-size: 1.3em;
            color: #2c3e50;
            font-weight: 700;
            margin-top: 5px;
        }

        .section {
            margin-bottom: 40px;
        }

        .section-title {
            font-size: 1.8em;
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
            font-weight: 700;
        }

        .model-card {
            background: white;
            border: 2px solid #e9ecef;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.07);
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .model-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.15);
        }

        .model-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
            flex-wrap: wrap;
        }

        .model-name {
            font-size: 1.8em;
            font-weight: 700;
            color: #2c3e50;
        }

        .signal-badge {
            padding: 10px 25px;
            border-radius: 25px;
            font-weight: 700;
            font-size: 1.1em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .signal-buy {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
        }

        .signal-short {
            background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
            color: white;
        }

        .signal-hold {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }

        .stat-box {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid #667eea;
        }

        .stat-label {
            font-size: 0.85em;
            color: #6c757d;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .stat-value {
            font-size: 1.6em;
            font-weight: 700;
            color: #2c3e50;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 0.5px;
        }

        td {
            padding: 15px;
            border-bottom: 1px solid #e9ecef;
        }

        tr:hover {
            background-color: #f8f9fa;
        }

        tr:last-child td {
            border-bottom: none;
        }

        .probability-section {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 12px;
            padding: 30px;
            margin-top: 25px;
        }

        .probability-header {
            text-align: center;
            margin-bottom: 30px;
        }

        .probability-title {
            font-size: 1.5em;
            color: #2c3e50;
            margin-bottom: 15px;
            font-weight: 700;
        }

        .ensemble-result {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 30px;
            flex-wrap: wrap;
            margin-bottom: 30px;
        }

        .probability-circle {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            color: white;
            font-weight: 700;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }

        .prob-high {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }

        .prob-medium {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }

        .prob-low {
            background: linear-gradient(135deg, #868f96 0%, #596164 100%);
        }

        .probability-value {
            font-size: 2.5em;
        }

        .probability-label {
            font-size: 0.9em;
            margin-top: 5px;
        }

        .confidence-badge {
            padding: 12px 30px;
            border-radius: 25px;
            font-weight: 700;
            font-size: 1.1em;
            color: white;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .conf-high {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }

        .conf-medium {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }

        .conf-low {
            background: linear-gradient(135deg, #868f96 0%, #596164 100%);
        }

        .recommendation-badge {
            padding: 15px 40px;
            border-radius: 30px;
            font-weight: 700;
            font-size: 1.2em;
            color: white;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        }

        .rec-take {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }

        .rec-skip {
            background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        }

        .approaches-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 25px;
        }

        .approach-card {
            background: white;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .approach-header {
            font-size: 1.2em;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }

        .approach-probability {
            font-size: 2.5em;
            font-weight: 700;
            color: #667eea;
            margin: 15px 0;
            text-align: center;
        }

        .approach-details {
            font-size: 0.95em;
            color: #6c757d;
            line-height: 1.8;
        }

        .path-table {
            margin-top: 15px;
            font-size: 0.9em;
        }

        .path-table td {
            padding: 8px;
        }

        .hit-marker {
            background: #38ef7d;
            color: white;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.85em;
            font-weight: 600;
        }

        .disclaimer {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            border-radius: 8px;
            margin: 30px 0;
        }

        .disclaimer h3 {
            color: #856404;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .disclaimer p {
            color: #856404;
            margin: 5px 0;
        }

        .footer {
            text-align: center;
            padding: 20px;
            color: #6c757d;
            font-size: 0.9em;
            border-top: 1px solid #e9ecef;
        }

        pre {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
        }
        .header-info {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .consensus {
            font-size: 1.2em;
            font-weight: bold;
            padding: 15px;
            border-radius: 4px;
            text-align: center;
            margin: 20px 0;
        }
        .consensus-buy {
            background-color: #d4edda;
            color: #155724;
            border: 2px solid #c3e6cb;
        }
        .consensus-short {
            background-color: #f8d7da;
            color: #721c24;
            border: 2px solid #f5c6cb;
        }
        .consensus-mixed {
            background-color: #fff3cd;
            color: #856404;
            border: 2px solid #ffeaa7;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Stock Price Prediction Report</h1>
            <div class="ticker-badge">""" + ticker + """</div>
        </div>
        <div class="content">
            <div class="info-bar">
                <div class="info-item">
                    <div class="info-label">Generated</div>
                    <div class="info-value">""" + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Data Source</div>
                    <div class="info-value">""" + os.path.basename(csv_file) + """</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Models Trained</div>
                    <div class="info-value">""" + str(len(results)) + """</div>
                </div>
            </div>
""")

    # Executive Summary
    html.append("<h2>Executive Summary</h2>")
    html.append("<div style='white-space: pre-wrap; font-family: monospace; background: #f4f4f4; padding: 15px; border-radius: 4px;'>")

    successful_models = [r for r in results if r['success']]
    failed_models = [r for r in results if not r['success']]

    html.append(f"<p><strong>Models Trained:</strong> {len(results)}</p>")
    html.append(f"<p><strong>Successful:</strong> <span class='badge badge-success'>{len(successful_models)}</span></p>")
    html.append(f"<p><strong>Failed:</strong> <span class='badge badge-danger'>{len(failed_models)}</span></p>")

    if failed_models:
        html.append("<h3>Failed Models</h3>")
        html.append("<ul>")
        for model in failed_models:
            html.append(f"<li><strong>{model['name']}:</strong> {model['error']}</li>")
        html.append("</ul>")

    # Model Performance Comparison
    if successful_models:
        html.append("<h2>Model Performance Comparison</h2>")

        # Parse metrics for all successful models
        model_metrics = []
        for result in successful_models:
            metrics = parse_metrics(result['output'])
            metrics['name'] = result['name']
            model_metrics.append(metrics)

        # Create comparison table
        html.append("### Regression Metrics (Price Prediction)\n")
        html.append("| Model | Test MAE ($) | Test RMSE ($) |")
        html.append("|-------|--------------|---------------|")

        for m in model_metrics:
            mae = f"${m.get('test_mae', 0):.2f}" if 'test_mae' in m else "N/A"
            rmse = f"${m.get('test_rmse', 0):.2f}" if 'test_rmse' in m else "N/A"
            html.append(f"| {m['name']} | {mae} | {rmse} |")

        html.append("\n*Lower is better*\n")

        # Classification metrics
        html.append("### Classification Metrics (Direction Prediction)\n")
        html.append("| Model | Accuracy | Precision | Recall | F1-Score |")
        html.append("|-------|----------|-----------|--------|----------|")

        for m in model_metrics:
            acc = f"{m.get('test_accuracy', 0):.1f}%" if 'test_accuracy' in m else "N/A"
            prec = f"{m.get('test_precision', 0):.1f}%" if 'test_precision' in m else "N/A"
            rec = f"{m.get('test_recall', 0):.1f}%" if 'test_recall' in m else "N/A"
            f1 = f"{m.get('test_f1', 0):.1f}%" if 'test_f1' in m else "N/A"
            html.append(f"| {m['name']} | {acc} | {prec} | {rec} | {f1} |")

        html.append("\n*Higher is better*\n")

        # Find best models
        best_mae_model = min(model_metrics, key=lambda x: x.get('test_mae', float('inf')))
        best_acc_model = max(model_metrics, key=lambda x: x.get('test_accuracy', 0))

        html.append("### Best Models:\n")
        html.append(f"- **Best Price Accuracy (Lowest MAE):** {best_mae_model['name']} (${best_mae_model.get('test_mae', 0):.2f})")
        html.append(f"- **Best Direction Accuracy:** {best_acc_model['name']} ({best_acc_model.get('test_accuracy', 0):.1f}%)\n")

    # Trading Signals
    if successful_models:
        html.append('<div class="section">')
        html.append('    <h2 class="section-title">📊 Trading Signals & Analysis</h2>')

        for result in successful_models:
            signal = parse_trading_signal(result['output'])

            if signal:
                # Determine signal class
                signal_text = signal.get('signal', 'UNKNOWN')
                if 'BUY' in signal_text:
                    signal_class = 'signal-buy'
                    signal_badge = f'<span class="signal-badge signal-buy">🚀 BUY (LONG)</span>'
                elif 'SHORT' in signal_text:
                    signal_class = 'signal-short'
                    signal_badge = f'<span class="signal-badge signal-short">📉 SHORT (SELL)</span>'
                else:
                    signal_class = 'signal-hold'
                    signal_badge = f'<span class="signal-badge signal-hold">⏸️ HOLD</span>'

                # Recommendation badge for header
                rec = signal.get('recommendation', '')
                if 'TAKE' in rec:
                    rec_badge = '<span class="signal-badge" style="background:linear-gradient(135deg,#27ae60,#2ecc71);color:#fff;">✓ TAKE TRADE</span>'
                elif 'SKIP' in rec:
                    rec_badge = '<span class="signal-badge" style="background:linear-gradient(135deg,#c0392b,#e74c3c);color:#fff;">✗ SKIP TRADE</span>'
                else:
                    rec_badge = ''

                html.append('    <div class="model-card">')
                html.append('        <div class="model-header">')
                html.append(f'            <div class="model-name">{result["name"]}</div>')
                html.append(f'            {signal_badge}')
                if rec_badge:
                    html.append(f'            {rec_badge}')
                html.append('        </div>')

                # Stats Grid — TP probability is the headline stat
                html.append('        <div class="stats-grid">')

                if 'ensemble_probability' in signal:
                    prob = signal['ensemble_probability']
                    if prob >= 75:
                        prob_color = '#27ae60'
                    elif prob >= 65:
                        prob_color = '#f39c12'
                    else:
                        prob_color = '#e74c3c'
                    conf_label = signal.get('confidence_level', '')
                    html.append('            <div class="stat-box" style="border-left:4px solid ' + prob_color + '; background:rgba(0,0,0,0.02);">')
                    html.append('                <div class="stat-label">🎯 TP Win Probability</div>')
                    html.append(f'                <div class="stat-value" style="color:{prob_color};font-size:1.8em;font-weight:800;">{prob:.1f}%</div>')
                    if conf_label:
                        html.append(f'                <div style="font-size:0.75em;color:#7f8c8d;margin-top:2px;">{conf_label} CONFIDENCE</div>')
                    html.append('            </div>')

                if 'current_price' in signal:
                    html.append('            <div class="stat-box">')
                    html.append('                <div class="stat-label">Current Price</div>')
                    html.append(f'                <div class="stat-value">${signal["current_price"]:,.2f}</div>')
                    html.append('            </div>')

                if 'expected_move' in signal and 'expected_move_pct' in signal:
                    move_color = '#27ae60' if signal['expected_move'] > 0 else '#e74c3c'
                    html.append('            <div class="stat-box">')
                    html.append('                <div class="stat-label">Expected Move</div>')
                    html.append(f'                <div class="stat-value" style="color:{move_color}">${signal["expected_move"]:+,.2f} ({signal["expected_move_pct"]:+.2f}%)</div>')
                    html.append('            </div>')

                if 'stop_loss' in signal and 'take_profit' in signal:
                    html.append('            <div class="stat-box">')
                    html.append('                <div class="stat-label">SL / TP Levels</div>')
                    html.append(f'                <div class="stat-value" style="color:#e74c3c;font-size:0.95em;">SL ${signal["stop_loss"]:,.2f} ({signal.get("stop_loss_pct",0):+.1f}%)</div>')
                    html.append(f'                <div class="stat-value" style="color:#27ae60;font-size:0.95em;">TP ${signal["take_profit"]:,.2f} ({signal.get("take_profit_pct",0):+.1f}%)</div>')
                    html.append('            </div>')

                if 'confidence' in signal:
                    html.append('            <div class="stat-box">')
                    html.append('                <div class="stat-label">Model Confidence</div>')
                    html.append(f'                <div class="stat-value">{signal["confidence"]:.1f}%</div>')
                    if 'predicted_price' in signal:
                        html.append(f'                <div style="font-size:0.75em;color:#7f8c8d;margin-top:2px;">Pred. tomorrow: ${signal["predicted_price"]:,.2f}</div>')
                    html.append('            </div>')

                html.append('        </div>')

                # Leverage table (5x P&L)
                if ('stop_loss' in signal and 'stop_loss_pct' in signal) or \
                   ('take_profit' in signal and 'take_profit_pct' in signal) or \
                   ('volatility' in signal):
                    html.append('        <h4 style="margin-top:25px; color:#2c3e50;">5x Leverage Position P&L</h4>')
                    html.append('        <table>')
                    html.append('            <tr>')
                    html.append('                <th>Parameter</th>')
                    html.append('                <th>Stock Price Level</th>')
                    html.append('                <th>Stock Move %</th>')
                    html.append('                <th>Position P&L (5x)</th>')
                    html.append('            </tr>')

                    if 'stop_loss' in signal and 'stop_loss_pct' in signal:
                        leverage_pct = signal['stop_loss_pct'] * 5
                        html.append('            <tr>')
                        html.append('                <td><strong>Stop Loss</strong></td>')
                        html.append(f'                <td>${signal["stop_loss"]:,.2f}</td>')
                        html.append(f'                <td>{signal["stop_loss_pct"]:+.2f}%</td>')
                        html.append(f'                <td style="color:#e74c3c; font-weight:700;">{leverage_pct:+.1f}%</td>')
                        html.append('            </tr>')

                    if 'take_profit' in signal and 'take_profit_pct' in signal:
                        leverage_pct = signal['take_profit_pct'] * 5
                        html.append('            <tr>')
                        html.append('                <td><strong>Take Profit</strong></td>')
                        html.append(f'                <td>${signal["take_profit"]:,.2f}</td>')
                        html.append(f'                <td>{signal["take_profit_pct"]:+.2f}%</td>')
                        html.append(f'                <td style="color:#27ae60; font-weight:700;">{leverage_pct:+.1f}%</td>')
                        html.append('            </tr>')

                    if 'volatility' in signal:
                        html.append('            <tr>')
                        html.append('                <td><strong>Daily Volatility</strong></td>')
                        html.append(f'                <td>${signal["volatility"]:,.2f}</td>')
                        html.append('                <td>-</td>')
                        html.append('                <td>-</td>')
                        html.append('            </tr>')

                    html.append('        </table>')

                # Probability Analysis (detailed breakdown)
                prob_html = generate_probability_html(signal)
                if prob_html:
                    html.append(prob_html)

                html.append('    </div>')  # Close model-card

        html.append('</div>')  # Close section

    # Signal Consensus
    if successful_models:
        html.append("### Signal Consensus\n")

        buy_count = 0
        short_count = 0
        hold_count = 0
        take_count = 0
        skip_count = 0
        tp_probs = []

        for result in successful_models:
            signal = parse_trading_signal(result['output'])
            signal_text = signal.get('signal', '')
            rec = signal.get('recommendation', '')

            if 'BUY' in signal_text:
                buy_count += 1
            elif 'SHORT' in signal_text:
                short_count += 1
            else:
                hold_count += 1

            if 'TAKE' in rec:
                take_count += 1
            elif 'SKIP' in rec:
                skip_count += 1

            if 'ensemble_probability' in signal:
                tp_probs.append(signal['ensemble_probability'])

        total = len(successful_models)
        avg_tp_prob = sum(tp_probs) / len(tp_probs) if tp_probs else None

        html.append(f"- **BUY signals:** {buy_count}/{total} ({buy_count/total*100:.0f}%)")
        html.append(f"- **SHORT signals:** {short_count}/{total} ({short_count/total*100:.0f}%)")
        html.append(f"- **HOLD signals:** {hold_count}/{total} ({hold_count/total*100:.0f}%)")
        if avg_tp_prob is not None:
            html.append(f"- **Avg TP Win Probability:** {avg_tp_prob:.1f}% ({take_count} TAKE / {skip_count} SKIP across {len(tp_probs)} models)\n")
        else:
            html.append("")

        if buy_count > total / 2:
            dir_text = "**Consensus Direction: BUY [UP]** - Majority of models predict upward movement"
        elif short_count > total / 2:
            dir_text = "**Consensus Direction: SHORT [DOWN]** - Majority of models predict downward movement"
        else:
            dir_text = "**Consensus Direction: MIXED** - No clear majority, proceed with caution"
        html.append(dir_text)

        if avg_tp_prob is not None:
            if take_count > total / 2 and avg_tp_prob >= 65:
                html.append(f"**Trade Recommendation: TAKE TRADE** - {take_count}/{total} models agree, avg {avg_tp_prob:.1f}% TP probability")
            elif skip_count > total / 2 or avg_tp_prob < 60:
                html.append(f"**Trade Recommendation: SKIP TRADE** - avg TP probability {avg_tp_prob:.1f}% is too low")
            else:
                html.append(f"**Trade Recommendation: CAUTION** - Models disagree ({take_count} take / {skip_count} skip), avg {avg_tp_prob:.1f}%")

        html.append("")

    # Detailed Results
    html.append("## Detailed Results by Model\n")

    for result in results:
        html.append(f"### {result['name']}\n")

        if result['success']:
            html.append("**Status:** [OK] Success\n")

            # Extract key sections from output
            output = result['output']

            # Find model summary section
            if 'MODEL EVALUATION RESULTS' in output:
                lines = output.split('\n')
                in_eval = False
                eval_lines = []

                for line in lines:
                    if 'MODEL EVALUATION RESULTS' in line:
                        in_eval = True
                    elif in_eval:
                        if line.startswith('====='):
                            if len(eval_lines) > 5:  # Found end
                                break
                        else:
                            eval_lines.append(line)

                if eval_lines:
                    html.append("```")
                    html.append('\n'.join(eval_lines[:20]))  # First 20 lines
                    html.append("```\n")
        else:
            html.append("**Status:** [FAIL] Failed\n")
            html.append(f"**Error:** {result['error']}\n")

    # Recommendations
    html.append("## Recommendations\n")

    if successful_models:
        html.append("### Model Selection:\n")

        # Get best models
        model_metrics = []
        for result in successful_models:
            metrics = parse_metrics(result['output'])
            metrics['name'] = result['name']
            model_metrics.append(metrics)

        best_mae = min(model_metrics, key=lambda x: x.get('test_mae', float('inf')))

        html.append(f"**For price prediction accuracy:** Use **{best_mae['name']}** (lowest MAE: ${best_mae.get('test_mae', 0):.2f})\n")

        html.append("### Trading Strategy:\n")
        html.append("1. **Check consensus:** If 3+ models agree on direction, signal is stronger")
        html.append("2. **Use stop loss:** Always set stop loss to limit downside risk")
        html.append("3. **Monitor confidence:** Higher confidence models (>60%) are more reliable")
        html.append("4. **Consider volatility:** High volatility = larger stop loss needed")
        html.append("5. **Combine with fundamentals:** Models only use price data, add fundamental analysis\n")

    html.append("### Model Comparison:\n")
    html.append("- **LSTM:** CNN-1D feature extraction + stacked LSTM + temporal attention")
    html.append("- **TFT:** CNN-1D + Gated Residual Networks + LSTM encoder + Multi-Head Attention")
    html.append("- **XGBoost:** Gradient boosting, good with features, interpretable")
    html.append("- **LightGBM:** Like XGBoost but faster, often more accurate")
    html.append("- **RandomForest:** Ensemble of decision trees, walk-forward validation, robust\n")

    html.append("</div>")  # Close content div

    html.append("""
            <!-- Disclaimer -->
            <div class="disclaimer">
                <h3>⚠️ DISCLAIMER</h3>
                <p>This report is generated by statistical models and is <strong>NOT financial advice</strong>.</p>
                <p>Past performance does not guarantee future results.</p>
                <p>Stock prices are inherently unpredictable and influenced by many factors not captured by these models.</p>
                <p>Always do your own research, understand the risks, and never invest more than you can afford to lose.</p>
            </div>

            <div class="footer">
                <p>Report generated by <strong>Claude Code Stock Prediction System</strong></p>
                <p>Generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            </div>
        </div> <!-- Close content -->
    </div> <!-- Close container -->
</body>
</html>""")

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))

    print(f"\nReport saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Train all prediction models and generate comparison report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Fetch data and train (recommended)
  python main.py --ticker MSFT
  python main.py --ticker MSFT --months 60  # 5 years of data
  python main.py --ticker BTC-USD --months 24  # 2 years

  # Train with existing CSV
  python main.py MSFT_daily_data_20260520.csv

This will:
  1. Fetch stock data (if --ticker provided) or use existing CSV
  2. Run all 4 models (LSTM, XGBoost, LightGBM, RandomForest) in parallel
  3. Generate plots for each model
  4. Create a comprehensive HTML report: RESULT-{TICKER}-{DATE}.html

Default: 48 months (4 years) of historical data
        '''
    )

    parser.add_argument('csv_file', type=str, nargs='?', help='Path to CSV file with stock data (optional if --ticker provided)')
    parser.add_argument('--ticker', type=str, help='Stock ticker to fetch (e.g., MSFT, BTC-USD)')
    parser.add_argument('--months', type=int, default=48, help='Months of historical data (default: 48 = 4 years)')

    args = parser.parse_args()

    # Determine data source
    if args.ticker:
        # Fetch stock data
        csv_file = fetch_stock_data(args.ticker, args.months)
    elif args.csv_file:
        # Use provided CSV file
        csv_file = args.csv_file
        if not os.path.exists(csv_file):
            print(f"Error: File {csv_file} not found!")
            sys.exit(1)
    else:
        print("Error: Must provide either --ticker or csv_file")
        parser.print_help()
        sys.exit(1)

    # Extract ticker and date for report filename
    basename = os.path.basename(csv_file)
    parts = basename.split('_')
    ticker = parts[0]
    date_str = datetime.now().strftime('%Y%m%d')
    output_file = f"RESULT-{ticker}-{date_str}.html"

    print("="*60)
    print("STOCK PRICE PREDICTION - ALL MODELS")
    print("="*60)
    print(f"Ticker: {ticker}")
    print(f"Data file: {csv_file}")
    print(f"Output report: {output_file}")
    print("="*60)
    print("\nTraining models in parallel...")
    print("This will take 5-15 minutes depending on data size and GPU availability\n")

    # Define models to run
    models = [
        ('LSTM', 'train_lstm.py'),
        ('TFT', 'train_tft.py'),
        ('XGBoost', 'train_xgboost.py'),
        ('LightGBM', 'train_lightgbm.py'),
        ('RandomForest', 'train_randomforest.py')
    ]

    # Run all models in parallel using ThreadPoolExecutor
    results = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        future_to_model = {
            executor.submit(run_model, name, script, csv_file): name
            for name, script in models
        }

        # Collect results as they complete
        for future in as_completed(future_to_model):
            result = future.result()
            results.append(result)

    # Sort results by model name for consistent ordering
    results.sort(key=lambda x: x['name'])

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

    # Print summary
    successful = sum(1 for r in results if r['success'])
    failed = sum(1 for r in results if not r['success'])

    print(f"\nResults: {successful} successful, {failed} failed")

    for result in results:
        status = "[OK]" if result['success'] else "[FAIL]"
        print(f"  {status} {result['name']}")

    # Generate HTML report
    print("\nGenerating comparison report...")
    generate_html_report(results, csv_file, output_file)

    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)
    print(f"\nView the report: {output_file}")
    print("\nGenerated files:")
    print("  - Plots: *_predictions.png, *_feature_importance.png")
    print("  - Models: *.pkl, *.keras  (includes best_tft_model.keras)")
    print("  - Report: " + output_file)
    print("\n")

if __name__ == "__main__":
    main()
