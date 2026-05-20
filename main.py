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
    signal_match = re.search(r'SIGNAL:\s+([A-Z\s()]+)', output)
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

    return signal

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
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }
        h3 {
            color: #7f8c8d;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        .signal-box {
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }
        .signal-buy {
            border-left-color: #27ae60;
            background-color: #e8f8f5;
        }
        .signal-short {
            border-left-color: #e74c3c;
            background-color: #fadbd8;
        }
        .signal-hold {
            border-left-color: #f39c12;
            background-color: #fef5e7;
        }
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: bold;
        }
        .badge-success {
            background-color: #27ae60;
            color: white;
        }
        .badge-danger {
            background-color: #e74c3c;
            color: white;
        }
        .badge-warning {
            background-color: #f39c12;
            color: white;
        }
        .disclaimer {
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }
        .metric {
            display: inline-block;
            margin-right: 20px;
        }
        .metric-label {
            font-weight: bold;
            color: #7f8c8d;
        }
        .metric-value {
            color: #2c3e50;
            font-size: 1.1em;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
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
        <h1>Stock Price Prediction Report: """ + ticker + """</h1>
        <div class="header-info">
            <div class="metric">
                <span class="metric-label">Generated:</span>
                <span class="metric-value">""" + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</span>
            </div>
            <div class="metric">
                <span class="metric-label">Data Source:</span>
                <span class="metric-value">""" + csv_file + """</span>
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
        html.append("## Trading Signals\n")
        html.append("All models predict tomorrow's price and provide trading recommendations.\n")

        for result in successful_models:
            signal = parse_trading_signal(result['output'])

            if signal:
                html.append(f"### {result['name']}\n")

                # Signal
                signal_text = signal.get('signal', 'UNKNOWN')
                if 'BUY' in signal_text:
                    prefix = "[BUY]"
                elif 'SHORT' in signal_text:
                    prefix = "[SHORT]"
                else:
                    prefix = "[HOLD]"

                html.append(f"**{prefix} Signal:** {signal_text}\n")

                # Prices
                if 'current_price' in signal:
                    html.append(f"- **Current Price:** ${signal['current_price']:,.2f}")
                if 'predicted_price' in signal:
                    html.append(f"- **Predicted Tomorrow:** ${signal['predicted_price']:,.2f}")
                if 'expected_move' in signal and 'expected_move_pct' in signal:
                    html.append(f"- **Expected Move:** ${signal['expected_move']:+,.2f} ({signal['expected_move_pct']:+.2f}%)")

                # Risk management
                html.append("\n**Risk Management:**")
                if 'stop_loss' in signal and 'stop_loss_pct' in signal:
                    html.append(f"- **Stop Loss:** ${signal['stop_loss']:,.2f} ({signal['stop_loss_pct']:+.2f}%)")
                if 'take_profit' in signal and 'take_profit_pct' in signal:
                    html.append(f"- **Take Profit:** ${signal['take_profit']:,.2f} ({signal['take_profit_pct']:+.2f}%)")

                # Additional info
                if 'confidence' in signal:
                    html.append(f"- **Confidence:** {signal['confidence']:.1f}%")
                if 'volatility' in signal:
                    html.append(f"- **Daily Volatility:** ${signal['volatility']:,.2f}")

                html.append("")

    # Signal Consensus
    if successful_models:
        html.append("### Signal Consensus\n")

        buy_count = 0
        short_count = 0
        hold_count = 0

        for result in successful_models:
            signal = parse_trading_signal(result['output'])
            signal_text = signal.get('signal', '')

            if 'BUY' in signal_text:
                buy_count += 1
            elif 'SHORT' in signal_text:
                short_count += 1
            else:
                hold_count += 1

        total = len(successful_models)
        html.append(f"- **BUY signals:** {buy_count}/{total} ({buy_count/total*100:.0f}%)")
        html.append(f"- **SHORT signals:** {short_count}/{total} ({short_count/total*100:.0f}%)")
        html.append(f"- **HOLD signals:** {hold_count}/{total} ({hold_count/total*100:.0f}%)\n")

        if buy_count > total / 2:
            html.append("**Consensus: BUY [UP]** - Majority of models predict upward movement")
        elif short_count > total / 2:
            html.append("**Consensus: SHORT [DOWN]** - Majority of models predict downward movement")
        else:
            html.append("**Consensus: MIXED** - No clear majority, proceed with caution")

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
    html.append("- **LSTM:** Deep learning, captures sequences, needs lots of data")
    html.append("- **XGBoost:** Gradient boosting, good with features, interpretable")
    html.append("- **LightGBM:** Like XGBoost but faster, often more accurate\n")

    html.append("</div>")  # Close content div

    # Disclaimer
    html.append('<div class="disclaimer">')
    html.append("<h2>DISCLAIMER</h2>")
    html.append("<p>This report is generated by statistical models and is NOT financial advice.</p>")
    html.append("<p>Past performance does not guarantee future results.</p>")
    html.append("<p>Stock prices are inherently unpredictable and influenced by many factors not captured by these models.</p>")
    html.append("<p>Always do your own research, understand the risks, and never invest more than you can afford to lose.</p>")
    html.append('</div>')

    html.append(f"<p style='text-align: center; color: #7f8c8d; margin-top: 30px;'><em>Report generated by main.py on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>")

    # Close HTML
    html.append("</div>")  # Close container
    html.append("</body>")
    html.append("</html>")

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
  2. Run all 3 models (LSTM, XGBoost, LightGBM) in parallel
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
    print("This will take 2-5 minutes depending on data size and GPU availability\n")

    # Define models to run
    models = [
        ('LSTM', 'train_lstm.py'),
        ('XGBoost', 'train_xgboost.py'),
        ('LightGBM', 'train_lightgbm.py')
    ]

    # Run all models in parallel using ThreadPoolExecutor
    results = []

    with ThreadPoolExecutor(max_workers=3) as executor:
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
    print("  - Models: *.pkl, *.keras")
    print("  - Report: " + output_file)
    print("\n")

if __name__ == "__main__":
    main()
