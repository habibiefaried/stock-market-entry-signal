"""
Stock Ranking Tool - ranks target stocks by agent confidence and win probability.

Reads `target_stock.txt`, runs the full pipeline (models + RL agent) on each
ticker, then produces `stock-ranking-result.txt` sorted by expected value
(confidence x winrate - highest probability of hitting TP).

Usage:
    python rank_stocks.py
    python rank_stocks.py --top 5
    python rank_stocks.py --months 84 --leverage 5
"""

import argparse, os, sys, subprocess, re, glob
from datetime import datetime


def run_pipeline(ticker, months=84):
    """Run main.py for a ticker and return parsed agent result from HTML."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(
        [sys.executable, os.path.join(base_dir, 'main.py'),
         '--ticker', ticker, '--months', str(months)],
        capture_output=True, text=True, timeout=7200,  # 2h per ticker (full model training + RL agent)
        cwd=base_dir
    )
    # Find the generated HTML report
    html_files = glob.glob(os.path.join(base_dir, 'REPORT', f'RESULT-{ticker}-*.html'))
    if not html_files:
        html_files = glob.glob(os.path.join(base_dir, f'RESULT-{ticker}-*.html'))
    if not html_files:
        return None

    html_path = max(html_files, key=os.path.getmtime)  # newest
    with open(html_path, encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Parse values from HTML
    def _num(pattern, text=content):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else 0.0

    # Extract RL agent decision only (not model-level badges)
    # The RL agent badge is a <div> right after "RL AGENT DECISION" section
    m = re.search(
        r'RL AGENT DECISION.*?<div[^>]*>\s*(LONG\s*\(BUY\)|SHORT\s*\(SELL\)|HOLD\s*\(WAIT\))\s*</div>',
        content, re.DOTALL
    )
    decision = 'HOLD'
    if m:
        badge = m.group(1)
        if 'LONG' in badge:
            decision = 'LONG'
        elif 'SHORT' in badge:
            decision = 'SHORT'

    return {
        'ticker': ticker,
        'decision': decision,
        'confidence': _num(r'Agent Confidence</div>\s*<div[^>]*>([\d.]+)%'),
        'winrate': _num(r'Win Rate</div>\s*<div[^>]*>([\d.]+)%'),
        'profit_factor': _num(r'Profit Factor</div>\s*<div[^>]*>([\d.]+)'),
        'sharpe': _num(r'Sharpe Ratio</div>\s*<div[^>]*>([\d.]+)'),
        'trades': int(_num(r'Total Trades</div>\s*<div[^>]*>(\d+)')),
        'entry': _num(r'Entry Price</div>\s*<div[^>]*>\$([\d.]+)'),
        'sl_pct': _num(r'Stop Loss</div>\s*<div[^>]*>\$[\d.]+\s*\(([+-]?[\d.]+)%\)'),
        'tp_pct': _num(r'Take Profit</div>\s*<div[^>]*>\$[\d.]+\s*\(([+-]?[\d.]+)%\)'),
    }


def main():
    parser = argparse.ArgumentParser(description='Rank target stocks by trading confidence')
    parser.add_argument('--top', type=int, default=0, help='Show only top N (0 = all)')
    parser.add_argument('--months', type=int, default=84, help='Months of data (default: 84)')
    parser.add_argument('--input', type=str, default='target_stock.txt', help='Input file')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, args.input)
    if not os.path.exists(input_path):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    with open(input_path) as f:
        tickers = [line.strip() for line in f if line.strip()]

    print(f"Ranking {len(tickers)} stocks ({args.months} months data)...")
    print("=" * 60)

    results = []
    for i, ticker in enumerate(tickers):
        print(f"\n[{i+1}/{len(tickers)}] {ticker}...")
        try:
            r = run_pipeline(ticker, months=args.months)
            if r:
                results.append(r)
                print(f"  {r['decision']} | Conf={r['confidence']:.1f}% | "
                      f"WR={r['winrate']:.1f}% | PF={r['profit_factor']:.2f}")
            else:
                print(f"  FAILED - no report generated")
        except Exception as e:
            print(f"  ERROR: {e}")

    if not results:
        print("\nNo results. Check that main.py runs correctly.")
        return

    # Sort by expected value: confidence × winrate (as promised in docstring)
    results.sort(key=lambda r: r['confidence'] * r['winrate'] / 100, reverse=True)

    # Generate ranking file
    output_path = os.path.join(base_dir, 'stock-ranking-result.txt')
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    top_n = args.top if args.top > 0 else len(results)

    lines = []
    lines.append("=" * 70)
    lines.append("STOCK RANKING - MOST CONFIDENT TO LEAST CONFIDENT")
    lines.append(f"Generated: {now}  |  Data: {args.months} months")
    lines.append(f"TP=1.5 ATR  |  SL=1.0 ATR  |  Ratio 1.5:1  |  Break-even 40%  |  Production SL/TP match training")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"{'#':>3s}  {'Ticker':<8s} {'Action':>6s} {'Conf':>7s} {'WinRate':>8s} {'PF':>7s} {'SL%':>7s} {'TP%':>7s} {'Trades':>7s} {'Entry':>10s}")
    lines.append("-" * 82)

    for rank, r in enumerate(results[:top_n], 1):
        lines.append(
            f"{rank:>3d}  {r['ticker']:<8s} {r['decision']:>6s} "
            f"{r['confidence']:>6.1f}% {r['winrate']:>7.1f}% "
            f"{r['profit_factor']:>6.2f} "
            f"{r['sl_pct']:>+6.1f}% {r['tp_pct']:>+6.1f}% "
            f"{r['trades']:>7d} ${r['entry']:>9.2f}"
        )

    lines.append("")
    lines.append("---")
    lines.append("Legend:")
    lines.append("  Conf   = Agent confidence in the current trade direction")
    lines.append("  WR     = Backtest winrate on validation data (~7 months)")
    lines.append("  PF     = Profit Factor (gross profit / gross loss, >1 = profitable)")
    lines.append("  SL%    = Stop Loss distance from entry (percentage)")
    lines.append("  TP%    = Take Profit distance from entry (percentage)")
    lines.append("")
    lines.append("DISCLAIMER: Rankings are statistical, NOT financial advice.")
    lines.append("Always verify with your own analysis before placing real trades.")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nRanking saved to: {output_path}")
    print('\n'.join(lines))


if __name__ == "__main__":
    main()
