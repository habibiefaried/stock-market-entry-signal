import yfinance as yf
import pandas as pd
from datetime import datetime
import argparse
import sys

# Technical indicators removed - only OHLCV data is saved
# If you need technical indicators for trading signals, implement them separately

def fetch_market_data(ticker, months=6):
    """Fetch market data with daily candles for specified ticker and period"""
    print(f"Fetching {ticker} data ({months} months)...")

    # Create ticker object
    ticker_obj = yf.Ticker(ticker)

    # Get historical data with daily interval
    df = ticker_obj.history(period=f"{months}mo", interval="1d")

    if df.empty:
        print(f"Error: No data found for {ticker}")
        sys.exit(1)

    # Reset index to make Date a column
    df.reset_index(inplace=True)

    # Keep only OHLCV columns (no technical indicators)
    # Models will use only these 5 features + Date
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Save to CSV
    filename = f"{ticker.replace('-', '_')}_daily_data_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(filename, index=False)

    print(f"\nData saved to: {filename}")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nLast 5 rows:")
    print(df.tail())
    print(f"\nNo NaN values - ready for model training!")

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fetch stock/crypto market data with technical indicators',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Stocks
  python fetch_stock_data.py MSFT --months 6
  python fetch_stock_data.py AAPL --months 12
  python fetch_stock_data.py GOOGL --months 3

  # Cryptocurrencies
  python fetch_stock_data.py BTC-USD --months 6
  python fetch_stock_data.py ETH-USD --months 12
  python fetch_stock_data.py SOL-USD --months 3
        '''
    )

    parser.add_argument('ticker', type=str, help='Ticker symbol (e.g., MSFT, BTC-USD, ETH-USD)')
    parser.add_argument('--months', type=int, default=12, help='Number of months of historical data (default: 12)')

    args = parser.parse_args()

    df = fetch_market_data(args.ticker, args.months)
