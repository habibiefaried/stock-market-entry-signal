import yfinance as yf
import pandas as pd
from datetime import datetime
import argparse
import sys

def calculate_technical_indicators(df):
    """Calculate technical indicators for the stock data"""
    # Moving Averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()

    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

    # Volume Moving Average
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()

    return df

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

    # Calculate technical indicators
    df = calculate_technical_indicators(df)

    # Reorder columns for better readability
    columns_order = [
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
        'MA_5', 'MA_10', 'MA_20', 'MA_50', 'MA_200',
        'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Middle', 'BB_Lower', 'Volume_MA_20'
    ]

    # Keep only columns that exist
    columns_order = [col for col in columns_order if col in df.columns]
    df = df[columns_order]

    # Save to CSV
    filename = f"{ticker.replace('-', '_')}_daily_data_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(filename, index=False)

    print(f"\nData saved to: {filename}")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nLast few rows:")
    print(df.tail())
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData info:")
    print(df.info())

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
    parser.add_argument('--months', type=int, default=6, help='Number of months of historical data (default: 6)')

    args = parser.parse_args()

    df = fetch_market_data(args.ticker, args.months)
