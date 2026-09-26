import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_stock_data(ticker, start_date, end_date):
    # Downloads Stock Data from Yahoo Finance
     
    print(f"Fetching Data for {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)

    # Keep only the Close price column
    df = df[['Close']].copy()
    return df

# Test fetching data for Apple
df = get_stock_data("AAPL", "2023-01-01", "2024-01-01")
print(df.head())

# Calculate daily percent change: (Price today - Price yesterday) / price yesterday
df['Daily_Return']  = df['Close'].pct_change()

# Look at the first 5 rows to verify
print(df.head())

# Calculate 21 Day rolling volatility (annualized)
df['Volatility_21D'] = df['Daily_Return'].rolling(window=21).std() * (252 ** 0.5)

# Print the updated DataFrame to inspect
print(df.tail(10))

# Calculate Max Drawdown
# 1. Calculate cumulative return(growth of 1$ invested of Day 1)
df['Cumulative_Return'] = (1+ df['Daily_Return']).cumprod()

# 2. Track the Highest peak hit up to each day
df['Peak'] = df['Cumulative_Return'].cummax()

#3. Calculate drawdown percentage from that peak
df['Drawdown'] = (df['Cumulative_Return'] - df['Peak']) / df['Peak']

# Print the maximum drop experienced
max_drawdown = df['Drawdown'].min()
print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")

def print_report(df, ticker, risk_free_rate=0.04):
    # Compute and displays overall risk adjusted data for a stock
    #Clean out nan values
    clean_returns = df['Daily_Return'].dropna()

    # Annualized return(Average daily)


