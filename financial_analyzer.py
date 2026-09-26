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

    # Annualized return(Average daily return scaled to 252 trading days)
    annual_return = clean_returns.mean() * 252 

    # Annual Volaitlity (std scaled to 252 trading days)
    annual_volatility = clean_returns.std() * 252 (252 ** 0.5)

    # Sharpe ratio 
    
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

    # Max drawdown
    max_drawdown = df['Drawdown'].min()

    # Printable Report Output
    print("\n" + "=" * 45)
    print(f" Quant Risk Analysis Report: {ticker}")
    print("=" * 45)
    print(f" Annualized Return: {annual_return * 100:>8.2f}%")
    print(f" Annualized Volatility: {annual_volatility * 100:>8.2f}%")
    print(f" Sharpe Ratio: {sharpe_ratio:>8.2f}")
    print(f" Max Drawdown: {max_drawdown * 100:>8.2f}%")
    print("=" * 45 + "\n")

    # Run the summary function on you dataset
    print_report(df, "AAPL")

# Matplot lib
def plot_risk_dashboard(df, ticker):
    # 3 panel matplotlib with Price, Daily returns, and drawdown over time

    fig,(ax1, ax2, ax3) = plt.subplots(3,1, figsize=(12,8), sharex=True)

    # Top Panel: Closing Price
    ax1.plot(df.index, df['Close'], color='#841077', linewidth=1.5, label='Close Price ($)')
    ax1.set_title(f"{ticker} Quantitative Risk Analysis Dashboard", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Middle Panel, Daily Returns (Volatility)
    ax2.plot(df.index, df['Daily_Return'], color='#D43131', alpha=0.6, label='Daily Returns')
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_ylabel("Daily Change")
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Bottom Panel: Drawdown Area Chart
    ax3.fill_between(df.index, df['Drawdown'] * 100, 0, color='#1A21D9', alpha=0.4, label='Drawdown (%)')
    ax3.set_ylabel("Drawdown %")
    ax3.set_xlabel("Date")
    ax3.legend(loc='lower left')
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

# Call the plot function
plot_risk_dashboard(df, "AAPL")


