import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page Setup
st.set_page_config(page_title="Live Financial Portfolio and Stock Risk Analyzer", layout="wide")

st.title("Live Financial Portfolio and Stock Risk Analyzer")
st.write("welcome to your fintech dashboard. Choose a stock ticker and date range in te sidebar to run quantitative analysis.")

# Sidebar Inputs
st.sidebar.header("User inputs")

# Interactive ticker input with AAPL as default
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()

# Date picker for start and end date
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))

# Risk Free Rate Slider (default 4%)
risk_free_rate = st.sidebar.slider("Risk-Free-Rate (%)", min_value=0.0, max_value=10.0, value=4.0, step=0.1) / 100

# Data fetching function
@st.cache_data
def get_stock_data(symbol, start, end):
    stock = yf.Ticker(symbol)
    df = stock.history(start=start)
    if not df.empty:
        df = df[['Close']].copy()
    return df

# Run Analysis when user presses the button
if st.sidebar.button("Run Risk Analysis"):
    df = get_stock_data(ticker, start_date, end_date)

    if df.empty:
        st.error(f"No Data found for ticker '{ticker}'. Please check the symbol and try again.")
    else:
        # Quant Calculations
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility_21D'] = df['Daily_Return'].rolling(window=21).std() * (252 ** 0.5)
        df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod()
        df['Peak'] = df['Cumulative_Return'].cummax()
        df['Drawdown'] = (df['Cumulative_Return'] - df['Peak']) / df['Peak']

        # Clean returns for overall stats
        clean_returns = df['Daily_Return'].dropna()
        annual_return = clean_return.mean() * 252
        annual_volatility = clean_return/std() * (252 ** 0.5)
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        max_drawdown = df['Drawdown'].min()

        # Display Metric cards
        st.subheader(f"Key Risk metric for {ticker}")

        # Streamlit metrics layout in 4 columns
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Annualized Return", f"{annual_return * 100:.2f}%")
        col2.metric("Annualized Volatility", f"{annual_volatility * 100:.2f}%")
        col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        col4.metric("Max Drawdown", f"{max_drawdown * 100:.2f}%")

        st.markdown("---")

        # Display Charts
        st.subheader("Visual Analysis Dashboard")

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Top Panel: Price
        ax1.plot(df.index, df['Close'], color="#D20FB5", linewidth=1.5, label='Close Price ($)' )
        ax1.set_title(f"{ticker} Historical Performance", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Price ($)") 
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Middle Panel, Daily Returns
        ax2.plot(df.index, df['Daily_Return'], color='#9467bd', alpha=0.6)   
