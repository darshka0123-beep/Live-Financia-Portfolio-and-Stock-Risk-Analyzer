import asyncio
from io import StringIO
import urllib.parse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as g_obj
from pyodide.http import pyfetch
from scipy.optimize import minimize
import seaborn as sns
import streamlit as st

# Page Setup
st.set_page_config(
    page_title="Live Financial Portfolio and Stock Risk Analyzer", layout="wide"
)

st.title("Live Financial Portfolio and Stock Risk Analyzer")
st.write(
    "Welcome to your FinTech dashboard! Choose a stock ticker and date range in"
    " the sidebar to run quantitative analysis."
)

# Sidebar Inputs
st.sidebar.header("User inputs")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
start_date = st.sidebar.date_input(
    "Start Date", value=pd.to_datetime("2023-01-01")
)
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))
risk_free_rate = (
    st.sidebar.slider(
        "Risk-Free-Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=4.0,
        step=0.1,
    )
    / 100
)


# Async browser-compatible data fetcher using Stooq + CORS Proxy
@st.cache_data(ttl=3600)
async def fetch_yahoo_data(symbol, start_dt, end_dt):
  try:
    s_date = pd.to_datetime(start_dt).strftime("%Y%m%d")
    e_date = pd.to_datetime(end_dt).strftime("%Y%m%d")

    stooq_url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&d1={s_date}&d2={e_date}&i=d"
    proxy_url = (
        f"https://corsproxy.io/?{urllib.parse.quote(stooq_url, safe='')}"
    )

    response = await pyfetch(proxy_url)
    if response.status != 200:
      return pd.DataFrame()

    csv_text = await response.string()
    if "No data" in csv_text or not csv_text.strip():
      return pd.DataFrame()

    df = pd.read_csv(StringIO(csv_text))
    if df.empty or "Date" not in df.columns:
      return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
  except Exception:
    return pd.DataFrame()


@st.cache_data(ttl=3600)
async def fetch_multiple_symbols(symbols, start_dt, end_dt):
  combined_df = pd.DataFrame()
  for s in symbols:
    data = await fetch_yahoo_data(s, start_dt, end_dt)
    if not data.empty and "Close" in data.columns:
      combined_df[s] = data["Close"]
  return combined_df.dropna()


# Session State initialized
if "run_analysis" not in st.session_state:
  st.session_state.run_analysis = False

if st.sidebar.button("Run Risk Analysis"):
  st.session_state.run_analysis = True

if st.session_state.run_analysis:
  df = asyncio.run(fetch_yahoo_data(ticker, start_date, end_date))

  if df.empty:
    st.error(
        f"No Data found for ticker '{ticker}'. Please check the symbol and try"
        " again."
    )
  else:
    # Quant Calculations
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility_21D"] = (
        df["Daily_Return"].rolling(window=21).std() * (252**0.5)
    )
    df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod()
    df["Peak"] = df["Cumulative_Return"].cummax()
    df["Drawdown"] = (df["Cumulative_Return"] - df["Peak"]) / df["Peak"]

    clean_returns = df["Daily_Return"].dropna()
    annual_return = clean_returns.mean() * 252
    annual_volatility = clean_returns.std() * (252**0.5)
    sharpe_ratio = (
        (annual_return - risk_free_rate) / annual_volatility
        if annual_volatility != 0
        else 0
    )
    max_drawdown = df["Drawdown"].min()

    st.subheader(f"Key Risk metrics for {ticker}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annualized Return", f"{annual_return * 100:.2f}%")
    col2.metric("Annualized Volatility", f"{annual_volatility * 100:.2f}%")
    col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    col4.metric("Max Drawdown", f"{max_drawdown * 100:.2f}%")

    st.markdown("---")
    st.header("Technical Analysis Indicators")

    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    fig_sma, ax_sma = plt.subplots(figsize=(10, 4))
    ax_sma.plot(df.index, df["Close"], label="Close Price", alpha=0.5)
    ax_sma.plot(df.index, df["SMA_50"], label="50-Day SMA", color="orange")
    ax_sma.plot(df.index, df["SMA_200"], label="200-Day SMA", color="red")
    ax_sma.set_title(f"{ticker} Moving Averages")
    ax_sma.legend()
    ax_sma.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig_sma)

    st.markdown("---")
    st.header("Interactive Technical Analysis (Plotly)")

    df["20_SMA"] = df["Close"].rolling(window=20).mean()
    df["20_STD"] = df["Close"].rolling(window=20).std()
    df["Upper_Band"] = df["20_SMA"] + (df["20_STD"] * 2)
    df["Lower_Band"] = df["20_SMA"] - (df["20_STD"] * 2)

    fig_candle = g_obj.Figure()
    fig_candle.add_trace(
        g_obj.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price OHLC",
        )
    )
    fig_candle.add_trace(
        g_obj.Scatter(
            x=df.index,
            y=df["Upper_Band"],
            line=dict(color="grey", width=1),
            name="Upper Band",
        )
    )
    fig_candle.add_trace(
        g_obj.Scatter(
            x=df.index,
            y=df["Lower_Band"],
            line=dict(color="grey", width=1),
            name="Lower Band",
        )
    )
    fig_candle.update_layout(
        title=f"{ticker} Price & Bollinger Bands",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    st.markdown("---")
    st.header("Monte Carlo Risk Simulation (30-Day Outlook)")

    if st.button("Run 1000 Path Simulation"):
      num_simulations = 1000
      time_horizon = 30
      last_price = df["Close"].iloc[-1]
      daily_mean = clean_returns.mean()
      daily_std = clean_returns.std()

      simulation_matrix = np.zeros((time_horizon, num_simulations))
      for i in range(num_simulations):
        prices = [last_price]
        for t in range(1, time_horizon):
          simulated_return = np.random.normal(daily_mean, daily_std)
          prices.append(prices[-1] * (1 + simulated_return))
        simulation_matrix[:, i] = prices

      fig_mc, ax_mc = plt.subplots(figsize=(10, 5))
      ax_mc.plot(simulation_matrix, color="blue", alpha=0.03)
      ax_mc.set_title(
          f"1,000 Simulated Price Paths for {ticker} over Next 30 Days"
      )
      ax_mc.set_ylabel("Simulated Prices ($)")
      ax_mc.set_xlabel("Days Ahead")
      st.pyplot(fig_mc)

      ending_prices = simulation_matrix[-1, :]
      percentile_5th = np.percentile(ending_prices, 5)
      max_expected_loss_pct = (
          (percentile_5th - last_price) / last_price
      ) * 100

      st.error(
          f"**95% Value at Risk (VaR):** There is a 5% chance {ticker} drops"
          f" below **${percentile_5th:.2f}** over the next 30 days (a loss of"
          f" **{max_expected_loss_pct:.2f}%**)."
      )
