import json
import time
import urllib.parse
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.graph_objects as g_obj
import plotly.express as px
from scipy.optimize import minimize

# Page Setup
st.set_page_config(page_title="Live Financial Portfolio and Stock Risk Analyzer", layout="wide")

st.title("Live Financial Portfolio and Stock Risk Analyzer")
st.write("Welcome to your FinTech dashboard! Choose a stock ticker and date range in the sidebar to run quantitative analysis.")

# Sidebar Inputs
st.sidebar.header("User inputs")

# Interactive ticker input with AAPL as default
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()

# Date picker for start and end date
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))

# Risk Free Rate Slider (default 4%)
risk_free_rate = st.sidebar.slider("Risk-Free-Rate (%)", min_value=0.0, max_value=10.0, value=4.0, step=0.1) / 100

# Helper function to route requests through a CORS proxy for browser execution
def get_cors_url(target_url):
    return f"https://corsproxy.io/?{urllib.parse.quote(target_url, safe='')}"

# Pure Python Data Fetcher via Yahoo Query API + CORS Proxy
@st.cache_data(ttl=3600)
def fetch_yahoo_data(symbol, start_dt, end_dt):
    try:
        p1 = int(pd.to_datetime(start_dt).timestamp())
        p2 = int(pd.to_datetime(end_dt).timestamp())
        raw_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={p1}&period2={p2}&interval=1d"
        proxy_url = get_cors_url(raw_url)
        
        res = requests.get(proxy_url)
        data = res.json()
        
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        
        df = pd.DataFrame({
            "Date": pd.to_datetime(timestamps, unit="s"),
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
            "Volume": quote.get("volume")
        })
        df.set_index("Date", inplace=True)
        return df.dropna()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_multiple_symbols(symbols, start_dt, end_dt):
    combined_df = pd.DataFrame()
    for s in symbols:
        data = fetch_yahoo_data(s, start_dt, end_dt)
        if not data.empty and "Close" in data.columns:
            combined_df[s] = data["Close"]
    return combined_df.dropna()

@st.cache_data(ttl=3600)
def fetch_ticker_info(symbol):
    try:
        raw_url = f"https://query1.finance.yahoo.com/v7/finance/options/{symbol}"
        proxy_url = get_cors_url(raw_url)
        
        res = requests.get(proxy_url)
        data = res.json()
        meta = data["optionChain"]["result"][0]["quote"]
        return meta
    except Exception:
        return {}

# Session State initialized
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

if st.sidebar.button("Run Risk Analysis"):
    st.session_state.run_analysis = True

if st.session_state.run_analysis:
    df = fetch_yahoo_data(ticker, start_date, end_date)

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
        annual_return = clean_returns.mean() * 252
        annual_volatility = clean_returns.std() * (252 ** 0.5)
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
        max_drawdown = df['Drawdown'].min()
        
        # Display Metric cards
        st.subheader(f"Key Risk metric for {ticker}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Annualized Return", f"{annual_return * 100:.2f}%")
        col2.metric("Annualized Volatility", f"{annual_volatility * 100:.2f}%")
        col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        col4.metric("Max Drawdown", f"{max_drawdown * 100:.2f}%")
        
        st.markdown("---")
        st.header("Technical Analysis Indicators")

        # Moving Averages Calculation
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        # RSI Calculation (14 day window)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Plot SMA Chart
        fig_sma, ax_sma = plt.subplots(figsize=(10, 4))
        ax_sma.plot(df.index, df['Close'], label='Close Price', alpha=0.5)
        ax_sma.plot(df.index, df['SMA_50'], label='50-Day SMA', color='orange')
        ax_sma.plot(df.index, df['SMA_200'], label='200 Day SMA', color='red')
        ax_sma.set_title(f"{ticker} Moving Averages")
        ax_sma.legend()
        ax_sma.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_sma)
        
        # Plot RSI Chart
        fig_rsi, ax_rsi = plt.subplots(figsize=(10, 3))
        ax_rsi.plot(df.index, df['RSI'], color='purple', label='RSI (14)')
        ax_rsi.axhline(70, color='red', linestyle='--', label='Overbought (70)')
        ax_rsi.axhline(30, color='green', linestyle='--', label='Oversold (30)')
        ax_rsi.set_title(f"{ticker} Relative Strength Index (RSI)")
        ax_rsi.legend(loc='lower left')
        ax_rsi.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_rsi)
        
        st.markdown("---")
        st.header("Visual Analysis Dashboard")
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        # Top Panel: Price
        ax1.plot(df.index, df['Close'], color="#D20FB5", linewidth=1.5, label='Close Price ($)')
        ax1.set_title(f"{ticker} Historical Performance", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Price ($)") 
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Middle Panel: Daily Returns
        ax2.plot(df.index, df['Daily_Return'], color="#79ee9c", alpha=0.6, label='Daily Returns')
        ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax2.set_ylabel("Daily Change")
        ax2.legend(loc='upper left')
        ax2.grid(True, linestyle='--', alpha=0.5)   
        
        # Bottom Panel: Drawdown
        ax3.fill_between(df.index, df['Drawdown'] * 100, 0, color="#DC0B0B", alpha=0.4, label='Drawdown (%)')
        ax3.set_ylabel("Drawdown %")
        ax3.set_xlabel("Date")
        ax3.legend(loc='lower left')
        ax3.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.header("Multi-Stock Portfolio Comparison")

        tickers = st.multiselect(
            "Select stocks to compare:",
            [
                "AAPL", "MSFT", "AMZN", "GOOGL", "META", "SPY", "NFLX", "AMD", "QQQ", "JPM",
                "BRK-B", "GE", "DIS", "NVDA", "HD", "SBUX", "MS", "NKE", "WMT", "CRWD",
                "AMC", "NU", "SPCX", "GRAB", "PLUG", "AAL", "NOK", "SOFI", "ONDS", "PATH",
                "AGNC", "RKT", "F", "WBD", "AUR", "HL", "RIG", "KOD", "CDE", "IONQ",
                "TSLA", "IVZ", "HAYW", "NWSA", "PHYS", "GILD", "SO", "ADBE", "CRM", "CSCO",
                "ORCL", "INTU", "INTC", "AVGO", "QCOM", "TXN", "MU", "BAC", "WFC", "C",
                "GS", "BLK", "V", "MA", "PYPL", "HOOD", "JNJ", "LLY", "PFE", "MRK",
                "UNH", "ABBV", "TMO", "COST", "TGT", "LOW", "MCD", "KO", "PEP", "CMG",
                "SONY", "CAT", "HON", "MMM", "XOM", "CVX", "GM", "RIVN", "LCID", "VOO",
                "IVV", "IWM", "DIA", "VTI", "PG", "TJX", "LULU", "ABT", "FDX", "UPS",
                "PLTR", "TSM", "UBER", "PANW", "SHOP", "SPOT", "SQ", "SNOW", "DELL", "LUV"
            ],
            default=["AAPL", "MSFT", "SPY"],
        )

        st.markdown("---")
        st.header(f"Company Profile & Fundamentals for {ticker}")

        info = fetch_ticker_info(ticker)

        fund_col1, fund_col2, fund_col3, fund_col4 = st.columns(4)

        market_cap = info.get('marketCap', 'N/A')
        if isinstance(market_cap, (int, float)):
            market_cap = f"${market_cap / 1e9:.2f}B"

        pe_ratio = info.get('trailingPE', 'N/A')
        if isinstance(pe_ratio, (int, float)):
            pe_ratio = f"{pe_ratio:.2f}"

        div_yield = info.get('trailingAnnualDividendYield', 'N/A')
        if isinstance(div_yield, (int, float)):
            div_yield = f"{div_yield * 100:.2f}%"

        fifty_two_high = info.get('fiftyTwoWeekHigh', 'N/A')
        if isinstance(fifty_two_high, (int, float)):
            fifty_two_high = f"${fifty_two_high:.2f}"

        fund_col1.metric("Market Cap", market_cap)
        fund_col2.metric("P/E Ratio", pe_ratio)
        fund_col3.metric("Dividend Yield", div_yield)
        fund_col4.metric("52-Week High", fifty_two_high)

        with st.expander("Company Business Summary"):
            st.write(f"Displaying available fundamentals for {info.get('longName', ticker)} ({info.get('fullExchangeName', 'N/A')}).")

        st.markdown("---")
        st.header("Interactive Technical Analysis (PlotLy)")

        # Bollinger Bands
        df['20_SMA'] = df['Close'].rolling(window=20).mean()
        df['20_STD'] = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['20_SMA'] + (df['20_STD'] * 2)
        df['Lower_Band'] = df['20_SMA'] - (df['20_STD'] * 2)

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        fig_candle = g_obj.Figure()
        fig_candle.add_trace(g_obj.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price OHLC"
        ))

        fig_candle.add_trace(g_obj.Scatter(x=df.index, y=df['Upper_Band'], line=dict(color='grey', width=1), name='Upper Band'))
        fig_candle.add_trace(g_obj.Scatter(x=df.index, y=df['Lower_Band'], line=dict(color='grey', width=1), name='Lower Band'))

        fig_candle.update_layout(
            title=f"{ticker} Interactive Price & Bollinger Bands",
            yaxis_title="Stock Price($)",
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig_candle, use_container_width=True)

        fig_macd = g_obj.Figure()
        fig_macd.add_trace(g_obj.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue', width=1.5), name='MACD'))
        fig_macd.add_trace(g_obj.Scatter(x=df.index, y=df['Signal_Line'], line=dict(color='orange', width=1.5), name='Signal Line'))
        fig_macd.update_layout(title="MACD (Moving Average Convergence Divergence)", yaxis_title='Value', height=300)
        st.plotly_chart(fig_macd, use_container_width=True)

        # Portfolio Optimization
        st.markdown("---")
        st.header("Portfolio Optimization (Max Sharpe Ratio)")

        opt_tickers = st.multiselect("Select assets for portfolio optimization:", ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "SPY"], default=["AAPL", "MSFT", "SPY"])

        if len(opt_tickers) >= 2:
            opt_data = fetch_multiple_symbols(opt_tickers, start_date, end_date)
            if not opt_data.empty:
                opt_returns = opt_data.pct_change().dropna()
                mean_returns = opt_returns.mean() * 252
                cov_matrix = opt_returns.cov() * 252

                def negative_sharpe(weights):
                    p_return = np.sum(mean_returns * weights)
                    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    return -(p_return - risk_free_rate) / p_vol if p_vol != 0 else 0

                num_assets = len(opt_tickers)
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bounds = tuple((0, 1) for _ in range(num_assets))
                initial_weights = num_assets * [1. / num_assets,]

                opt_results = minimize(negative_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

                if opt_results.success:
                    optimal_weights = opt_results.x
                    st.subheader("Optimal Asset Weights (Max Sharpe Ratio)")

                    weight_col1, weight_col2 = st.columns(2)
                    with weight_col1:
                        for t_symbol, w_val in zip(opt_tickers, optimal_weights):
                            st.write(f"**{t_symbol}:** {w_val * 100:.2f}%")

                    with weight_col2:
                        opt_return = np.sum(mean_returns * optimal_weights)
                        opt_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
                        opt_sharpe = (opt_return - risk_free_rate) / opt_vol if opt_vol != 0 else 0

                        st.metric("Expected Portfolio Return", f"{opt_return * 100:.2f}%")
                        st.metric("Expected Portfolio Volatility", f"{opt_vol * 100:.2f}%")
                        st.metric("Maximized Sharpe Ratio", f"{opt_sharpe:.2f}")

        # Downside Risk Analytics
        st.markdown("---")
        st.subheader(f"Downside Risk Analytics for {ticker}")

        downside_returns = clean_returns[clean_returns < 0]
        downside_volatility = downside_returns.std() * (252 ** 0.5)
        sortino_ratio = (annual_return - risk_free_rate) / downside_volatility if downside_volatility != 0 else 0.0

        is_drawdown = df['Drawdown'] < 0
        drawdown_periods = (~is_drawdown).cumsum()[is_drawdown]
        max_drawdown_days = drawdown_periods.value_counts().max() if not drawdown_periods.empty else 0

        col_sort1, col_sort2, col_sort3 = st.columns(3)
        col_sort1.metric("Sortino Ratio", f"{sortino_ratio:.2f}", help="Measures risk-adjusted return relative to downside volatility.")
        col_sort2.metric("Downside Volatility", f"{downside_volatility * 100:.2f}%")
        col_sort3.metric("Max Drawdown Duration", f"{max_drawdown_days} Days", help="Longest consecutive period spent in drawdown.")

        if 'optimal_weights' in locals() and len(opt_tickers) >= 2:
            st.subheader("Optimal Asset Allocation Visualizer")
            pie_fig = g_obj.Figure(data=[g_obj.Pie(
                labels=opt_tickers,
                values=optimal_weights * 100,
                hole=.4,
            )])
            pie_fig.update_layout(title_text="Optimal Portfolio Weight Distribution")
            st.plotly_chart(pie_fig, use_container_width=True)

        # Benchmark Overlay
        st.markdown("---")
        st.subheader(f"{ticker} vs. S&P 500 (SPY) Relative Growth")

        spy_data = fetch_yahoo_data("SPY", start_date, end_date)

        if not spy_data.empty and not df.empty:
            norm_stock = (df['Close'] / df['Close'].iloc[0]) * 100
            norm_spy = (spy_data['Close'] / spy_data['Close'].iloc[0]) * 100

            benchmark_fig = g_obj.Figure()
            benchmark_fig.add_trace(g_obj.Scatter(x=df.index, y=norm_stock, name=ticker, line=dict(width=2)))
            benchmark_fig.add_trace(g_obj.Scatter(x=spy_data.index, y=norm_spy, name="S&P 500 (SPY)", line=dict(dash='dash', color='grey')))

            benchmark_fig.update_layout(
                yaxis_title="Growth of $100 Initial Investment",
                xaxis_title="Date",
                hovermode="x unified"
            )
            st.plotly_chart(benchmark_fig, use_container_width=True)

        # Multi Ticker Visualizations
        if tickers:
            comp_data = fetch_multiple_symbols(tickers, start_date, end_date)
            if not comp_data.empty:
                st.subheader("Cross-Asset RSI Momentum Tracker")
                delta_c = comp_data.diff()
                gain_c = (delta_c.where(delta_c > 0, 0)).rolling(window=14).mean()
                loss_c = (-delta_c.where(delta_c < 0, 0)).rolling(window=14).mean()
                rs_c = gain_c / loss_c
                rsi_df = 100 - (100 / (1 + rs_c))

                latest_rsi = rsi_df.iloc[-1].to_frame(name="Latest RSI (14)").T

                fig_rsi_hm, ax_rsi_hm = plt.subplots(figsize=(8, 1.5))
                sns.heatmap(latest_rsi, annot=True, fmt=".1f", cmap="RdYlGn_r", vmin=20, vmax=80, cbar=False, ax=ax_rsi_hm)
                plt.title("Current 14-day RSI Levels (<30 Oversold, >70 Overbought)")
                st.pyplot(fig_rsi_hm)

                normalized_df = (comp_data / comp_data.iloc[0]) * 100
                st.subheader("Normalized Performance (Starting at $100)")
                st.line_chart(normalized_df)

                comp_returns = comp_data.pct_change().dropna()
                correlation_matrix = comp_returns.corr()

                num_tickers = len(tickers)
                fig_size = max(6, num_tickers * 0.7)
                fig_corr, ax_corr = plt.subplots(figsize=(fig_size, fig_size * 0.7))

                sns.heatmap(
                    correlation_matrix,
                    annot=True,
                    fmt=".2f",
                    annot_kws={"size": max(7, 12 - num_tickers)},
                    cmap="coolwarm",
                    vmin=-1,
                    vmax=1,
                    linewidths=0.5,
                    ax=ax_corr,
                )
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                st.pyplot(fig_corr)

        # Monte Carlo Simulation
        st.markdown("---")
        st.header("Monte Carlo Risk Simulation (30-Day Outlook)")

        if st.button("Run 1000 Path Simulation"):
            num_simulations = 1000
            time_horizon = 30

            last_price = df['Close'].iloc[-1]
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
            ax_mc.plot(simulation_matrix, color='blue', alpha=0.03)
            ax_mc.set_title(f"1,000 Simulated Price Paths for {ticker} over Next 30 Days")
            ax_mc.set_ylabel("Simulated Prices ($)")
            ax_mc.set_xlabel("Days Ahead")
            st.pyplot(fig_mc)

            ending_prices = simulation_matrix[-1, :]
            percentile_5th = np.percentile(ending_prices, 5)
            max_expected_loss_pct = ((percentile_5th - last_price) / last_price) * 100

            st.error(f"**95% Value at Risk (VaR):** There is a 5% chance {ticker} drops below **${percentile_5th:.2f}** over the next 30 days (a loss of **{max_expected_loss_pct:.2f}%**).")

            st.markdown("---")
            st.header(f"Live Sentiment Analysis for {ticker}")

            try:
                sia = SentimentIntensityAnalyzer()
            except Exception:
                nltk.download('vader_lexicon', quiet=True)
                sia = SentimentIntensityAnalyzer()

            sample_headlines = [
                f"{ticker} reports steady Q3 growth amid market fluctuations.",
                f"Analysts highlight potential risks and headwinds for {ticker} stock.",
                f"Institutional investors adjust holdings in {ticker} ahead of earnings."
            ]

            compound_scores = []
            st.subheader("Recent Market Headline Samples")

            for headline in sample_headlines:
                sentiment = sia.polarity_scores(headline)
                compound = sentiment['compound']
                compound_scores.append(compound)
                
                st.markdown(f"• **{headline}**")
                st.caption(f"Sentiment Score: {compound:.2f}")

            avg_score = np.mean(compound_scores)
            st.subheader("Overall Market Sentiment")

            if avg_score >= 0.05:
                st.success(f"**Overall Sentiment: Bullish** (Average Score: {avg_score:.2f})")
            elif avg_score <= -0.05:
                st.error(f"**Overall Sentiment: Bearish** (Average Score: {avg_score:.2f})")
            else:
                st.warning(f"**Overall Sentiment: Neutral** (Average Score: {avg_score:.2f})")
