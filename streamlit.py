import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
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

# Data fetching function
@st.cache_data
def get_stock_data(symbol, start, end):
    stock = yf.Ticker(symbol)
    df = stock.history(start=start)
    if not df.empty:
        df = df[['Close']].copy()
    return df

# Run Analysis when user presses the button
# Session State intitialized
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

# Update when the sidebar is pressed
if st.sidebar.button("Run Risk Analysis"):
    st.session_state.run_analysis = True

# Check Session State instead of checking the button directly
if st.session_state.run_analysis:
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
        annual_return = clean_returns.mean() * 252
        annual_volatility = clean_returns.std() * (252 ** 0.5)
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
        st.header("Technical Analysis Indicators")

        # Moving Averages Calculation
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        # RSI Calculation (14 day window)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0,0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Plot SMA Chart
        fig_sma, ax_sma = plt.subplots(figsize=(10,4))
        ax_sma.plot(df.index, df['Close'], label='Close Price', alpha=0.5)
        ax_sma.plot(df.index, df['SMA_50'], label='50-Day SMA', color='orange')
        ax_sma.plot(df.index, df['SMA_200'], label='200 Day SMA', color='red')
        ax_sma.set_title(f"{ticker} Moving Averages")
        ax_sma.legend()
        ax_sma.grid(True, linestyle='--', alpha=0.5)
        ax_sma.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_sma)
        
        # Plot RSI Chart
        fig_rsi, ax_rsi = plt.subplots(figsize=(10,3))
        ax_rsi.plot(df.index, df['RSI'], color='purple', label='RSI (14)')
        ax_rsi.axhline(70, color='red', linestyle='--', label='Overbrought (70)')
        ax_rsi.axhline(30, color='green', linestyle='--', label='Oversold (30)')
        ax_rsi.set_title(f"{ticker} Relative Strength Index (RSI)")
        ax_rsi.legend(loc='lower left')
        ax_rsi.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_rsi)
        
        st.markdown("---")
        st.header("Visual Analysis Dashboard")
       
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        # Top Panel: Price
        ax1.plot(df.index, df['Close'], color="#D20FB5", linewidth=1.5, label='Close Price ($)' )
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
        ax3.fill_between(df.index,df['Drawdown'] * 100, 0, color="#DC0B0B", alpha=0.4, label='Drawdown (%)')
        ax3.set_ylabel("Drawdown %")
        ax3.set_xlabel("Date")
        ax3.legend(loc='lower left')
        ax3.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        st.pyplot(fig)

        # Multiselect widget for comparing multiple tickers

        st.markdown("---")
        st.header("Multi-Stock Portfolio Comaparison")

        tickers = st.multiselect(
            "Select stocks to compare:",
            ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "SPY", "NFLX", "AMD", "QQQ", "JPM", "BRK-B", "GE", "DIS", "NVDA", "HD", "SBUX", "MS", "NKE", "WMT", "CRWD", "AMC", "NU", "SPCX", "GRAB", "PLUG", "AAL", "NOK", "SOFI", "ONDS", "PATH", "AGNC", "RKT", "F", "WBD", "AUR", "HL", "RIG", "KOD", "CDE", "IONQ", "TSLA", "IVZ", "HAYW", "NWSA", "PHYS", "GILD", "SO", "ADBE", "CRM", "CSCO", "ORCL", "INTU","INTC" "AVGA", "QCOM", "TXN", "MU", "BAC", "WFC", "C", "GS", "BLK", "V", "MA", "PYPL", "HOOD", "JNJ", "LLY", "PFE", "MRK", "UNH", "ABV", "TMO", "COST", "TGT", "LOW", "MCD", "KO", "PEP", "CMG", "SONY", "CAT", "HON", "MMM", "XOM", "CVX", "GM", "RIVN", "LCID", "VOO", "IVV", "IWM", "DIA", "VTI", "PG", "TJX", "LULU", "ABT", "FEDX", "UPS", "PLTR", "TSM", "UBER", "PANW", "SHOP", "SPOT", "SQ", "SNOW", "DELL", "LUV", "WOOF", "CAKE", "EAT", "FIZZ", "TAP", "HOG", "FUN", "UNP", "AXP", "NEE", "DE", "LMT", "RTX", "COP", "VUG", "VYM", "XLK", "XLF", "SCHD", "RACE", "MAR", "BKNG", "ABNB", "DAL", "REGN", "ISRG", "ZTS", "CVS", "MDT"],
            default=["AAPL", "MSFT", "SPY"],
        )

        st.markdown("---")
        st.header(f"Company Profile & Fundamentals for {ticker}")

        stock_info = yf.Ticker(ticker).info

        # Company Metrics Summary
        fund_col1, fund_col2, fund_col3, fund_col4 = st.columns(4)

        market_cap = stock_info.get('trailingPE', 'N/A')
        if isinstance(market_cap, (int, float)):
            market_cap = f"${market_cap / 1e9:.2f}B"

        pe_ratio = stock_info.get('trailingPE', 'N/A')
        if isinstance(pe_ratio, (int, float)):
            pe_ratio = f"{pe_ratio:.2f}"

        div_yield = stock_info.get('dividendYield', 'N/A')
        if isinstance(pe_ratio, (int, float)):
            div_yield = f"{div_yield * 100:.2f}%"

        fifty_two_high = stock_info.get('fiftyTwoWeekHigh', 'N/A')
        if isinstance(fifty_two_high, (int, float)):
            fifty_two_high = f"${fifty_two_high:.2f}"

        fund_col1.metric("Market Cap", market_cap)
        fund_col2.metric("P/E Ratio", pe_ratio)
        fund_col3.metric("Dividend Yield", div_yield)
        fund_col4.metric("52-Week High", fifty_two_high)

        with st.expander("Company Business Summary"):
            st.write(stock_info.get('longBusinessSummary', 'No summary available.'))

        # Interactive Candlestick Chart and Bollinger Bands and MACD
        st.markdown("---")
        st.header("Interactive Technical Analysis (PlotLy)")

        # Calculate Bollinger Bands
        df['20_SMA'] = df['Close'].rolling(window=20).mean()
        df['20_STD'] = df['Close'].rolling(window=20).mean()
        df['Upper_Band'] = df['20_SMA'] + (df['20_STD'] * 2)
        df['Lower_Band'] = df['20_SMA'] - (df['20_STD'] * 2)

        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Interactive Candlestick Plot
        fig_candle = g_obj.Figure()

        # Add Candlestick trace (requires Open, High, Low, Close from yfinance)
        full_stock_df = yf.Ticker(ticker).history(start=start_date, end=end_date)
        if not full_stock_df.empty:
            fig_candle.add_trace(g_obj.Candlestick(
                x=full_stock_df.index,
                open=full_stock_df['Open'],
                high=full_stock_df['High'],
                low=full_stock_df['Low'],
                close=full_stock_df['Close'],
                name="Price OHLC"
            ))

        # Overlay Bollinger Bands
        fig_candle.add_trace(g_obj.Scatter(x=df.index, y=df['Upper_Band'], line=dict(color='grey', width=1), name='Upper Band'))
        fig_candle.add_trace(g_obj.Scatter(x=df.index, y=df['Lower_Band'], line=dict(color='grey', width=1), name='Lower Band'))

        fig_candle.update_layout(
            title=f"{ticker} Interactive Price & Bollinger Bands",
            yaxis_title="Stock Price($)",
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig_candle, use_container_width=True)

        # Plot MACD Subchart
        fig_macd = g_obj.Figure()
        fig_macd.add_trace(g_obj.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue', width=1.5), name='MACD'))
        fig_macd.add_trace(g_obj.Scatter(x=df.index, y=df['Signal_Line'], line=dict(color='orange', width=1.5), name='Signal Line'))
        fig_macd.update_layout(title="MACD (Moving Average Convergence Divergence)", yaxis_title='Value', height=300)
        st.plotly_chart(fig_macd, use_container_width=True)

        # Efficient Frontier & Portfolio Optimization
        st.markdown("---")
        st.header("Portfolio Optimization (Max Sharpe Ratio)")

        opt_tickers = st.multiselect("Select assets for portfolio optimization:", ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "SPY"], default=["AAPL", "MSFT", "SPY"])

        if len(opt_tickers) >=2:
            opt_data = yf.download(opt_tickers, start=start_date, end=end_date)['Close']
            opt_returns = opt_data.pct_change().dropna()
            mean_returns = opt_returns.mean() * 252
            cov_matrix = opt_returns.cov() * 252

            # Optimization function
            def negative_sharpe(weights):
                p_return = np.sum(mean_returns * weights)
                p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return -(p_return - risk_free_rate) / p_vol
            num_assets = len(opt_tickers)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) -1})
            bounds = tuple((0,1) for _ in range(num_assets))
            initial_weights = num_assets * [1. / num_assets,]

            opt_results = minimize(negative_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

            if opt_results.success:
                optimal_weights = opt_results.x
                st.subheader("Optimal Asset Weights (Max Sharpe Ratio)")

                weight_col1, weight_col2 = st.columns(2)
                with weight_col1:
                    for t_symbol, w_val in zip(opt_tickers, optimal_weights):
                        st.write(f"**{t_symbol}:**{w_val} * 100:.2f")

                with weight_col2:
                    opt_return = np.sum(mean_returns * optimal_weights)
                    opt_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
                    opt_sharpe = (opt_return - risk_free_rate) / opt_vol

                    st.metric("Expected Portfolio Return", f"{opt_return * 100:.2f}%")
                    st.metric("Expected Portfolio Volatility", f"{opt_vol * 100:.2f}%")
                    st.metric("Maximized Sharpe Ratio", f"{opt_sharpe:.2f}")

        # Advanced Downside Risk Metrics
        st.markdown("---")
        st.subheader(f"Downside Risk Analytics for {ticker}")

        # Sortino Ratio (Focuses only on downside volatility)
        downside_returns = clean_returns[clean_returns < 0]
        downside_volatility = downside_returns.std() * (252 ** 0.5)

        if downside_volatility != 0:
            sortino_ratio = (annual_return - risk_free_rate) / downside_volatility
        else:
            sortino_ratio = 0.0

        # Maximum Drawdown Duration (In Trading Days)
        is_drawdown = df['Drawdown'] < 0
        drawdown_periods = (~is_drawdown).cumsum()[is_drawdown]
        if not drawdown_periods.empty:
            max_drawdown_days = drawdown_periods.value_counts().max()
        else:
            max_drawdown_days = 0

        col_sort1, col_sort2, col_sort3 = st.columns(3)
        col_sort1.metric("Sortino Ratio", f"{sortino_ratio:.2f}", help="Measures risk-adjusted return relative to downside volatility.")
        col_sort2.metric("Downside Volatility", f"{downside_volatility * 100:.2f}%")
        col_sort3.metric("Max Drawdown Duration", f"{max_drawdown_days} Days", help="Longest consecutive period spent in drawdown.")

        # Portfolio Allocation Donut Chart (Using g_obi)
        if 'optimal_weights' in locals() and len(opt_tickers) >= 2:
            st.subheader("Optimal Asset Allocation Visualizer")

            pie_fig = g_obj.Figure(data=[g_obj.Pie(
                labels=opt_tickers,
                values=optimal_weights * 100,
                hole=.4,
            )])
            pie_fig.update_layout(title_text="Optimal Portfolio Weight Distribution")
            st.plotly_chart(pie_fig, use_container_width=True)

        # Benchmark Comparison Overlay (Using g_obj)
        st.markdown("---")
        st.subheader(f"{ticker} vs. S&P 500 (SPY) Relative Growth")

        spy_data = yf.Ticker("SPY").history(start=start_date, end=end_date)[['Close']]

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

        # Institutional Fundamentals Section
        st.markdown("---")
        st.header(f"Detailed Valuation & Financial Health for {ticker}")

        info = yf.Ticker(ticker).info

        f_col1, f_col2, f_col3, f_col4 = st.columns(4)

        # Safely pulll financial ratios
        forward_pe = info.get('forwardPE', 'N/A')
        peg_ratio = info.get('pegRatio', 'N/A')
        price_to_book = info.get('priceToBook', 'N/A')
        profit_margins = info.get('profitMargins', 'N/A')

        if isinstance(forward_pe, (int, float)): forward_pe = f"{forward_pe:.2f}"
        if isinstance(peg_ratio, (int, float)): peg_ratio = f"{peg_ratio:.2f}"
        if isinstance(price_to_book, (int, float)): price_to_book = f"{price_to_book:.2f}"
        if isinstance(profit_margins, (int, float)): profit_margins = f"{profit_margins * 100:.2f}%"

        f_col1.metric("Forward P/E", forward_pe)
        f_col2.metric("PEG Ratio", peg_ratio)
        f_col3.metric("Price to Book", price_to_book)
        f_col4.metric("Profit Margin", profit_margins)

        # RSI Cross Asset Comparison
        if 'tickers' in locals() and len(tickers) >= 2:
            st.subheader("Cross-Asset RSI Momentum Tracker")

            comp_data = yf.download(tickers, start=start_date, end=end_date)['Close']

            # Calculate 14 day RSI for all assets
            delta = comp_data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs= gain / loss
            rsi_df = 100 - (100 / (1 + rs))

            latest_rsi = rsi_df.iloc[-1].to_frame(name="Latest RSI (14)").T

            fig_rsi_hm, ax_rsi_hm = plt.subplots(figsize=(8,1.5))
            sns.heatmap(latest_rsi, annot=True, fmt=".1f", cmap="RdYlGn_r", vmin=20, vmax=80, cbar=False, ax=ax_rsi_hm)
            plt.title("Current 14-day RSI Levels (<30 Oversold, >70 Overbought)")
            st.pyplot(fig_rsi_hm)

        if tickers:
            # Download data for all the selected tickers at the same time
            comparison_df = yf.download(tickers, start=start_date, end=end_date)['Close']

            # Normalize prices to start at $100 so they can be compared fairly
            normalized_df = (comparison_df / comparison_df.iloc[0]) * 100

            # Plot normalized growth chart
            st.subheader("Normalized Performance (Starting at $100)")
            st.line_chart(normalized_df)

            # Calculate daily returns for correlation
            comp_returns = comparison_df.pct_change().dropna()
            correlation_matrix = comp_returns.corr()

            # Display Correlation Heatmap
            num_tickers = len(tickers)
            fig_size = max(6, num_tickers * 0.7) # Grows larger when more tickers are selected

            fig_corr, ax_corr = plt.subplots(figsize=(fig_size, fig_size * 0.7))

            # Add fmt=".2f" to round numbers, annot_kws to shrink  text size, and cbar_kws to scale the colorbar
            sns.heatmap(
                correlation_matrix,
                annot=True,
                fmt=".2f",
                annot_kws={"size": max(7, 12 - num_tickers)}, # Shrinks text as you add more tickets to fit it
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                linewidths=0.5,
                ax=ax_corr,
            )

            # Rotate labels so they don't overlap on the axes
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)

            st.pyplot(fig_corr)

        # Monte Carlo Risk Simulation
        st.markdown("---")
        st.header("Monte Carlo Risk Simulation (30-Day Outlook)")

        if st.button("Run 1000 Path Simulation"):
            num_simulations = 1000
            time_horizon = 30 # forecast 30 days into the future

            last_price = df['Close'].iloc[-1]
            daily_mean = clean_returns.mean()
            daily_std = clean_returns.std()

            # Matrix to store simulation results
            simulation_matrix = np.zeros((time_horizon, num_simulations))

            for i in range(num_simulations):
                prices = [last_price]
                for t in range(1, time_horizon):
                    # Generate random return using normal distribution
                    simulated_return = np.random.normal(daily_mean, daily_std)
                    prices.append(prices[-1] * (1 + simulated_return))
                simulation_matrix[:, i] = prices
            st.session_state['simulation_matrix'] = simulation_matrix

            # Plot Simulation Paths
            fig_mc, ax_mc = plt.subplots(figsize=(10,5))
            ax_mc.plot(simulation_matrix, color='blue', alpha=0.03)
            ax_mc.set_title(f"1,000 Simulated Price Paths for {ticker} over Next 30 Days")
            ax_mc.set_ylabel("Simulated Prices ($)")
            ax_mc.set_xlabel("Days Ahead")
            st.pyplot(fig_mc)

            # Calculate 95% Value at Risk (VaR)
            ending_prices = simulation_matrix[-1, :]
            percentile_5th = np.percentile(ending_prices, 5)
            max_expected_loss_pct = ((percentile_5th - last_price) / last_price) * 100

            st.error(f"**95% Value at Risk (VaR):** There is a 5% chance {ticker} drops below **${percentile_5th:.2f}** over the next 30 days (a loss of **{max_expected_loss_pct:.2f}%**).")

            # Download VADER lexicon (runs once automatically)
            nltk.download('vader_lexicon', quiet=True)

            st.markdown("---")
            st.header(f"Live News & Sentiment Analysis for {ticker}")

            # Fetch stock news from yfinance
            stock = yf.Ticker(ticker)
            news_list = stock.news

            if not news_list:
                st.info(f"No recent news found or {ticker}.")
            else:
                sia = SentimentIntensityAnalyzer()
                compound_scores = []

                st.subheader("Latest Headlines")

                for item in news_list[:5]: 
                    content = item.get('content', item)

                    title = content.get('title', 'No Title')
                    publisher = content.get('provider', {}).get('displayName', 'Unknown')

                    # Extract article link
                    click_through = content.get('canonicalUrl', {})
                    link = click_through.get('url', '#') if isinstance(click_through, dict) else '#'

                    # Analyze headline sentiment
                    sentiment = sia.polarity_scores(title)
                    compound = sentiment['compound']
                    compound_scores.append(compound)

                    # Categorize sentiment and set custom badge colors
                    if compound >= 0.05:
                        sentiment_label = "Bullish"
                        color_hex = "#28a745" # Green
                    elif compound <= -0.05:
                        sentiment_label = "Bearish"
                        color_hex = "#dc3545" # Red
                    else: 
                        sentiemnt_label = "Neutral"
                        color_hex = "#6c757d" # Grey

                    # Display Headline entry
                    col1, col2 = st.columns([4,1])
                    with col1:
                        st.markdown(f"**[{title}]({link})**")
                        st.caption(f"Score: {compound:.2f}")

                    st.markdown("---")

                # Overall Sentiment Summary Badge
                if compound_scores:
                    avg_score = np.mean(compound_scores)
                    st.subheader("Overall Market Sentiment")

                    if avg_score >= 0.05:
                        st.success(f"**Overall Sentiment: Bullish** (Average Score: {avg_score:.2f})")
                    elif avg_score<= - 0.05:
                        st.error(f"**Overall Sentiment: Bearish** (Average Score: {avg_score:.2f})")
                    else:
                        st.warning(f"**Overall Sentiment: Neutral** (Average Score: {avg_score:.2f})")

    
