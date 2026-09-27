import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
            ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "SPY", "NFLX", "AMD", "QQQ", "JPM", "BRK-B", "GE", "DIS"],
            default=["AAPL", "MSFT", "SPY"],
        )

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
            plt.xticker(rotation=45, ha='right')
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
