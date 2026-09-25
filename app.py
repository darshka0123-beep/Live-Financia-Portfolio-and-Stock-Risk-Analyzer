import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Financial Portfolio and Risk Analyzer",
    page_icon="📊", 
    layout="wide",
)

st.title("Live Portfolio and Stock Risk Analyzer")
st.markdown(
    "Analyze historical performance, risk metrics (Beta, Sharpe Ratio), and simulate future portfolio growth."
)

# Sidebar Inputs
st.sidebar.header("Portfolio Configuration")
tickers_input = st.sidebar.text_input(
    "Stock Tickers (comma seperated)", "AAPL, MSFT, GOOGL, AMZN, TSLA"
)

tickers = [
    t.strip().upper() for t in tickers_input.split(",") if t.strip() != ""
]

weights_input = st.sidebar.text_input(
    "Portfolio Weights (must sum to 1.0)", "0.20, 0.20, 0.20, 0.20, 0.20"
)
try: 
    weights = [float(w.strip()) for w in weights_input.split(",")]
except ValueError:
    st.sidebar.error("Weights must be numbers seperated by commas.")
    st.stop()

if len(tickers) != len(weights):
    st.sidebar.error("The number of tickers must match the number of weights!")
    st.stop()

if not np.isclose(sum(weights), 1.0):
    weights = [w / sum(weights) for w in weights]

time_horizon = st.sidebar.selectbox(
    "Historical Data Range", ["1y", "2y", "5y"], index=1
)
risk_free_rate = (
    st.sidebar.number_input(
        "Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=4.25
    )
    / 100
)

# Fetch Data 
@st.cache_data(ttl=3600)
def fetch_data(ticker_list, period):
    # Fetch user assets plus ^GSPC for benchmark calculations
    all_tickers = list(set(ticker_list + ["^GSPC"]))
    data = yf.download(all_tickers, period=period)["Close"]
    return data

with st.spinner("Fetching market data..."):
    df_raw = fetch_data(tickers, time_horizon)

if df_raw.empty:
    st.error("Failed to retrieve market data. Check ticker symbols.")
    st.stop()

df_prices = df_raw.dropna()
returns = df_prices.pct_change().dropna()

if df_raw.empty:
    st.error("Failed to retrieve market data. Check ticker symbols.")
    st.stop()

sp500_returns = returns["^GSPC"]
asset_returns = returns[tickers]
portfolio_returns = (asset_returns * weights).sum(axis=1)

# Risk Metrics
trading_days = 252
annual_return = portfolio_returns.mean() * trading_days
annual_volatility = portfolio_returns.std() * np.sqrt(trading_days)
sharpe_ratio = (
    (annual_return - risk_free_rate) / annual_volatility
    if annual_volatility > 0
    else 0
)

covariance = np.cov(portfolio_returns, sp500_returns)[0][1]
sp500_variance = np.var(sp500_returns)
portfolio_beta = covariance / sp500_variance if sp500_variance > 0 else 1.0

# Display Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Expected Annual Return", f"{annual_return * 100:.2f}%")
col2.metric("Annual Volatility (Risk)", f"{annual_volatility * 100:.2f}%")
col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
col4.metric("Portfolio Beta (vs. S&P 500)", f"{portfolio_beta: .2f}")

st.markdown("---")

# Cumulative Growth Plot
st.subheader("Portfolio Historical Growth vs. S&P 500")
cum_portfolio = ( 1 + portfolio_returns).cumprod()
cum_sp500 = (1+ sp500_returns).cumprod()
df_cum = pd.DataFrame(
    {"Portfolio": cum_portfolio, "S&P 500 Benchmark": cum_sp500}
)

fig_growth = px.line(
    df_cum,
    labels={"value": "Growth (Base 1.0)", "index": "Date"}, 
    title="Grwoth of 1$ Investment",
)
st.plotly_chart(fig_growth, use_container_width=True)

st.markdown("---")
st.subheader(" Asset Correlation & Risk Breakdown")

col_corr1, col_corr2 = st.columns(2)

with col_corr1:
    # Asset Correlation Heatmap
    corr_matrix = asset_returns.corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Asset Correlation Matrix",
        aspect="auto",
    )
    st.plotly_chart(fig_corr,use_container_width=True)

with col_corr2:
    # Individual Asset Risk vs Return Scatter
    ind_returns = asset_returns.mean() * trading_days * 100
    ind_vol = asset_returns.std() * np.sqrt(trading_days) * 100

    df_ind = pd.DataFrame(
        {
            "Ticker": tickers,
            "Expected Return (%)": ind_returns.values,
            "Volatility (%)": ind_vol.values,
            "Weight": weights,
        }
    )

    fig_scatter = px.scatter(
        df_ind,
        x="Volatility (%)",
        y="Expected Return (%)", 
        size=[w*100 for w in weights],
        text="Ticker",
        title="Individual Risk vs. Return (Bubble Size = Weight)",
    )
    fig_scatter.update_traces(textposition="top center")
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")
st.subheader("Downside Risk & Drawdown Analysis")

# Calculate Historical Drawdown
cumulative_returns = (1 + portfolio_returns).cumprod()
running_max = cumulative_returns.cummax()
drawdown = (cumulative_returns - running_max) / running_max
max_drawdown = drawdown.min() * 100

# Value at Risk (VaR) & Conditional VaR (Expected Shortfall) at 95% Confidence
var_95 = np.percentile(portfolio_returns, 5) * 100
cvar_95 = portfolio_returns[portfolio_returns <= (var_95 / 100)].mean() * 100

col_d1, col_d2, col_d3 = st.columns(3)
col_d1.metric("Maximum Historical Drawdown", f"{max_drawdown:.2f}%")
col_d2.metric("Daily 95% Value at Risk (VaR)", f"{var_95:.2f}%")
col_d3.metric("Daily 95% Conditional VaR (CVaR)", f"{cvar_95:.2f}%")

# Plot Drawdown over time
fig_dd = px.area(
    x=drawdown.index,
    y=drawdown.values * 100,
    labels={"x": "Date", "y": "Drawdown (%)"},
    title="Historical Portfolio Drawdown Depth",
)
st.plotly_chart(fig_dd, use_container_width=True)

# Monte Carlo Simulation
st.markdown("---")
st.subheader("Monte Carlo Future Value Simulation")
col_mc1, col_mc2, col_mc3 = st.columns(3)
initial_investment = col_mc1.number_input(
    "Initial Investment ($)", value=10000, step=1000
)
sim_years = col_mc2.slider("Simulation Horizon (Years)", 1, 10, 5)
num_simulations = col_mc3.selectbox(
    "Number of Simulations", [100, 500, 1000], index=1
)

np.random.seed(42)
sim_days = sim_years * trading_days
mean_daily_return = portfolio_returns.mean()
vol_daily_return = portfolio_returns.std()

# Initialize Matrix
simulation_matrix = np.zeros((sim_days, num_simulations))
simulation_matrix[0] = initial_investment

for t in range(1, sim_days):
    random_shocks = np.random.normal(
        mean_daily_return, vol_daily_return, num_simulations
    )
    simulation_matrix[t] = simulation_matrix[t-1] * (1 + random_shocks)

fig_mc = go.Figure()
for i in range(min(num_simulations, 100)):
    fig_mc.add_trace(
        go.Scatter(
            y=simulation_matrix[:,i],
            mode="lines",
            line=dict(width=0.8),
            showlegend=False,
            opacity=0.3,
        )
    )

fig_mc.update_layout(
    title=f"{num_simulations} Simulated Portfolio Growth Paths over {sim_years} Years",
    xaxis_title="Trading Days",
    yaxis_title ="Portfolio Value ($)",
)


st.plotly_chart(fig_mc, use_container_width=True)

final_values = simulation_matrix[-1]
p10 = np.percentile(final_values, 10)
p50 = np.percentile(final_values, 50)
p90 = np.percentile(final_values, 90)

st.markdown(f"**Projected Portfolio Outcomes after {sim_years} Years:**")
col_res1, col_res2, col_res3 = st.columns(3)
col_res1.metric("Conservative (10th Percentile)", f"${p10:,.2f}")
col_res2.metric("Median Target (50th Percentile)", f"${p50:,.2f}")
col_res3.metric("Optimistic (90th Percentile)", f"${p90:,.2f}")

# MPT optimizer using scipy.optimize

import scipy.optimize as sco
# Function to calculate portfolio metrics for optimization
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = (returns - risk_free_rate) / std
    return returns, std, sharpe

# Optimization target: Maximize Sharpe Ratio (Minimize Negative Sharpe)
def min_func_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

# Constraints & Bounds
num_assets = len(tickers)
args = (asset_returns.mean(), asset_returns.cov(), risk_free_rate)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0.0, 1.0) for _ in range(num_assets))
initial_guess = num_assets * [1.0 / num_assets]

# Run Scipy Optimization
optimized_result = sco.minimize(min_func_sharpe, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
opt_weights = optimized_result.x

st.subheader("Automated Mean-Variance Portfolio Optimization")
df_opt = pd.DataFrame({"Ticker": tickers, "Current Weight": weights, "Optimal Weight (Max Sharpe)": opt_weights})
st.dataframe(df_opt.style.format({"Current Weight": "{:.2%}", "Optimal Weight (Max Sharpe)": "{:.2%}"}))

import io 

# -----------------------
# Stress Testing and Historical crises

st.markdown("---")
st.subheader("Historical Crisis Stress Testing")
st.markdown(
    "Evaluate how your current portfolio weights would have performed during major historical market shocks."
)

# Crises Windows Definition
crises = {
    "2008 Financial Crisis": ("2007-10-01", "2009-03-09"),
    "2020 COVID-19": ("2020-02-19", "2020-03-23"),
    "2022 Inflation & Rate Hike Bear Market": ("2022-01-03", "2022-10-12"),
}

selected_crisis = st.selectbox("Select Historical Crisis Window", list(crises.keys()))
start_date, end_date = crises[selected_crisis]

@st.cache_data(ttl=3600)
def fetch_crisis_data(ticker_list, start, end):
    all_tickers = list(set(ticker_list + ["^GSPC"]))
    return yf.download(all_tickers, start=start, end=end)["Close"]

df_crisis_raw = fetch_crisis_data(tickers, start_date, end_date)

if not df_crisis_raw.empty and len(df_crisis_raw.dropna()) > 5:
    df_crisis_prices = df_crisis_raw.dropna()
    crisis_returns = df_crisis_prices.pct_change().dropna()

    crisis_port_returns = (crisis_returns[tickers] * weights).sum(axis=1)
    crisis_sp500_returns = crisis_returns["^GSPC"]

    cum_crisis_port = (1 + crisis_port_returns).cumprod()
    cum_crisis_sp500 = (1 + crisis_sp500_returns).cumprod()

    df_crisis_cum = pd.DataFrame({
        "Portfolio": cum_crisis_port,
        "S&P 500 Benchmark": cum_crisis_sp500
    })

    # Calculate Peak-to-Trough Loss
    port_max_loss = ((cum_crisis_port.min() - cum_crisis_port.iloc[0]) / cum_crisis_port.iloc[0]) * 100
    sp500_max_loss = ((cum_crisis_sp500.min() - cum_crisis_sp500.iloc[0]) / cum_crisis_sp500[0]) * 100

    col_c1, col_c2 = st.columns(2)
    col_c1.metric("Portfolio Max Drop During Crisis", f"{port_max_loss:.2f}%")
    col_c2.metric("S&P 500 Max Drop During Crisis", f"{sp500_max_loss:.2f}%")

    fig_crisis = px.line(
        df_crisis_cum,
        labels={"value": "Normalized Growth (Base 1.0)", "index": "Date"},
        title=f"Portfolio Trajectory During {selected_crisis}"
    )
    st.plotly_chart(fig_crisis, use_container_width=True)
else: 
    st.warning(f"Insufficient historical price data available for selected assets during {selected_crisis}.")

# -------------------------------------------------------------------------------------------------------------
# Fundamentals and Report Export Pipeline
st.markdown("---")
st.subheader("Asset Fundamentals & Export Center")

col_f1, col_f2 = st.columns([2,1])

with col_f1:
    st.markdown("Fundamental Multiples Lookup")
    fundamentals = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            fundamentals.append({
                "Ticker": ticker,
                "Name": info.get("shortName", "N/A"),
                "Forward P/E": info.get("forwardPE", "N/A"),
                "Market Cap ($B)": round(info.get("marketCap", 0) / 1e9, 2) if info.get("marketCap") else "N/A"
                "Profit Margin (%)": round(info.get("profitMargins", 0) * 100, 2) if info.get("profitMargins") else "N/A"
            })
        except Exception:
            fundamentals.append({"Ticker": ticker, "Name": "N/A", "Forward P/E": "N/A", "Market Cap ($B)": "N/A", "Pro"})

                      