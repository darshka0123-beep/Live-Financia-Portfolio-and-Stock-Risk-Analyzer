import yfinance as yf

df = yf.Ticker("TSLA").history(period="1y")
print(df.head())

returns = df["Close"].pct_change().dropna()
print(returns.head())
print(returns.std())
