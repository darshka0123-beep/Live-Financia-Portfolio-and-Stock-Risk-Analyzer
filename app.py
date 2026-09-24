import yfinance as yf

def load_prices(tickers, period):
    raw = yf.download(tickers, period=period)
    prices = raw["Close"]
    return prices.dropna()
prices = load_prices(["AAPL", "MSFT", "TSLA"], "1y")
print(prices.shape)
print(prices.head())
print(prices.columns)

def to_returns(prices): 
    returns = prices.pct_change()
    return returns.dropna()

returns = to_returns(prices)
print(returns.shape)
print(returns.head())

vol = returns.std() * math.sqrt(252)

