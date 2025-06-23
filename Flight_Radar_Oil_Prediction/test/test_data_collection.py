import yfinance as yf

brent = yf.Ticker("BZ=F")
texas = yf.Ticker("CL=F")

hist = brent.history(period="6mo", interval="1d")

