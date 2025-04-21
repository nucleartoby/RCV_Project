import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import plotly.express as px

ticker = 'sie.de'

ticker_data = yf.Ticker(ticker)

hiso_data = ticker_data.history(period='5y', interval='1d')

px.line(hiso_data, x=hiso_data.index, y='Close', title=f'{ticker} 1Y Price History').show()
hiso_data['Close'].plot(title=f'{ticker} 1Y Price History', figsize=(10, 5))