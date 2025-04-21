import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import plotly.express as px

ticker = 'SIE.DE' # ticker for the stock to analyse

ticker_data = yf.Ticker(ticker) # function to get the ticker data

histo_data = ticker_data.history(period='5y', interval='1d') # function to select the period and interval

histo_data.reset_index(inplace=True) # needed to reset the index for date column
histo_data.drop(columns=['Dividends', 'Stock Splits'], inplace=True)

df = histo_data.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}) # formatting for the model

df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col])

print(df)

df.to_csv('price_data.csv', index=False)

##px.line(hiso_data, x=hiso_data.index, y='Close', title=f'{ticker} 1Y Price History').show() # visualisations
##hiso_data['Close'].plot(title=f'{ticker} 1Y Price History', figsize=(10, 5))