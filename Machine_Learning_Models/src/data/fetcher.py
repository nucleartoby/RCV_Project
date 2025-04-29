import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def market_data(ticker='^IXIC', vix_ticker='^VIX', 
                                treasury_ticker='^TNX', nasdaq_futures_ticker='NQ=F',
                                usd_index_ticker='DX-Y.NYB',
                                start_date=None, end_date=None):
    try:
        if end_date is None: # Set default date range: today and 7 years prior
            end_date = datetime.today().strftime('%Y-%m-%d')
        
        if start_date is None:
            start_date = (datetime.today() - timedelta(days=365*7)).strftime('%Y-%m-%d')
            
        print(f"Getting data from {start_date} to {end_date}")
        
        nasdaq_data = yf.download(ticker, start=start_date, end=end_date)
        vix_data = yf.download(vix_ticker, start=start_date, end=end_date)

        treasury_data = yf.download(treasury_ticker, start=start_date, end=end_date)
        nasdaq_futures_data = yf.download(nasdaq_futures_ticker, start=start_date, end=end_date)
        usd_index_data = yf.download(usd_index_ticker, start=start_date, end=end_date)

        vix_data = vix_data.rename(columns={
            'Open': 'VIX_Open', 
            'High': 'VIX_High', 
            'Low': 'VIX_Low', 
            'Close': 'VIX_Close'
        })

        treasury_data = treasury_data.rename(columns={
            'Open': 'TNX_Open', 
            'High': 'TNX_High', 
            'Low': 'TNX_Low', 
            'Close': 'TNX_Close',
            'Volume': 'TNX_Volume'
        })

        nasdaq_futures_data = nasdaq_futures_data.rename(columns={
            'Open': 'NQF_Open', 
            'High': 'NQF_High', 
            'Low': 'NQF_Low', 
            'Close': 'NQF_Close',
            'Volume': 'NQF_Volume'
        })

        usd_index_data = usd_index_data.rename(columns={
            'Open': 'USD_Open', 
            'High': 'USD_High', 
            'Low': 'USD_Low', 
            'Close': 'USD_Close',
            'Volume': 'USD_Volume'
        })

        nasdaq_data['Volume_Imbalance'] = (
            nasdaq_data['Volume'] - nasdaq_data['Volume'].rolling(window=5).mean()
        ) / nasdaq_data['Volume'].rolling(window=5).std()

        apple_ticker = yf.Ticker('AAPL')

        try:
            financials = apple_ticker.info

            price_to_earnings = financials.get('trailingPE', np.nan)
            price_to_book = financials.get('priceToBook', np.nan)

            nasdaq_data['PE_Ratio'] = price_to_earnings
            nasdaq_data['Price_to_Book'] = price_to_book
        except Exception as e:
            print(f"Error getting financial ratios: {e}")
            nasdaq_data['PE_Ratio'] = np.nan
            nasdaq_data['Price_to_Book'] = np.nan

        nasdaq_data['Options_Implied_Vol'] = vix_data['VIX_Close']

        common_dates = nasdaq_data.index.intersection(nasdaq_futures_data.index)
        nasdaq_data.loc[common_dates, 'Futures_Premium'] = (
            nasdaq_futures_data.loc[common_dates, 'NQF_Close'] - 
            nasdaq_data.loc[common_dates, 'Close']
        ) / nasdaq_data.loc[common_dates, 'Close'] * 100

        treasury_data['TNX_Daily_Change'] = treasury_data['TNX_Close'].pct_change() * 100

        usd_index_data['USD_Daily_Change'] = usd_index_data['USD_Close'].pct_change() * 100

        merged_data = nasdaq_data
        for df in [vix_data, treasury_data, nasdaq_futures_data, usd_index_data]:
            merged_data = pd.merge(merged_data, df, how='left', left_index=True, right_index=True)
        
        return merged_data
    except Exception as e:
        print(f"Error fetching additional market data: {e}")
        return None
    
if __name__ == "__main__":
    print("module is not meant to be run directly")
    print("import and use functions in main script")