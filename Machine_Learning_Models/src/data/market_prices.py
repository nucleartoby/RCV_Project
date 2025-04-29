import yfinance as yf
from datetime import datetime, timedelta

def get_current_nasdaq_level():

    try:
        nasdaq = yf.Ticker("^IXIC")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        data = nasdaq.history(start=start_date, end=end_date)
        
        if data.empty:
            print("Warning: Could not fetch NASDAQ data. Using fallback value.")
            return 15900

        latest_close = data['Close'].iloc[-1]
        latest_date = data.index[-1].strftime('%Y-%m-%d')
        
        print(f"Current NASDAQ Level (as of {latest_date}): {latest_close:.2f}")
        return latest_close
    
    except Exception as e:
        print(f"Error fetching NASDAQ data: {e}")
        print("Using fallback value for NASDAQ level.")
        return 15900

def get_market_data(symbol, days=7):

    try:
        ticker = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            print(f"Warning: Could not fetch data for {symbol}")
            return None
            
        return data
    
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None