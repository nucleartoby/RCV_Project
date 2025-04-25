import pandas as pd
import numpy as np

def calculate_rsi(price_series, window=14):
    delta = price_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def create_features(df):
    """Create additional features for the model including the new data sources"""
    try:
        if df is None or df.empty:
            print("Error: Empty input DataFrame")
            return pd.DataFrame()

        data = df.copy()

        base_features = {
            'Daily_Return': lambda x: x['Close'].pct_change(),
            'Volatility': lambda x: x['High'] - x['Low'],
            'RSI': lambda x: calculate_rsi(x['Close']),
            'SMA_10': lambda x: x['Close'].rolling(window=10, min_periods=1).mean(),
            'SMA_50': lambda x: x['Close'].rolling(window=50, min_periods=1).mean()
        }

        for name, func in base_features.items():
            try:
                data[name] = func(data)
            except Exception as e:
                print(f"Error calculating {name}: {e}")
                return pd.DataFrame()

        data = data.fillna(method='ffill').fillna(method='bfill')

        essential_cols = ['Close', 'Daily_Return', 'Volatility', 'RSI']
        if data[essential_cols].isnull().any().any():
            print("Error: Missing values in essential columns")
            return pd.DataFrame()
            
        print(f"Successfully created features. Shape: {data.shape}")
        return data
        
    except Exception as e:
        print(f"Error in create_features: {e}")
        return pd.DataFrame()
    
    if __name__ == "__main__":
        print("module is not meant to be run directly")
        print("import and use functions in main script")