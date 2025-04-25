import pandas as pd
import numpy as np

def prepare_model_data(df):
    """Prepare features and target variable for modeling"""
    try:
        if df is None or len(df.index) == 0:
            print("Error: Input DataFrame is empty")
            return None, None

        essential_cols = ['Close', 'Daily_Return', 'Volatility', 'RSI']
        missing_cols = [col for col in essential_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing essential columns: {missing_cols}")
            return None, None

        feature_cols = [col for col in df.columns 
                       if col != 'Close' and not df[col].isnull().all()]

        if len(feature_cols) < 5:
            print("Error: Insufficient features available")
            return None, None

        X = df[feature_cols].copy()
        y = df['Close'].copy()

        has_nulls_x = X.isnull().values.any()
        has_nulls_y = y.isnull().values.any()
        
        if has_nulls_x or has_nulls_y:
            print("Error: Dataset contains NaN values")
            return None, None
            
        print(f"Data prepared successfully. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
        
    except Exception as e:
        print(f"Error in prepare_model_data: {e}")
        return None, None
    
if __name__ == "__main__":
    print("module is not meant to be run directly")
    print("Import and use functions in main script")