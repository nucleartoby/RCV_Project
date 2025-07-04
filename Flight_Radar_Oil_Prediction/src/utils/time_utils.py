from datetime import datetime, timedelta
import pandas as pd
from typing import List, Union

def get_trading_hours_filter() -> pd.Series:
    now = datetime.now()
    hour = now.hour

    return (hour >= 6) & (hour <= 20)  # 6 AM to 8 PM local time

def align_time_series(df1: pd.DataFrame, df2: pd.DataFrame, 
                     time_col1: str = 'timestamp', 
                     time_col2: str = 'timestamp',
                     method: str = 'nearest') -> pd.DataFrame:
    df1_copy = df1.copy()
    df2_copy = df2.copy()

    df1_copy.set_index(time_col1, inplace=True)
    df2_copy.set_index(time_col2, inplace=True)
    
    if method == 'nearest':
        return pd.merge_asof(df1_copy.sort_index(), df2_copy.sort_index(), 
                           left_index=True, right_index=True, 
                           direction='nearest')
    elif method == 'inner':
        return pd.merge(df1_copy, df2_copy, left_index=True, right_index=True, how='inner')
    else:
        raise ValueError("Method must be 'nearest' or 'inner'")

def create_time_windows(df: pd.DataFrame, time_col: str, 
                       windows: List[str]) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy[time_col] = pd.to_datetime(df_copy[time_col])
    df_copy.set_index(time_col, inplace=True)
    
    result = pd.DataFrame(index=df_copy.index)
    
    for window in windows:
        rolled = df_copy.rolling(window)
        result[f'mean_{window}'] = rolled.mean()
        result[f'std_{window}'] = rolled.std()
        result[f'count_{window}'] = rolled.count()
    
    return result