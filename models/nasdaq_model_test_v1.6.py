import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta

def fetch_additional_market_data(ticker='^IXIC', vix_ticker='^VIX', 
                                treasury_ticker='^TNX', nasdaq_futures_ticker='NQ=F',
                                usd_index_ticker='DX-Y.NYB',
                                start_date=None, end_date=None):
    """
    Fetch additional market data including high-frequency, financial ratios, 
    options market data, 10-year treasury yields, Nasdaq-100 futures, and USD index
    """
    try:
        # Set default date range: today and 7 years prior
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
        
        if start_date is None:
            # Default to ~7 years of historical data
            start_date = (datetime.today() - timedelta(days=365*7)).strftime('%Y-%m-%d')
            
        print(f"Fetching data from {start_date} to {end_date}")
        
        # Download main datasets
        nasdaq_data = yf.download(ticker, start=start_date, end=end_date)
        vix_data = yf.download(vix_ticker, start=start_date, end=end_date)
        
        # Download new datasets
        treasury_data = yf.download(treasury_ticker, start=start_date, end=end_date)
        nasdaq_futures_data = yf.download(nasdaq_futures_ticker, start=start_date, end=end_date)
        usd_index_data = yf.download(usd_index_ticker, start=start_date, end=end_date)

        # Rename VIX columns
        vix_data = vix_data.rename(columns={
            'Open': 'VIX_Open', 
            'High': 'VIX_High', 
            'Low': 'VIX_Low', 
            'Close': 'VIX_Close'
        })
        
        # Rename Treasury data columns
        treasury_data = treasury_data.rename(columns={
            'Open': 'TNX_Open', 
            'High': 'TNX_High', 
            'Low': 'TNX_Low', 
            'Close': 'TNX_Close',
            'Volume': 'TNX_Volume'
        })
        
        # Rename Nasdaq futures columns
        nasdaq_futures_data = nasdaq_futures_data.rename(columns={
            'Open': 'NQF_Open', 
            'High': 'NQF_High', 
            'Low': 'NQF_Low', 
            'Close': 'NQF_Close',
            'Volume': 'NQF_Volume'
        })
        
        # Rename USD index columns
        usd_index_data = usd_index_data.rename(columns={
            'Open': 'USD_Open', 
            'High': 'USD_High', 
            'Low': 'USD_Low', 
            'Close': 'USD_Close',
            'Volume': 'USD_Volume'
        })

        # Calculate volume imbalance
        nasdaq_data['Volume_Imbalance'] = (
            nasdaq_data['Volume'] - nasdaq_data['Volume'].rolling(window=5).mean()
        ) / nasdaq_data['Volume'].rolling(window=5).std()
  
        # Fetch Apple financial ratios as in the original code
        apple_ticker = yf.Ticker('AAPL')

        try:
            financials = apple_ticker.info

            price_to_earnings = financials.get('trailingPE', np.nan)
            price_to_book = financials.get('priceToBook', np.nan)

            nasdaq_data['PE_Ratio'] = price_to_earnings
            nasdaq_data['Price_to_Book'] = price_to_book
        except Exception as e:
            print(f"Error fetching financial ratios: {e}")
            nasdaq_data['PE_Ratio'] = np.nan
            nasdaq_data['Price_to_Book'] = np.nan

        nasdaq_data['Options_Implied_Vol'] = vix_data['VIX_Close']
        
        # Calculate spread between Nasdaq and Nasdaq futures (premium/discount)
        # Match the dates first to handle missing data
        common_dates = nasdaq_data.index.intersection(nasdaq_futures_data.index)
        nasdaq_data.loc[common_dates, 'Futures_Premium'] = (
            nasdaq_futures_data.loc[common_dates, 'NQF_Close'] - 
            nasdaq_data.loc[common_dates, 'Close']
        ) / nasdaq_data.loc[common_dates, 'Close'] * 100  # in percentage
        
        # Calculate the rate change for 10-year Treasury
        treasury_data['TNX_Daily_Change'] = treasury_data['TNX_Close'].pct_change() * 100
        
        # Calculate USD strength change
        usd_index_data['USD_Daily_Change'] = usd_index_data['USD_Close'].pct_change() * 100

        # Merge all datasets
        merged_data = nasdaq_data
        for df in [vix_data, treasury_data, nasdaq_futures_data, usd_index_data]:
            merged_data = pd.merge(merged_data, df, how='left', left_index=True, right_index=True)
        
        return merged_data
    except Exception as e:
        print(f"Error fetching additional market data: {e}")
        return None

def create_features(df):
    """Create additional features for the model including the new data sources"""
    try:
        if df is None or df.empty:
            print("Error: Empty input DataFrame")
            return pd.DataFrame()

        data = df.copy()
        
        # Calculate required features
        base_features = {
            'Daily_Return': lambda x: x['Close'].pct_change(),
            'Volatility': lambda x: x['High'] - x['Low'],
            'RSI': lambda x: calculate_rsi(x['Close']),
            'SMA_10': lambda x: x['Close'].rolling(window=10, min_periods=1).mean(),
            'SMA_50': lambda x: x['Close'].rolling(window=50, min_periods=1).mean()
        }
        
        # Calculate base features
        for name, func in base_features.items():
            try:
                data[name] = func(data)
            except Exception as e:
                print(f"Error calculating {name}: {e}")
                return pd.DataFrame()

        # Fill missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Verify data quality
        essential_cols = ['Close', 'Daily_Return', 'Volatility', 'RSI']
        if data[essential_cols].isnull().any().any():
            print("Error: Missing values in essential columns")
            return pd.DataFrame()
            
        print(f"Successfully created features. Shape: {data.shape}")
        return data
        
    except Exception as e:
        print(f"Error in create_features: {e}")
        return pd.DataFrame()

def calculate_rsi(price_series, window=14):
    delta = price_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-10)  # Add small epsilon to prevent division by zero
    return 100 - (100 / (1 + rs))

def prepare_model_data(df):
    """Prepare features and target variable for modeling"""
    try:
        # Validate input data
        if df is None or len(df.index) == 0:
            print("Error: Input DataFrame is empty")
            return None, None
            
        # Verify essential columns exist
        essential_cols = ['Close', 'Daily_Return', 'Volatility', 'RSI']
        missing_cols = [col for col in essential_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing essential columns: {missing_cols}")
            return None, None
            
        # Select features (exclude target variable and completely null columns)
        feature_cols = [col for col in df.columns 
                       if col != 'Close' and not df[col].isnull().all()]
        
        # Validate feature count
        if len(feature_cols) < 5:
            print("Error: Insufficient features available")
            return None, None
            
        # Create copies to avoid SettingWithCopyWarning
        X = df[feature_cols].copy()
        y = df['Close'].copy()
        
        # Check for NaN values explicitly
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

def build_ann_model(input_shape):
    """
    Build an enhanced Artificial Neural Network model with additional capacity
    for handling more features
    """
    model = Sequential([
        # Increased capacity for more features
        Dense(128, activation='relu', input_shape=(input_shape,), 
              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.2),

        Dense(256, activation='relu', 
              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.3),
        
        Dense(128, activation='relu', 
              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.2),
        
        Dense(64, activation='relu', 
              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.1),

        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae', 'mse']
    )
    
    return model

def train_model(X, y, test_size=0.2, random_state=42):
    """
    Train and evaluate the Artificial Neural Network model with proper scaling
    of both features and target variables
    """
    if X is None or y is None:
        raise ValueError("Invalid input data: X or y is None")
        
    if X.empty or y.empty:
        raise ValueError("Empty input data provided")
    
    if len(X) < 100:
        raise ValueError(f"Insufficient data: {len(X)} samples. Need at least 100.")
    
    # Calculate minimum test size
    min_test_samples = max(20, int(len(X) * 0.1))  # At least 20 samples or 10%
    actual_test_size = max(min(test_size, 0.5), min_test_samples / len(X))
    print(f"Using test_size: {actual_test_size}")

    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=actual_test_size, shuffle=False
    )

    # Scale features
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    # Scale target variable - FIX #1: Also scale the target variable
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

    model = build_ann_model(X_train_scaled.shape[1])

    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=15,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train_scaled, y_train_scaled,  # FIX #2: Use scaled y values for training
        validation_split=0.2,
        epochs=150,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # Make predictions on scaled test data
    y_pred_scaled = model.predict(X_test_scaled).flatten()
    
    # FIX #3: Inverse transform the predictions to get actual price values
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(model.layers[0].get_weights()[0]).mean(axis=1)
    }).sort_values('importance', ascending=False)
    
    return model, feature_scaler, target_scaler, {
        'mse': mse,
        'r2': r2,
        'feature_importance': feature_importance,
        'training_history': history.history
    }

def visualize_feature_importance(feature_importance):
    """Create a bar plot of feature importances"""
    plt.figure(figsize=(14, 8))
    feature_importance.head(15).plot(x='feature', y='importance', kind='bar')
    plt.title('Top 15 Feature Importances in Neural Network')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance_ann.png')
    plt.close()

def visualize_training_history(history):
    """Visualize the training history of the neural network"""
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['mae'], label='Training MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.title('Model Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_ann.png')
    plt.close()

def visualize_cross_asset_relationships(df):
    """Visualize relationships between different asset classes"""
    try:
        # Validate required columns
        required_cols = ['Close', 'TNX_Close', 'USD_Close', 'NQF_Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns for visualization: {missing_cols}")
            return
            
        # Create figure only if we have valid data
        plt.figure(figsize=(18, 12))
        
        # Plot each subplot with error handling
        try:
            plt.subplot(2, 2, 1)
            if 'TNX_Close' in df.columns:
                valid_mask = df[['TNX_Close', 'Close']].notna().all(axis=1)
                plt.scatter(df.loc[valid_mask, 'TNX_Close'], 
                          df.loc[valid_mask, 'Close'], alpha=0.5)
                plt.title('NASDAQ vs. 10-Year Treasury Yield')
                plt.xlabel('10-Year Treasury Yield')
                plt.ylabel('NASDAQ Close')
        except Exception as e:
            print(f"Error plotting Treasury relationship: {e}")

        try:
            plt.subplot(2, 2, 2)
            if 'USD_Close' in df.columns:
                valid_mask = df[['USD_Close', 'Close']].notna().all(axis=1)
                plt.scatter(df.loc[valid_mask, 'USD_Close'], 
                          df.loc[valid_mask, 'Close'], alpha=0.5)
                plt.title('NASDAQ vs. USD Index')
                plt.xlabel('USD Index')
                plt.ylabel('NASDAQ Close')
        except Exception as e:
            print(f"Error plotting USD relationship: {e}")

        try:
            plt.subplot(2, 2, 3)
            if 'NQF_Close' in df.columns:
                valid_mask = df[['NQF_Close', 'Close']].notna().all(axis=1)
                plt.scatter(df.loc[valid_mask, 'NQF_Close'], 
                          df.loc[valid_mask, 'Close'], alpha=0.5)
                plt.title('NASDAQ vs. NASDAQ Futures')
                plt.xlabel('NASDAQ Futures')
                plt.ylabel('NASDAQ Close')
        except Exception as e:
            print(f"Error plotting NASDAQ Futures relationship: {e}")

        try:
            plt.subplot(2, 2, 4)
            if 'USD_Close' in df.columns and 'TNX_Close' in df.columns:
                valid_mask = df[['USD_Close', 'TNX_Close']].notna().all(axis=1)
                plt.scatter(df.loc[valid_mask, 'USD_Close'], 
                          df.loc[valid_mask, 'TNX_Close'], alpha=0.5)
                plt.title('Treasury Yield vs. USD Index')
                plt.xlabel('USD Index')
                plt.ylabel('10-Year Treasury Yield')
        except Exception as e:
            print(f"Error plotting Treasury Yield vs. USD Index relationship: {e}")
        
        plt.tight_layout()
        plt.savefig('cross_asset_relationships.png')
        plt.close()
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        plt.close()  # Ensure figure is closed on error

def main():
    try:
        # Call fetch_additional_market_data with the new data sources
        print("Fetching market data including Treasury yields, Nasdaq futures, and USD index...")
        merged_data = fetch_additional_market_data()
        
        if merged_data is None:
            print("Failed to fetch market data. Exiting.")
            return

        print("Creating features from the data...")
        featured_data = create_features(merged_data)

        # Skip visualization as requested by user
        # print("Visualizing cross-asset relationships...")
        # visualize_cross_asset_relationships(featured_data)
        
        print("Preparing model data...")
        X, y = prepare_model_data(featured_data)
        if X is None or y is None:
            print("Failed to prepare model data. Exiting.")
            return
            
        print(f"Training model with {X.shape[1]} features...")
        model, feature_scaler, target_scaler, metrics = train_model(X, y)

        print("\nModel Performance:")
        print(f"Mean Squared Error: {metrics['mse']}")
        print(f"R-squared Score: {metrics['r2']}")

        # Skip visualizations as requested by user
        # visualize_feature_importance(metrics['feature_importance'])
        # visualize_training_history(metrics['training_history'])

        # Save the model with a timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d")
        model_filename = f'nasdaq_prediction_ann_model_enhanced_{timestamp}.h5'
        feature_scaler_filename = f'nasdaq_prediction_feature_scaler_{timestamp}.joblib'
        target_scaler_filename = f'nasdaq_prediction_target_scaler_{timestamp}.joblib'
        
        model.save(model_filename)
        joblib.dump(feature_scaler, feature_scaler_filename)
        joblib.dump(target_scaler, target_scaler_filename)
        
        print(f"\nModel saved as {model_filename}")
        print(f"Feature scaler saved as {feature_scaler_filename}")
        print(f"Target scaler saved as {target_scaler_filename}")

        print("\nTop 15 Most Important Features:")
        print(metrics['feature_importance'].head(15))

        # Make prediction for the next day with proper scaling
        last_features = X.iloc[-1].values.reshape(1, -1)
        last_features_scaled = feature_scaler.transform(last_features)
        next_day_prediction_scaled = model.predict(last_features_scaled)[0][0]
        
        # FIX #4: Inverse transform the prediction to get actual price
        next_day_prediction = target_scaler.inverse_transform([[next_day_prediction_scaled]])[0][0]
        
        # Get actual numeric values from Series
        last_close_value = float(y.iloc[-1])
        
        # FIX #5: Add reasonableness check - cap extreme predictions
        if next_day_prediction > last_close_value * 1.1:  # More than 10% increase
            print("Warning: Prediction seems unreasonably high")
            next_day_prediction = last_close_value * 1.01  # Cap at 1% increase
        elif next_day_prediction < last_close_value * 0.9:  # More than 10% decrease
            print("Warning: Prediction seems unreasonably low")
            next_day_prediction = last_close_value * 0.99  # Cap at 1% decrease
        
        predicted_change = ((next_day_prediction - last_close_value) / last_close_value) * 100
        
        print("\nNext Day Prediction:")
        print(f"Last Close: {last_close_value:.2f}")
        print(f"Predicted Close: {next_day_prediction:.2f}")
        print(f"Predicted Change: {predicted_change:.2f}%")
        
        # FIX #6: Add sanity check comparison to current Nasdaq level
        current_nasdaq_level = 16300  # Approximate level as of April 2025
        print(f"\nSanity Check:")
        print(f"Current Nasdaq Level (Apr 2025): ~{current_nasdaq_level}")
        percent_diff = ((next_day_prediction - current_nasdaq_level) / current_nasdaq_level) * 100
        print(f"Prediction differs from current by: {percent_diff:.2f}%")
        if abs(percent_diff) > 10:
            print("WARNING: Prediction deviates significantly from current market levels")
        
        # Print a summary of the new data sources contribution (simplified)
        print("\nContribution of New Data Sources:")
        for source_prefix, name in [('TNX', '10-Year Treasury'), ('NQF', 'Nasdaq-100 Futures'), ('USD', 'USD Index')]:
            source_features = [i for i in metrics['feature_importance']['feature'] if source_prefix in str(i)]
            if source_features:
                source_importance = metrics['feature_importance'][
                    metrics['feature_importance']['feature'].isin(source_features)
                ]['importance'].sum()
                print(f"{name} Features: {source_importance:.4f}")
    
    except Exception as e:
        print(f"An error occurred in the main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()