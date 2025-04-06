import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime, timedelta
import traceback

def fetch_additional_market_data(ticker='^IXIC', vix_ticker='^VIX',
                                 start_date=None, end_date=None):
    """
    Fetch additional market data including high-frequency, financial ratios, 
    and options market data
    
    Args:
        ticker (str): Stock market index ticker
        vix_ticker (str): Volatility index ticker
        start_date (str): Start date for data collection (default: 7 years ago)
        end_date (str): End date for data collection (default: today)
    
    Returns:
        pd.DataFrame: Processed market data with additional features
    """
    try:
        # Set default date range: today and 7 years prior
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
        
        if start_date is None:
            # FIX: Reduced to 5 years to avoid very old, potentially irrelevant data
            start_date = (datetime.today() - timedelta(days=365*5)).strftime('%Y-%m-%d')
            
        print(f"Fetching data from {start_date} to {end_date}")
        
        # Fetch original data
        nasdaq_data = yf.download(ticker, start=start_date, end=end_date)
        vix_data = yf.download(vix_ticker, start=start_date, end=end_date)

        # ENHANCEMENT 1: Add Treasury Yields (10-Year)
        try:
            treasury_data = yf.download("^TNX", start=start_date, end=end_date)
            treasury_data = treasury_data.rename(columns={
                'Close': 'Treasury_10Y'
            })
            # Keep only the Treasury yield column we need
            treasury_data = treasury_data[['Treasury_10Y']]
        except Exception as e:
            print(f"Error fetching Treasury data: {e}")
            treasury_data = pd.DataFrame(index=nasdaq_data.index)
            treasury_data['Treasury_10Y'] = np.nan

        # ENHANCEMENT 2: Add Tech Sector performance (using XLK as proxy)
        try:
            tech_sector = yf.download("XLK", start=start_date, end=end_date)
            tech_sector = tech_sector.rename(columns={
                'Close': 'Tech_Sector'
            })
            # Keep only the Tech sector close price
            tech_sector = tech_sector[['Tech_Sector']]
        except Exception as e:
            print(f"Error fetching Tech sector data: {e}")
            tech_sector = pd.DataFrame(index=nasdaq_data.index)
            tech_sector['Tech_Sector'] = np.nan

        # ENHANCEMENT 3: Add USD strength (USD Index)
        try:
            usd_index = yf.download("DX-Y.NYB", start=start_date, end=end_date)
            usd_index = usd_index.rename(columns={
                'Close': 'USD_Index'
            })
            # Keep only the USD index close price
            usd_index = usd_index[['USD_Index']]
        except Exception as e:
            print(f"Error fetching USD Index data: {e}")
            usd_index = pd.DataFrame(index=nasdaq_data.index)
            usd_index['USD_Index'] = np.nan

        vix_data = vix_data.rename(columns={
            'Open': 'VIX_Open', 
            'High': 'VIX_High', 
            'Low': 'VIX_Low', 
            'Close': 'VIX_Close'
        })

        # FIX: Use more robust method for volume imbalance calculation to handle outliers
        nasdaq_data['Volume_Imbalance'] = (
            (nasdaq_data['Volume'] - nasdaq_data['Volume'].rolling(window=5).median()) / 
            (nasdaq_data['Volume'].rolling(window=5).std() + 1e-6)  # Add epsilon to prevent division by zero
        ).clip(-5, 5)  # Clip extreme values
  
        # Add VIX as options implied volatility
        nasdaq_data['Options_Implied_Vol'] = vix_data['VIX_Close']

        # Merge all data sources
        merged_data = pd.concat([
            nasdaq_data, 
            vix_data, 
            treasury_data, 
            tech_sector, 
            usd_index
        ], axis=1)
        
        # FIX: Check for and handle outliers in the main price data
        # Calculate z-scores for Close prices
        z_scores = np.abs((merged_data['Close'] - merged_data['Close'].rolling(window=20).mean()) / 
                         merged_data['Close'].rolling(window=20).std())
        
        # Identify potential outliers
        outliers = z_scores > 3
        if outliers.sum() > 0:
            print(f"Found {outliers.sum()} potential outliers in Close prices")
            # For visualization/debugging only, not modifying prices
            merged_data['outlier'] = outliers
        
        return merged_data
    except Exception as e:
        print(f"Error fetching additional market data: {e}")
        return None

def create_features(df):
    """Create additional features for the model"""
    print(f"Initial shape before feature creation: {df.shape}")
    data = df.copy()
    
    # Debug merged data
    print("\nColumns in input data:")
    print(data.columns.tolist())
    
    # Ensure index is properly set
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Basic features - validate and fill
    base_columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    for col in base_columns:
        if col not in data.columns:
            print(f"Warning: Missing {col} column")
            data[col] = np.nan
    
    # Create features safely
    try:
        # Technical indicators
        data['Daily_Return'] = data['Close'].pct_change().clip(-0.2, 0.2)  # FIX: Clip extreme returns
        data['Volatility'] = data['High'] - data['Low']
        
        # Normalize volatility by price level to prevent scale issues
        data['Normalized_Volatility'] = (data['Volatility'] / data['Close']).clip(0, 0.1)  # FIX: Clip extreme values
        
        # Moving averages
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # Add normalized price levels relative to moving averages
        # FIX: Ensure we don't divide by zero and clip extreme values
        data['Price_to_SMA10'] = (data['Close'] / (data['SMA_10'] + 1e-6)).clip(0.7, 1.3)
        data['Price_to_SMA50'] = (data['Close'] / (data['SMA_50'] + 1e-6)).clip(0.7, 1.3)
        
        # Calculate RSI
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            # FIX: Prevent division by zero
            rs = gain / (loss + 1e-6)
            return 100 - (100 / (1 + rs))
        
        data['RSI'] = calculate_rsi(data['Close'])
        
        # Handle VIX features if available
        vix_cols = ['VIX_High', 'VIX_Low', 'VIX_Close']
        if all(col in data.columns for col in vix_cols):
            data['VIX_Volatility'] = (data['VIX_High'] - data['VIX_Low']).clip(0, 10)  # FIX: Clip extreme values
        
        # Create lags for available columns - FIX: Use fewer lags to reduce noise
        lags = [1, 2, 3]  # Removed lag 5
        for col in ['Close', 'VIX_Close', 'Treasury_10Y', 'Tech_Sector', 'USD_Index']:
            if col in data.columns:
                for lag in lags:
                    data[f'{col}_Lag_{lag}'] = data[col].shift(lag)
                    
                    # Add percentage changes for lagged values
                    if lag > 1:
                        pct_change = data[col].pct_change(periods=lag)
                        # FIX: Clip extreme percentage changes
                        data[f'{col}_Pct_Change_{lag}'] = pct_change.clip(-0.2, 0.2)
        
        # Calculate the rate of change for Treasury yields
        if 'Treasury_10Y' in data.columns:
            # FIX: Clip extreme changes in treasury yields
            data['Treasury_10Y_Rate_of_Change'] = data['Treasury_10Y'].pct_change().clip(-0.2, 0.2)
            
        # Calculate relative performance metrics
        if 'Tech_Sector' in data.columns and 'Close' in data.columns:
            # Tech sector performance relative to NASDAQ
            # FIX: Normalize and clip to prevent extreme values
            data['Tech_vs_NASDAQ'] = (data['Tech_Sector'] / (data['Close'] + 1e-6)).clip(0.5, 2)
            # Tech sector momentum - clipped
            data['Tech_Momentum'] = data['Tech_Sector'].pct_change(3).clip(-0.2, 0.2)
            
        # Calculate USD momentum
        if 'USD_Index' in data.columns:
            data['USD_Momentum'] = data['USD_Index'].pct_change(3).clip(-0.1, 0.1)
            
            # Calculate correlation between USD and NASDAQ using a rolling window
            # FIX: No action needed here as correlation is naturally bounded [-1, 1]
            data['USD_NASDAQ_Corr'] = data['USD_Index'].rolling(window=30).corr(data['Close'])
        
        # Fill NaN values by group
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        # FIX: Add recent price range feature to help model stay within reasonable bounds
        data['Price_52w_High'] = data['Close'].rolling(window=252).max()
        data['Price_52w_Low'] = data['Close'].rolling(window=252).min()
        data['Price_Range_Ratio'] = ((data['Close'] - data['Price_52w_Low']) / 
                                   (data['Price_52w_High'] - data['Price_52w_Low'] + 1e-6)).clip(0, 1)
        
        print("\nFinal shape:", data.shape)
        print("NaN count after processing:")
        print(data.isna().sum())
        
        return data
        
    except Exception as e:
        print(f"Error in feature creation: {e}")
        traceback.print_exc()
        return pd.DataFrame()  # Return empty frame on error

def prepare_model_data(df):
    """
    Prepare features and target variable for modeling
    
    Args:
        df (pd.DataFrame): Input dataframe with features
    
    Returns:
        tuple: X (features), y (target), target_scaler (for inverse transform)
    """
    # First create features list with available columns
    features = [
        'Open', 'High', 'Low', 
        'VIX_Open', 'VIX_High', 'VIX_Low', 'VIX_Close',
        'Daily_Return', 'RSI', 'VIX_Volatility',
        'SMA_10', 'SMA_50', 'Normalized_Volatility',
        'Price_to_SMA10', 'Price_to_SMA50',
        'Price_Range_Ratio'  # FIX: Added new feature
    ]

    # Add lag features
    lags = [1, 2, 3]  # FIX: Removed lag 5
    for lag in lags:
        lag_features = [
            f'Close_Lag_{lag}', 
            f'VIX_Close_Lag_{lag}'
        ]
        
        # Add percentage change features
        if lag > 1:
            pct_change_features = [
                f'Close_Pct_Change_{lag}',
                f'VIX_Close_Pct_Change_{lag}'
            ]
            lag_features.extend(pct_change_features)
            
        features.extend(lag_features)
        
        # Add lagged features for new data sources
        if f'Treasury_10Y_Lag_{lag}' in df.columns:
            features.append(f'Treasury_10Y_Lag_{lag}')
            if lag > 1 and f'Treasury_10Y_Pct_Change_{lag}' in df.columns:
                features.append(f'Treasury_10Y_Pct_Change_{lag}')
        
        if f'Tech_Sector_Lag_{lag}' in df.columns:
            features.append(f'Tech_Sector_Lag_{lag}')
            if lag > 1 and f'Tech_Sector_Pct_Change_{lag}' in df.columns:
                features.append(f'Tech_Sector_Pct_Change_{lag}')
            
        if f'USD_Index_Lag_{lag}' in df.columns:
            features.append(f'USD_Index_Lag_{lag}')
            if lag > 1 and f'USD_Index_Pct_Change_{lag}' in df.columns:
                features.append(f'USD_Index_Pct_Change_{lag}')

    # Other features
    new_features = ['Volume_Imbalance', 'Options_Implied_Vol']
    features.extend(new_features)
    
    # High-impact features
    high_impact_features = [
        'Treasury_10Y', 'Treasury_10Y_Rate_of_Change',
        'Tech_vs_NASDAQ', 'Tech_Momentum',
        'USD_Momentum', 'USD_NASDAQ_Corr',
        'Price_52w_High', 'Price_52w_Low'  # FIX: Added price range features
    ]
    
    # Only add features that actually exist in the dataframe
    additional_features = [f for f in high_impact_features if f in df.columns]
    features.extend(additional_features)

    # Check for missing features and filter out any that don't exist
    available_features = [f for f in features if f in df.columns]
    missing_features = set(features) - set(available_features)
    if missing_features:
        print(f"Warning: Some features are not available and will be skipped: {missing_features}")
    
    X = df[available_features]
    y = df['Close']
    
    # FIX: Get more robust statistics about our target variable for validation later
    last_original_close = y.iloc[-1]
    avg_price = y.mean()
    min_price = y.min()
    max_price = y.max()
    std_price = y.std()
    
    price_stats = {
        'last_close': last_original_close,
        'avg_price': avg_price,
        'min_price': min_price,
        'max_price': max_price,
        'std_price': std_price
    }
    
    return X, y, price_stats

def build_ann_model(input_shape):
    """
    Build an Artificial Neural Network model with adjustments to prevent overfitting
    
    Args:
        input_shape (int): Number of input features
    
    Returns:
        tf.keras.Model: Compiled neural network model
    """
    # FIX: Simplify model architecture to prevent overfitting
    model = Sequential([
        # Start with fewer neurons
        Dense(16, activation='relu', input_shape=(input_shape,), 
              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.2),

        Dense(32, activation='relu', 
              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.3),

        # Output layer
        Dense(1)
    ])

    # FIX: Use an even lower learning rate for stability
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='mean_squared_error',
        metrics=['mae', 'mse']
    )
    
    return model

def train_model(X, y, price_stats, test_size=0.2, random_state=42):
    """
    Train and evaluate the Artificial Neural Network model with data validation
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable (Close price)
        price_stats (dict): Statistics about the price data for validation
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: model, features_scaler, target_scaler, metrics
    """
    # Validate input data
    if X.empty or y.empty:
        raise ValueError("Empty input data provided")
    
    print(f"Input data shape - X: {X.shape}, y: {y.shape}")
    
    # Check if we have enough samples
    min_samples = 100  # Minimum number of samples needed
    if len(X) < min_samples:
        raise ValueError(f"Insufficient data: {len(X)} samples. Need at least {min_samples}.")
    
    # Adjust test_size if necessary
    actual_test_size = max(min(test_size, 0.5), len(X) // 10 / len(X))
    print(f"Using test_size: {actual_test_size}")
    
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    try:
        # FIX: Train-test split ensuring chronological order but use a validation set too
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)
        test_size = len(X) - train_size - val_size
        
        # Split data chronologically 
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        
        X_val = X.iloc[train_size:train_size+val_size]
        y_val = y.iloc[train_size:train_size+val_size]
        
        X_test = X.iloc[-test_size:]
        y_test = y.iloc[-test_size:]
        
        print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
        
        # FIX: Use RobustScaler instead of StandardScaler to be less sensitive to outliers
        features_scaler = RobustScaler()
        X_train_scaled = features_scaler.fit_transform(X_train)
        X_val_scaled = features_scaler.transform(X_val)
        X_test_scaled = features_scaler.transform(X_test)
        
        # FIX: Use RobustScaler for target instead of MinMaxScaler
        # This helps reduce the impact of outliers
        target_scaler = RobustScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
        y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1))
        y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

        model = build_ann_model(X_train_scaled.shape[1])

        # FIX: Add ModelCheckpoint to save best model
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10,
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            'best_nasdaq_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
        
        # Use more epochs but rely on early stopping
        # FIX: Pass validation data explicitly instead of using validation_split
        history = model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )

        # Make predictions on the test set (still in scaled form)
        y_pred_scaled = model.predict(X_test_scaled)
        
        # Transform predictions back to original scale
        y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()

        # Calculate metrics on original scale
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Calculate percentage error
        pct_error = np.abs((y_test.values - y_pred) / y_test.values).mean() * 100
        
        # FIX: Make prediction for next trading day with additional validation
        last_features = X.iloc[-1].values.reshape(1, -1)
        last_features_scaled = features_scaler.transform(last_features)
        next_day_prediction_scaled = model.predict(last_features_scaled)
        next_day_prediction_raw = target_scaler.inverse_transform(next_day_prediction_scaled)[0][0]
        
        # FIX: Add sanity checks on the prediction
        last_close = price_stats['last_close']
        avg_price = price_stats['avg_price']
        std_price = price_stats['std_price']
        max_price = price_stats['max_price']
        
        # Calculate a reasonable upper bound for the prediction
        # No more than 10% above last close or 3 standard deviations above mean
        reasonable_max = min(last_close * 1.1, avg_price + 3 * std_price)
        reasonable_max = min(reasonable_max, max_price * 1.05)  # And no more than 5% above historical max
        
        # Apply sanity bounds
        if next_day_prediction_raw > reasonable_max:
            print(f"WARNING: Raw prediction ({next_day_prediction_raw:.2f}) exceeds reasonable max ({reasonable_max:.2f})")
            print(f"Applying sanity bounds to prediction")
            next_day_prediction = reasonable_max
        else:
            next_day_prediction = next_day_prediction_raw
        
        print(f"Last known Close: {last_close}")
        print(f"Raw next day prediction: {next_day_prediction_raw}")
        print(f"Bounded next day prediction: {next_day_prediction}")
        print(f"Percentage difference: {abs(next_day_prediction - last_close) / last_close * 100:.2f}%")
        
        # FIX: Analyze feature importance more carefully
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(model.layers[0].get_weights()[0]).mean(axis=1)
        }).sort_values('importance', ascending=False)
        
        # FIX: Calculate prediction accuracy on test set
        correct_direction = ((y_test.diff() > 0) == (np.diff(np.append([y_test.iloc[0]], y_pred)) > 0)).mean()
        print(f"Direction accuracy on test set: {correct_direction:.2f}")
        
        return model, features_scaler, target_scaler, {
            'mse': mse,
            'r2': r2,
            'pct_error': pct_error,
            'feature_importance': feature_importance,
            'training_history': history.history,
            'last_prediction': next_day_prediction,
            'direction_accuracy': correct_direction
        }
    except Exception as e:
        print(f"Error during model training: {e}")
        traceback.print_exc()
        raise
    
def visualize_feature_importance(feature_importance):
    """
    Create a bar plot of feature importances
    
    Args:
        feature_importance (pd.DataFrame): Dataframe with feature importances
    """
    plt.figure(figsize=(12, 6))
    feature_importance.head(10).plot(x='feature', y='importance', kind='bar')
    plt.title('Top 10 Feature Importances in Neural Network')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance_ann.png')
    plt.close()

def visualize_training_history(history):
    """
    Visualize the training history of the neural network
    
    Args:
        history (dict): Training history dictionary
    """
    plt.figure(figsize=(12, 6))
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

def visualize_predictions(y_test, y_pred):
    """
    Visualize model predictions against actual values
    
    Args:
        y_test (pd.Series): Actual values
        y_pred (np.array): Predicted values
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual')
    plt.plot(y_test.index, y_pred, label='Predicted')
    plt.title('NASDAQ Close Price: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig('prediction_comparison.png')
    plt.close()

def main():
    try:
        merged_data = fetch_additional_market_data()
        if merged_data is None or merged_data.empty:
            print("Failed to fetch market data or received empty data. Exiting.")
            return

        print(f"\nInitial merged data shape: {merged_data.shape}")
        print("\nMerged data columns:")
        print(merged_data.columns.tolist())
        
        featured_data = create_features(merged_data)
        print(f"Featured data shape: {featured_data.shape}")
        
        if featured_data.empty:
            print("No data after feature creation. Exiting.")
            return
            
        # Store last closing price for validation
        last_close_price = featured_data['Close'].iloc[-1]
        print(f"Last close price: {last_close_price}")

        X, y, last_close = prepare_model_data(featured_data)
        print(f"Final data shape before training - X: {X.shape}, y: {y.shape}")
        
        if X.empty or y.empty:
            print("No data after preparation. Exiting.")
            return

        model, features_scaler, target_scaler, metrics = train_model(X, y, last_close)

        print("\nModel Performance:")
        print(f"Mean Squared Error: {metrics['mse']}")
        print(f"R-squared Score: {metrics['r2']}")
        print(f"Mean Percentage Error: {metrics['pct_error']}%")

        visualize_feature_importance(metrics['feature_importance'])
        visualize_training_history(metrics['training_history'])

        # FIX: Generate predictions for test set and visualize
        # Split again just for visualization (could refactor this)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_test_scaled = features_scaler.transform(X_test)
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()
        
        visualize_predictions(y_test, y_pred)

        # Save the model with a timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d")
        model.save(f'nasdaq_prediction_ann_model_{timestamp}.h5')
        joblib.dump(features_scaler, f'nasdaq_prediction_features_scaler_{timestamp}.joblib')
        joblib.dump(target_scaler, f'nasdaq_prediction_target_scaler_{timestamp}.joblib')

        print("\nTop 10 Most Important Features:")
        print(metrics['feature_importance'].head(10))

        # Prediction for next trading day
        print("\nNext Day Predicted Close:")
        print(metrics['last_prediction'])
        
        # Additional validation
        if abs(metrics['last_prediction'] - last_close) / last_close > 0.1:
            print("\nWARNING: Prediction differs from last close by more than 10%")
            print("This may indicate a scaling issue or model instability")
    
    except Exception as e:
        print(f"An error occurred in the main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()