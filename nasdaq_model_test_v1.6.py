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
            # Default to ~7 years of historical data
            start_date = (datetime.today() - timedelta(days=365*7)).strftime('%Y-%m-%d')
            
        print(f"Fetching data from {start_date} to {end_date}")
        
        nasdaq_data = yf.download(ticker, start=start_date, end=end_date)
        vix_data = yf.download(vix_ticker, start=start_date, end=end_date)

        vix_data = vix_data.rename(columns={
            'Open': 'VIX_Open', 
            'High': 'VIX_High', 
            'Low': 'VIX_Low', 
            'Close': 'VIX_Close'
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
            print(f"Error fetching financial ratios: {e}")
            nasdaq_data['PE_Ratio'] = np.nan
            nasdaq_data['Price_to_Book'] = np.nan

        nasdaq_data['Options_Implied_Vol'] = vix_data['VIX_Close']

        merged_data = pd.concat([nasdaq_data, vix_data], axis=1)
        
        return merged_data
    except Exception as e:
        print(f"Error fetching additional market data: {e}")
        return None

def create_features(df):
    """
    Create additional features for the model
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    data = df.copy()

    required_columns = ['Close', 'High', 'Low', 'VIX_High', 'VIX_Low', 'VIX_Close']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    data.loc[:, 'Daily_Return'] = data['Close'].pct_change()
    data.loc[:, 'Volatility'] = data['High'] - data['Low']
    data.loc[:, 'VIX_Volatility'] = data['VIX_High'] - data['VIX_Low']

    lags = [1, 2, 3, 5]
    for lag in lags:
        data.loc[:, f'Close_Lag_{lag}'] = data['Close'].shift(lag)
        data.loc[:, f'VIX_Close_Lag_{lag}'] = data['VIX_Close'].shift(lag)

    data.loc[:, 'SMA_10'] = data['Close'].rolling(window=10).mean()
    data.loc[:, 'SMA_50'] = data['Close'].rolling(window=50).mean()

    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

    rs = gain / (loss + 1e-10)
    data.loc[:, 'RSI'] = 100 - (100 / (1 + rs))

    new_features = ['Volume_Imbalance', 'PE_Ratio', 'Price_to_Book', 'Options_Implied_Vol']
    for feature in new_features:
        if feature in data.columns:
            data.loc[:, feature] = data[feature].fillna(method='ffill')
        else:
            print(f"Warning: {feature} not found in the dataframe")

    data.dropna(inplace=True)
    
    return data

def prepare_model_data(df):
    """
    Prepare features and target variable for modeling
    
    Args:
        df (pd.DataFrame): Input dataframe with features
    
    Returns:
        tuple: X (features), y (target)
    """
    features = [
        'Open', 'High', 'Low', 
        'VIX_Open', 'VIX_High', 'VIX_Low', 'VIX_Close',
        'Daily_Return', 'Volatility', 'VIX_Volatility',
        'SMA_10', 'SMA_50', 'RSI'
    ]

    lags = [1, 2, 3, 5]
    for lag in lags:
        features.extend([
            f'Close_Lag_{lag}', 
            f'VIX_Close_Lag_{lag}'
        ])

    new_features = ['Volume_Imbalance', 'PE_Ratio', 'Price_to_Book', 'Options_Implied_Vol']
    features.extend(new_features)

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    X = df[features]
    y = df['Close']
    
    return X, y

def build_ann_model(input_shape):
    """
    Build an Artificial Neural Network model
    
    Args:
        input_shape (int): Number of input features
    
    Returns:
        tf.keras.Model: Compiled neural network model
    """
    model = Sequential([

        Dense(64, activation='relu', input_shape=(input_shape,), 
              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.2),

        Dense(128, activation='relu', 
              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.3),
        
        Dense(64, activation='relu', 
              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.2),

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
    Train and evaluate the Artificial Neural Network model
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: Trained model, scaler, evaluation metrics
    """
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_ann_model(X_train_scaled.shape[1])

    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train_scaled, y_train, 
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )

    y_pred = model.predict(X_test_scaled).flatten()

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(model.layers[0].get_weights()[0]).mean(axis=1)
    }).sort_values('importance', ascending=False)
    
    return model, scaler, {
        'mse': mse,
        'r2': r2,
        'feature_importance': feature_importance,
        'training_history': history.history
    }

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

def main():
    try:
        # Call fetch_additional_market_data without hardcoded dates
        # It will use the current date automatically
        merged_data = fetch_additional_market_data()
        
        if merged_data is None:
            print("Failed to fetch market data. Exiting.")
            return

        featured_data = create_features(merged_data)

        X, y = prepare_model_data(featured_data)

        model, scaler, metrics = train_model(X, y)

        print("Model Performance:")
        print(f"Mean Squared Error: {metrics['mse']}")
        print(f"R-squared Score: {metrics['r2']}")

        visualize_feature_importance(metrics['feature_importance'])

        visualize_training_history(metrics['training_history'])

        # Save the model with a timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d")
        model.save(f'nasdaq_prediction_ann_model_{timestamp}.h5')
        joblib.dump(scaler, f'nasdaq_prediction_scaler_{timestamp}.joblib')

        print("\nTop 10 Most Important Features:")
        print(metrics['feature_importance'].head(10))

        last_features = X.iloc[-1].values.reshape(1, -1)
        last_features_scaled = scaler.transform(last_features)
        next_day_prediction = model.predict(last_features_scaled)[0][0]
        
        print("\nNext Day Predicted Close:")
        print(next_day_prediction)
    
    except Exception as e:
        print(f"An error occurred in the main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

# Version 1.6
# - Added error handling for missing columns in the DataFrame.
# - Improved feature engineering with additional features.
# - Added auto set date for current date.

#Test