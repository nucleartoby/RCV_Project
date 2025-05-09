import os
import joblib
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf

from src.data.fetcher import market_data
from src.features.feature_engineering import create_features
from src.data.processing import prepare_model_data
from src.models.training import train_model
from src.data.market_prices import get_current_nasdaq_level
from src.visualisations.graphs import (
    visualise_feature_importance,
    visualise_training_history,
    visualise_cross_asset_relationships
)

def main():
    try:
        print("Getting market data including Treasury yields, Nasdaq futures, and USD index...")
        merged_data = market_data()
        
        if merged_data is None:
            print("Failed to get market data. Exiting.")
            return

        featured_data = create_features(merged_data)

        X, y = prepare_model_data(featured_data)
        if X is None or y is None:
            print("Failed to prepare model data. Exiting.")
            return
            
        print(f"Training model with {X.shape[1]} features...")
        model, feature_scaler, target_scaler, metrics = train_model(X, y)

        print("\nModel Performance:")
        print(f"Mean Squared Error: {metrics['mse']}")
        print(f"R-squared Score: {metrics['r2']}")

        output_dir = "model_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d")
        model_filename = os.path.join(output_dir, f'nasdaq_prediction_model_{timestamp}.h5')
        feature_scaler_filename = os.path.join(output_dir, f'feature_scaler_{timestamp}.joblib')
        target_scaler_filename = os.path.join(output_dir, f'target_scaler_{timestamp}.joblib')
        
        model.save(model_filename)
        joblib.dump(feature_scaler, feature_scaler_filename)
        joblib.dump(target_scaler, target_scaler_filename)
        
        print(f"\nModel saved as {model_filename}")
        print(f"Feature scaler saved as {feature_scaler_filename}")
        print(f"Target scaler saved as {target_scaler_filename}")

        print("\nMost Important Features:")
        print(metrics['feature_importance'].head(15))

        last_features = X.iloc[-1].values.reshape(1, -1)
        last_features_scaled = feature_scaler.transform(last_features)
        next_day_prediction_scaled = model.predict(last_features_scaled)[0][0]

        next_day_prediction = target_scaler.inverse_transform([[next_day_prediction_scaled]])[0][0]

        last_close_value = float(y.iloc[-1])

        if next_day_prediction > last_close_value * 1.1:
            print("Warning, prediction seems unreasonably high")
            next_day_prediction = last_close_value * 1.01
        elif next_day_prediction < last_close_value * 0.9:
            print("Warning, prediction seems unreasonably low")
            next_day_prediction = last_close_value * 0.99
        
        predicted_change = ((next_day_prediction - last_close_value) / last_close_value) * 100
        
        print("\nNext Day Prediction:")
        print(f"Last Close: {last_close_value:.2f}")
        print(f"Predicted Close: {next_day_prediction:.2f}")
        print(f"Predicted Change: {predicted_change:.2f}%")

        current_nasdaq_level = get_current_nasdaq_level()
        print(f"\nSanity Check:")
        current_date = datetime.now().strftime("%b %Y")
        print(f"Current Nasdaq Level ({current_date}): ~{current_nasdaq_level:.2f}")
        percent_diff = ((next_day_prediction - current_nasdaq_level) / current_nasdaq_level) * 100
        print(f"Prediction differs from current by: {percent_diff:.2f}%")
        if abs(percent_diff) > 10:
            print("WARNING, prediction deviates significantly from current market levels")

        print("\nContribution of New Data Sources:")
        for source_prefix, name in [('TNX', '10-Year Treasury'), ('NQF', 'Nasdaq-100 Futures'), ('USD', 'USD Index')]:
            source_features = [i for i in metrics['feature_importance']['feature'] if source_prefix in str(i)]
            if source_features:
                source_importance = metrics['feature_importance'][
                    metrics['feature_importance']['feature'].isin(source_features)
                ]['importance'].sum()
                print(f"{name} Features: {source_importance:.4f}")
        
        return {
            'model_file': model_filename,
            'feature_scaler_file': feature_scaler_filename,
            'target_scaler_file': target_scaler_filename,
            'prediction': next_day_prediction,
            'predicted_change': predicted_change
        }
    
    except Exception as e:
        print(f"An error occurred in the main execution: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    main()