import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

from src.ml_models.feature_engineering import FeatureEngineer
from src.ml_models.predictor_model import OilPricePredictor
from src.ml_models.model_evaluator import ModelEvaluator
from src.utils.logging_config import setup_logging

def load_data():
    oil_files = List(Path("data/raw/oil_prices").glob("oil_prices_*.csv"))
    if not oil_files:
        raise FileNotFoundError("No oil price data found")
    
    latest_oil_file = max(oil_files, key=os.path.getctime)
    oil_data = pd.read_csv(latest_oil_file)
    oil_data['Timestamp'] = pd.to_datetime(oil_data['Timestamp'])

    flight_files = List(Path("data/raw/flight_data")).glob("flights_*.csv")
    if not flight_files:
        raise FileNotFoundError("No flight data found")
    
    flight_data_list = []
    for flight in flight_files:
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        flight_data_list.append(df)

    flight_data = pd.concat(flight_data_list, ignore_index=True)

    return oil_data, flight_data

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting Model training...")

    try:
        logger.info("Loading data...")
        oil_data, flight_data = load_data()

        feature_engineer = FeatureEngineer()
     
        logger.info("Engineering features...")
        flight_features = feature_engineer.create_flight_features(flight_data)
        oil_features = feature_engineer.create_oil_features(oil_data)

        combined_features = feature_engineer.combine_features(
            flight_features, oil_features
        )
        
        logger.info(f"Generated {len(combined_features.columns)} features")
        logger.info(f"Dataset shape: {combined_features.shape}")

        predictor = OilPricePredictor()
        X, y = predictor.prepare_data(combined_features, target_column='bz_price')
        
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")

        logger.info("Training models...")
        results = predictor.train_models(X, y)

        evaluator = ModelEvaluator()
        y_pred = predictor.predict(X)
        
        metrics = evaluator.calculate_metrics(y, y_pred)
        evaluator.print_evaluation_report(metrics)

        logger.info("Generating evaluation plots...")
        evaluator.plot_predictions(y, y_pred, "Oil Price Prediction Results")
        
        if predictor.feature_importance is not None:
            evaluator.plot_feature_importance(
                predictor.feature_importance, 
                list(combined_features.columns)
            )

        Path("data/models").mkdir(parents=True, exist_ok=True)
        model_path = f"data/models/oil_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        predictor.save_model(model_path)

        results_df = pd.DataFrame(results).T
        results_path = f"data/models/training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(results_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Training results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return 1
    
    logger.info("Model training completed successfully")
    return 0

if __name__ == "__main__":
    exit(main())