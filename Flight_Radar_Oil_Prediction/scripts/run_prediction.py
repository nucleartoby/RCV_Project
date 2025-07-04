import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

from src.ml_models.predictor_model import OilPricePredictor
from src.ml_models.feature_engineering import FeatureEngineer
from src.data_collection.oil_price_collector import OilPriceCollector
from src.data_collection.flightradar_scraper import FlightRadarScraper
from src.data_collection.base_monitor import BaseMonitor
from src.utils.logging_config import setup_logging

def load_latest_model():
    model_files = list(Path("data/models").glob("oil_predictor_*.pkl"))
    if not model_files:
        raise FileNotFoundError("No trained models found")
    
    latest_model = max(model_files, key=os.path.getctime)
    
    predictor = OilPricePredictor()
    predictor.load_model(latest_model)
    
    return predictor

def collect_real_time_data():
    oil_collector = OilPriceCollector()
    flight_scraper = FlightRadarScraper()
    base_monitor = BaseMonitor()
    
    try:
        current_prices = oil_collector.fetch_current_prices()
        recent_oil = oil_collector.fetch_historical_data(days=7)
        
        flights = flight_scraper.get_middle_east_flights()
        
        if flights:
            flight_df = pd.DataFrame(flights)
            flight_df['is_military'] = flight_df['callsign'].apply(
                base_monitor.is_military_aircraft
            )

            base_info = []
            for _, flight in flight_df.iterrows():
                is_near, base_name = base_monitor.is_near_base(
                    flight['latitude'], flight['longitude']
                )
                base_info.append({
                    'is_near_base': is_near,
                    'base_name': base_name if is_near else 'None'
                })
            
            base_df = pd.DataFrame(base_info)
            flight_df = pd.concat([flight_df, base_df], axis=1)
            
            return recent_oil, flight_df, current_prices
        
        return recent_oil, pd.DataFrame(), current_prices
    
    finally:
        flight_scraper.close_driver()

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting oil price prediction...")
    
    try:
        logger.info("Loading trained model...")
        predictor = load_latest_model()
        
        logger.info("Collecting real-time data...")
        oil_data, flight_data, current_prices = collect_real_time_data()
        
        logger.info("Engineering features for prediction...")
        feature_engineer = FeatureEngineer()
        
        if not flight_data.empty:
            flight_features = feature_engineer.create_flight_features(flight_data)
        else:
            logger.warning("No current flight data available")
            flight_features = pd.DataFrame()
        
        oil_features = feature_engineer.create_oil_features(oil_data)

        if not flight_features.empty:
            combined_features = feature_engineer.combine_features(
                flight_features, oil_features
            )
        else:
            combined_features = oil_features
            logger.warning("Prediction based only on oil price features")

        logger.info("Making predictions...")

        latest_features = combined_features.iloc[-1:].values
        prediction = predictor.predict(latest_features)[0]

        logger.info("="*60)
        logger.info("OIL PRICE PREDICTION REPORT")
        logger.info("="*60)
        logger.info(f"Prediction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Current Brent Crude (BZ=F): ${current_prices.get('BZ=F', 'N/A')}")
        logger.info(f"Current WTI Crude (CL=F): ${current_prices.get('CL=F', 'N/A')}")
        logger.info(f"")
        logger.info(f"PREDICTED BRENT CRUDE PRICE (Next Period): ${prediction:.2f}")
        
        if not flight_data.empty:
            military_count = flight_data['is_military'].sum()
            total_flights = len(flight_data)
            logger.info(f"")
            logger.info(f"Current Military Activity:")
            logger.info(f"  Total Flights Monitored: {total_flights}")
            logger.info(f"  Military Flights: {military_count}")
            logger.info(f"  Military Flight Ratio: {military_count/total_flights:.2%}")
        
        logger.info("="*60)

        prediction_data = {
            'timestamp': datetime.now().isoformat(),
            'predicted_price': float(prediction),
            'current_bz_price': current_prices.get('BZ=F'),
            'current_cl_price': current_prices.get('CL=F'),
            'military_flights': int(flight_data['is_military'].sum()) if not flight_data.empty else 0,
            'total_flights': len(flight_data),
            'model_used': predictor.best_model_name
        }

        Path("data/predictions").mkdir(parents=True, exist_ok=True)
        prediction_file = f"data/predictions/prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        with open(prediction_file, 'w') as f:
            json.dump(prediction_data, f, indent=2)
        
        logger.info(f"Prediction saved to {prediction_file}")
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return 1
    
    logger.info("Prediction completed successfully")
    return 0

if __name__ == "__main__":
    exit(main())