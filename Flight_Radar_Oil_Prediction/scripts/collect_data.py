import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import json
from datetime import datetime, timedelta
import logging
from pathlib import Path

from src.data_collection.oil_price_collector import OilPriceCollector
from src.data_collection.flightradar_scraper import FlightRadarScraper
from src.data_collection.base_monitor import BaseMonitor
from src.utils.logging_config import setup_logging

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting data collection...")

    oil_collector = OilPriceCollector()
    flight_scraper = FlightRadarScraper()
    base_monitor = BaseMonitor()
    
    try:
        Path("data/raw/flight_data").mkdir(parents=True, exist_ok=True)
        Path("data/raw/oil_prices").mkdir(parents=True, exist_ok=True)

        logger.info("Collecting oil price data...")
        current_prices = oil_collector.fetch_current_prices()
        historical_oil = oil_collector.fetch_historical_data(days=90)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        oil_file = f"data/raw/oil_prices/oil_prices_{timestamp}.csv"
        historical_oil.to_csv(oil_file, index=False)
        
        with open(f"data/raw/oil_prices/current_prices_{timestamp}.json", 'w') as f:
            json.dump(current_prices, f)
        
        logger.info(f"Oil data saved to {oil_file}")

        logger.info("Collecting flight data...")
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

            flight_file = f"data/raw/flight_data/flights_{timestamp}.csv"
            flight_df.to_csv(flight_file, index=False)

            activity_summary = base_monitor.categorize_activity(flights)
            with open(f"data/raw/flight_data/activity_summary_{timestamp}.json", 'w') as f:
                json.dump(activity_summary, f)
            
            logger.info(f"Flight data saved to {flight_file}")
            logger.info(f"Military flights detected: {activity_summary['total_military_flights']}")
            
        else:
            logger.warning("No flight data collected")
        
    except Exception as e:
        logger.error(f"Error during data collection: {e}")
        return 1
    
    finally:
        flight_scraper.close_driver()
    
    logger.info("Data collection completed successfully")
    return 0

if __name__ == "__main__":
    exit(main())