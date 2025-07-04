import logging
import os
from pathlib import Path

def setup_logging(log_level='INFO'):
    Path("logs").mkdir(exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/military_oil_predictor.log'),
            logging.StreamHandler()
        ]
    )

import math
from typing import Tuple

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    r = 6371
    return c * r

def is_point_in_radius(center_lat: float, center_lon: float, 
                      point_lat: float, point_lon: float, 
                      radius_km: float) -> bool:
    distance = haversine_distance(center_lat, center_lon, point_lat, point_lon)
    return distance <= radius_km

def get_bounding_box(lat: float, lon: float, radius_km: float) -> Tuple[float, float, float, float]:
    lat_delta = radius_km / 111.32  # 1 degree lat ≈ 111.32 km
    lon_delta = radius_km / (111.32 * math.cos(math.radians(lat)))
    
    return (
        lat - lat_delta,  # min_lat
        lat + lat_delta,  # max_lat
        lon - lon_delta,  # min_lon
        lon + lon_delta   # max_lon
    )