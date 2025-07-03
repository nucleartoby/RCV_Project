import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
from typing import Dict, Tuple, Any
import logging
from config.settings import Config

class OilPricePredictor:
<<<<<<< HEAD
    
=======
>>>>>>> d982b6569a6895c0fc2872c389194b717f1b646c
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=Config.RANDOM_STATE
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=Config.RANDOM_STATE
            ),
            'linear_regression': LinearRegression()
        }
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        self.logger = logging.getLogger(__name__)
<<<<<<< HEAD
    
    def prepare_data(self, features: pd.DataFrame, 
                    target_column: str = 'bz_price') -> Tuple[np.ndarray, np.ndarray]:
        X = features.drop(columns=[col for col in features.columns 
                                  if 'price' in col], errors='ignore')

        y = features[target_column].shift(-1).dropna()
        X = X.iloc[:-1] 
        
=======

    def prepare_data(self, features: pd.DataFrame,
                     target_colimn: str = 'bz_price') -> Tuple[np.ndarray, np.ndarray]:
        X = features.drop(columns=[col for col in features.columns
                                   if 'price' in col], errors='ignore')
        
        y = features[target_column].shift(-1).dropna()
        X = X.iloc[:-1]

>>>>>>> d982b6569a6895c0fc2872c389194b717f1b646c
        return X.values, y.values
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        X_train, X_test, y_train, y_test = train_test_split(
<<<<<<< HEAD
            X, y, test_size=1-Config.TRAIN_TEST_SPLIT, 
            random_state=Config.RANDOM_STATE
        )
        
        results = {}
        
        for name, model in self.models.items():
            self.logger.info(f"Training {name}...")

            model.fit(X_train, y_train)
            
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
=======
            X, y, test_size=1-Config.TRAIN_TEST_SPLIT,
            random_state=Config.RANDOM_STATE
        ) 

        results = {}

        for name, model in self.models,items():
            self.logger.info(f"training {name}...")

            model.fit(X_train, y_train)

            y_pred_train = model.prdict(X_train)
            y_pred_train = model.prdict(X_test)
>>>>>>> d982b6569a6895c0fc2872c389194b717f1b646c

            results[name] = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'model': model
            }

            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                      scoring='neg_mean_squared_error')
            results[name]['cv_rmse'] = np.sqrt(-cv_scores.mean())
            
            self.logger.info(f"{name} - Test RMSE: {results[name]['test_rmse']:.4f}, "
                           f"Test R²: {results[name]['test_r2']:.4f}")

        best_name = min(results.keys(), key=lambda x: results[x]['test_rmse'])
        self.best_model = results[best_name]['model']
        self.best_model_name = best_name

        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = self.best_model.feature_importances_
        
        self.logger.info(f"Best model: {best_name}")
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Model not trained yet")
        return self.best_model.predict(X)
    
    def save_model(self, filepath: str):
        if self.best_model is None:
<<<<<<< HEAD
            raise ValueError("No model to save")
=======
            raise ValueError("no model to save")
>>>>>>> d982b6569a6895c0fc2872c389194b717f1b646c
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
<<<<<<< HEAD
        self.logger.info(f"Model saved to {filepath}")
=======
        self.logger.info(f"model saved to {filepath}")
>>>>>>> d982b6569a6895c0fc2872c389194b717f1b646c
    
    def load_model(self, filepath: str):
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.feature_importance = model_data.get('feature_importance')
<<<<<<< HEAD
        self.logger.info(f"Model loaded from {filepath}")
=======
        self.logger.info(f"Model loaded from {filepath}")
            
>>>>>>> d982b6569a6895c0fc2872c389194b717f1b646c
