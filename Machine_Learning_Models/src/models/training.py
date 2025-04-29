import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping

from src.models.ann import build_ann_model

def train_model(X, y, test_size=0.2, random_state=42):

    if X is None or y is None:
        raise ValueError("Invalid input data: X or y is None")
        
    if X.empty or y.empty:
        raise ValueError("Empty input data provided")
    
    if len(X) < 100:
        raise ValueError(f"Insufficient data: {len(X)} samples. Need at least 100.")

    min_test_samples = max(20, int(len(X) * 0.1))
    actual_test_size = max(min(test_size, 0.5), min_test_samples / len(X))
    print(f"Using test_size: {actual_test_size}")

    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=actual_test_size, shuffle=False
    )

    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

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
        X_train_scaled, y_train_scaled,
        validation_split=0.2,
        epochs=150,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    y_pred_scaled = model.predict(X_test_scaled).flatten()

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

if __name__ == "__main__":
    print("module is not meant to be run directly")
    print("import and use functions in main script")