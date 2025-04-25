import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

def build_ann_model(input_shape):
    """
    Build an enhanced Artificial Neural Network model with additional capacity
    for handling more features
    """
    model = Sequential([
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

if __name__ == "__main__":
      print("module is not meant to be run directly")
      print("import and use functions in main script")