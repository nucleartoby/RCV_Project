import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def visualise_feature_importance(feature_importance):
    plt.figure(figsize=(14, 8))
    feature_importance.head(15).plot(x='feature', y='importance', kind='bar')
    plt.title('Top Features')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance_ann.png')
    plt.close()

def visualise_training_history(history):
    plt.figure(figsize=(16, 8))
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

def visualise_cross_asset_relationships(df):
    try:
        required_cols = ['Close', 'TNX_Close', 'USD_Close', 'NQF_Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warning, missing columns: {missing_cols}")
            return

        plt.figure(figsize=(18, 12))

        try:
            plt.subplot(2, 2, 1)
            if 'TNX_Close' in df.columns:
                valid_mask = df[['TNX_Close', 'Close']].notna().all(axis=1)
                plt.scatter(df.loc[valid_mask, 'TNX_Close'], 
                          df.loc[valid_mask, 'Close'], alpha=0.5)
                plt.title('NASDAQ vs. 10-Year Treasury Yield')
                plt.xlabel('10-Year Treasury Yield')
                plt.ylabel('NASDAQ Close')
        except Exception as e:
            print(f"Error plotting Treasury relationship: {e}")

        try:
            plt.subplot(2, 2, 2)
            if 'USD_Close' in df.columns:
                valid_mask = df[['USD_Close', 'Close']].notna().all(axis=1)
                plt.scatter(df.loc[valid_mask, 'USD_Close'], 
                          df.loc[valid_mask, 'Close'], alpha=0.5)
                plt.title('NASDAQ vs. USD Index')
                plt.xlabel('USD Index')
                plt.ylabel('NASDAQ Close')
        except Exception as e:
            print(f"Error plotting USD relationship: {e}")

        try:
            plt.subplot(2, 2, 3)
            if 'NQF_Close' in df.columns:
                valid_mask = df[['NQF_Close', 'Close']].notna().all(axis=1)
                plt.scatter(df.loc[valid_mask, 'NQF_Close'], 
                          df.loc[valid_mask, 'Close'], alpha=0.5)
                plt.title('NASDAQ vs. NASDAQ Futures')
                plt.xlabel('NASDAQ Futures')
                plt.ylabel('NASDAQ Close')
        except Exception as e:
            print(f"Error plotting NASDAQ Futures relationship: {e}")

        try:
            plt.subplot(2, 2, 4)
            if 'USD_Close' in df.columns and 'TNX_Close' in df.columns:
                valid_mask = df[['USD_Close', 'TNX_Close']].notna().all(axis=1)
                plt.scatter(df.loc[valid_mask, 'USD_Close'], 
                          df.loc[valid_mask, 'TNX_Close'], alpha=0.5)
                plt.title('Treasury Yield vs. USD Index')
                plt.xlabel('USD Index')
                plt.ylabel('10-Year Treasury Yield')
        except Exception as e:
            print(f"Error plotting Treasury Yield vs. USD Index relationship: {e}")
        
        plt.tight_layout()
        plt.savefig('cross_asset_relationships.png')
        plt.close()
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        plt.close()

if __name__ == "__main__":
    print("module is not meant to be run directly")
    print("import and use functions in main script")