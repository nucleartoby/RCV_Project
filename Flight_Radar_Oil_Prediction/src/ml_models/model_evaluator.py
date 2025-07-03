import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import logging

class ModelEvaluator:

    def __init__(self):
        self.logger = logging.getlogger(__name__)

    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                         title: str = "Predictions vs Actual"):
        plt.figure(figzie=(12, 8))

        plt.subplot(2, 2, 1)
        plt.scattery(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min()])