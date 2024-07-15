# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:57:19 2024

@author: Zhaleh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import logging
import seaborn as sns

class Utils:
    def __init__(self, log_file="app.log"):
        self.setup_logging(log_file)

    def setup_logging(self,log_file):
        """
        Set up logging configuration.
        """
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s:%(levelname)s:%(message)s",
        )

    def calculate_metrics(self, actual, predicted):
        """
        Calculate and return common evaluation metrics.
        """
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = mean_absolute_percentage_error(actual, predicted) * 100
        ds = (
            np.mean(
                np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])
            )
            * 100
        )
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        return rmse, mape, mse, mae, ds

    def plot_summary_metrics(self, results, output_file='results/summary_results.png'):
        """
        Plot the summary metrics.
        """
        plt.figure(figsize=(12, 6))

        # Boxplot for RMSE
        plt.subplot(1, 2, 1)
        sns.boxplot(y=results['RMSE'])
        plt.title('Boxplot of RMSE')
        plt.ylabel('RMSE')

        # Boxplot for MAPE
        plt.subplot(1, 2, 2)
        sns.boxplot(y=results['MAPE'])
        plt.title('Boxplot of MAPE')
        plt.ylabel('MAPE')

        plt.tight_layout()
        plt.suptitle('Error Metrics Summary')
        plt.savefig(output_file)
        plt.show()

    def plot_results(self, df, date_col='WeekNum', target_col='Demand', title="Actual demand vs Predicted", output_file='results/demand_forecast.png'):
        """
        Plot the aggregated actual and forecasted values.
        """
        # Group by date_col and calculate the mean for 'Actual' and 'Forecast'
        aggregated_df = df.groupby(date_col).mean().reset_index()
        
        plt.figure(figsize=(10, 6))
        plt.plot(aggregated_df[date_col], aggregated_df['Actual'], label='Actual', marker='o')
        plt.plot(aggregated_df[date_col], aggregated_df['Forecast'], label='Forecast', marker='x')

        plt.title(title)
        plt.xlabel(date_col)
        plt.ylabel(target_col)
        plt.legend()
        plt.savefig(output_file)
        plt.show()

    def plot_loss_history(self, history, output_file='results/loss_history.png'):
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.savefig(output_file)
        plt.show()
# Example usage:
# utils = Utils()
# logging.info("Logging setup complete.")
