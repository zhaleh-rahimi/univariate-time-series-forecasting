# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:57:29 2024

@author: Zhaleh
"""

# main.py
import argparse
import logging
from data_prep.data_preparation import DataPrep
from forecasting.evaluation import ModelEvaluator
from models.lstm_model import LSTMModel
from tuning.hyperparam_tuning import HyperparameterTuning
from utils.utils import Utils
import numpy as np

def main(file_path, date_col, target_col, model_type, evaluation_type, output_file, steps, **kwargs):
    utils = Utils()
    logging.info("Starting time series forecasting application.")

    try:
        # Load and prepare data
        logging.info("Loading and preparing data.")
        data_prep = DataPrep(file_path=file_path, target_col=target_col)
        data = data_prep.get_data()
        
        # Initialize model evaluator
        evaluator = ModelEvaluator(data, n_window=10, n_features=1, n_forecast=steps)

        # Choose evaluation method
        logging.info(f"Evaluating model using {evaluation_type}.")
        if evaluation_type == 'rolling':
            results_df = evaluator.rolling_window_evaluation(model_type, window_size=52, steps=steps, **kwargs)
        elif evaluation_type == 'expanding':
            results_df = evaluator.expanding_window_evaluation(model_type, initial_window=52, steps=steps, **kwargs)
        else:
            raise ValueError(f"Unknown evaluation type: {evaluation_type}")

       
        # Save results to CSV
        results_df.to_csv(output_file, index=False)
        print(results_df)
        logging.info(f"Results saved to {output_file}")

        # Calculate and log metrics
        rmse, mape, mse, mae, ds = np.mean(results_df['RMSE']), np.mean(results_df['MAPE']), np.mean(results_df['MSE']), np.mean(results_df['MAE']), np.mean(results_df['DS'])
        logging.info(f"Evaluation metrics - RMSE: {rmse}, MAPE: {mape}, MSE: {mse}, MAE: {mae}, DS: {ds}")

        # Plot the results
        utils.plot_results(results_df,title=model_type)
        utils.plot_summary_metrics(results_df)

    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series Forecasting')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--date_col', type=str, required=True, help='Name of the week number column')
    parser.add_argument('--target_col', nargs='+',type=str, required=True, help='Name of the target columns')
    parser.add_argument('--model_type', type=str, required=True, choices=['ARIMA', 'AUTO-ARIMA', 'LSTM'], help='Type of model to use')
    parser.add_argument('--evaluation_type', type=str, required=True, choices=['rolling', 'expanding'], help='Type of evaluation method')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output CSV file')
    parser.add_argument('--steps', type=int, required=True, help='Number of steps ahead to forecast')

    args = parser.parse_args()

    main(args.file_path, args.date_col, args.target_col, args.model_type, args.evaluation_type, args.output_file, args.steps)
