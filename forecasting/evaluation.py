# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:28:02 2024

@author: Zhaleh
"""

# forecasting/evaluation.py
import numpy as np
import pandas as pd
from data_prep.data_preparation import DataPrep
from models import arima_model
from tuning.hyperparam_tuning import HyperparameterTuning
from utils.utils import Utils
from models.lstm_model import LSTMModel


class ModelEvaluator:
    def __init__(self, data, n_window, n_features, n_forecast):
        self.data = data
        self.data_prep = DataPrep()
        self.scaled_data, self.scaler = self.data_prep.scale_data(data)
        self.n_window = n_window
        self.n_features = n_features
        self.n_forecast=n_forecast
        self.utils = Utils()


    def rolling_window_evaluation(self, model_type, window_size, steps=1, **kwargs):
        """
        Perform rolling window evaluation on the data and store the results.
        """
        results = []        
        values= self.scaled_data.reshape(self.scaled_data.shape[0],self.n_features)

        # Initialize hyperparameter tuning
        tuning = HyperparameterTuning()

        for end in range(window_size, len(self.data) - steps):
            
            train = values[end - window_size:end]
           
            test = values[end:end + steps]
           
            test_week_numbers = range(end + 1, end + steps + 1)

            if model_type == 'LSTM': 
                spliting_point= int(round(end*0.8)) 
                          
                train_data = values[:spliting_point]
                val_data= values[spliting_point:end]
                test_data = values[(end - self.n_window):(end + self.n_forecast)]
            
           
                # Frame as supervised learning
                reframed = self.data_prep.series_to_supervised(pd.DataFrame(train_data), self.n_window, self.n_forecast, dropnan=True)
                val_supervised= self.data_prep.series_to_supervised(pd.DataFrame(val_data), self.n_window, self.n_forecast)
                
                train_x, train_y = reframed.iloc[:, :self.n_window*self.n_features].values, reframed.iloc[:, self.n_window*self.n_features:self.n_window*self.n_features + self.n_forecast*self.n_features].values
               
                val_x, val_y= val_supervised.iloc[:,:self.n_window*self.n_features].values, val_supervised.iloc[:, self.n_window*self.n_features:self.n_window*self.n_features + self.n_forecast*self.n_features].values
                
                test_x, test_y = test_data[:self.n_window].reshape(1,-1), test_data[self.n_window:self.n_window + self.n_forecast].reshape(1,-1)
               
                lstm_model = LSTMModel(self.n_window, self.n_features, self.n_forecast)

                # un-comment lines below, if param tuning is needed
                # param_grid = {'epochs': [10, 20, 50], 'batch_size': [32, 64]}       
                # best_params = tuning.lstm_param_selection(train_x,train_y, param_grid, model)

                #comment line below if applying hyper param tuning
                best_params={'epochs': 100,'batch_size': 64}
                print(best_params)
                kwargs.update(best_params)
                
                history=lstm_model.train(train_x, train_y,val_x, val_y, **kwargs)
                self.utils.plot_loss_history(history)
                forecast= lstm_model.model.predict(test_x,verbose=0)
                
                forecast = np.array(forecast).reshape(-1, 1)
                
                forecast = self.scaler.inverse_transform(forecast.reshape(self.n_forecast, self.n_features))

            elif model_type == 'ARIMA':
                best_params = tuning.arima_param_selection(train, p_values=[0, 1, 2,3,4], q_values=[0, 1, 2,3,4])
                model = arima_model.ARIMAA(order=best_params)                
                model.train(train)
                forecast= model.forecast(steps)
                forecast = np.array(forecast).reshape(-1, 1)
                forecast=self.scaler.inverse_transform(forecast).reshape(-1, 1)

            elif model_type == 'AUTO-ARIMA':
                model = arima_model.AutoARIMA()
                model.train(train)
                forecast= model.forecast(steps)
                forecast = np.array(forecast).reshape(-1, 1)
                forecast= self.scaler.inverse_transform(forecast).reshape(-1, 1)                
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            test = np.array(self.scaler.inverse_transform(test)).reshape(-1, 1)
         
            rmse, mape, mse, mae, ds = self.utils.calculate_metrics(test[:, 0], forecast[:, 0])

            #save results 
            for i in range(steps - 1):
                results.append((test_week_numbers[i], test[i, 0], forecast[i, 0]))
            results.append((test_week_numbers[i + 1], test[i + 1, 0], forecast[i + 1, 0], rmse, mape, mse, mae, ds))

        results_df = pd.DataFrame(results, columns=['WeekNum', 'Actual', 'Forecast', 'RMSE', 'MAPE', 'MSE', 'MAE', 'DS'])
        return results_df

    def expanding_window_evaluation(self, model_type, initial_window, steps=1, **kwargs):
        """
        Perform expanding window evaluation on the data and store the results.
        """
        results = [] 
        values= self.scaled_data.reshape(self.scaled_data.shape[0],self.n_features)

        # Initialize hyperparameter tuning
        tuning = HyperparameterTuning()
        for end in range(initial_window, len(self.data) - steps):
            train = values[end - initial_window:end]          
            test = values[end:end + steps]
            
            test_week_numbers = range(end + 1, end + steps + 1)

            if model_type == 'LSTM':                
                train_data = values[:end]
                val_data= values[end:]
                test_data = values[(end - self.n_window):(end + self.n_forecast)]
            
           
                # Frame as supervised learning
                reframed = self.data_prep.series_to_supervised(pd.DataFrame(train_data), self.n_window, self.n_forecast, dropnan=True)
                val_supervised= self.data_prep.series_to_supervised(pd.DataFrame(val_data), self.n_window, self.n_forecast)
                
                train_x, train_y = reframed.iloc[:, :self.n_window*self.n_features].values, reframed.iloc[:, self.n_window*self.n_features:self.n_window*self.n_features + self.n_forecast*self.n_features].values
                val_x, val_y= val_supervised.iloc[:,:self.n_window*self.n_features].values, val_supervised.iloc[:, self.n_window*self.n_features:self.n_window*self.n_features + self.n_forecast*self.n_features].values
                
                test_x, test_y = test_data[:self.n_window].reshape(1,-1), test_data[self.n_window:self.n_window + self.n_forecast].reshape(1,-1)
               
                lstm_model = LSTMModel(self.n_window, self.n_features, self.n_forecast)

                # un-comment lines below, if param tuning is needed
                # param_grid = {'epochs': [10, 20, 50], 'batch_size': [32, 64]}
                # best_params = tuning.lstm_param_selection(train_x,train_y, param_grid, model)

                #comment line below if applied hyper param tuning
                best_params={'epochs': 50,'batch_size': 64}
                
                print(best_params)
                kwargs.update(best_params)
                
                lstm_model.train(train_x, train_y, **kwargs)
         
                forecast= lstm_model.model.predict(test_x,verbose=0)
                forecast = np.array(forecast).reshape(-1, 1)
             
                forecast = self.scaler.inverse_transform(forecast.reshape(self.n_forecast, self.n_features))

            elif model_type == 'ARIMA':
                best_params = tuning.arima_param_selection(train, p_values=[0, 1, 2,3,4], q_values=[0, 1, 2,3,4])
                model = arima_model.ARIMAA(order=best_params)                
                model.train(train)
                forecast= model.forecast(steps)
                forecast = np.array(forecast).reshape(-1, 1)
                forecast=self.scaler.inverse_transform(forecast).reshape(-1, 1)
            elif model_type == 'AUTO-ARIMA':
                model = arima_model.AutoARIMA()
                model.train(train)
                forecast= model.forecast(steps)
                forecast = np.array(forecast).reshape(-1, 1)
                forecast= self.scaler.inverse_transform(forecast).reshape(-1, 1)                
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            test = np.array(self.scaler.inverse_transform(test)).reshape(-1, 1)
          
            rmse, mape, mse, mae, ds = self.utils.calculate_metrics(test[:, 0], forecast[:, 0])
            for i in range(steps - 1):
                results.append((test_week_numbers[i], test[i, 0], forecast[i, 0]))
            results.append((test_week_numbers[i + 1], test[i + 1, 0], forecast[i + 1, 0], rmse, mape, mse, mae, ds))
            results_df = pd.DataFrame(results, columns=['WeekNum', 'Actual', 'Forecast', 'RMSE', 'MAPE', 'MSE', 'MAE', 'DS'])
        return results_df