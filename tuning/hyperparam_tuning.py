# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:56:35 2024

@author: Zhaleh
"""

# hyperparameter_tuning.py
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from statsmodels.tsa.api import VARMAX
from statsmodels.tsa.stattools import arma_order_select_ic
import numpy as np
from data_prep.data_preparation import DataPrep
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

class HyperparameterTuning:
    def __init__(self):
        self.data_prep= DataPrep()

    def arima_param_selection(self, data, p_values, q_values):
        """
        Find the best ARIMA parameters 
        """

        best_score, best_cfg = float("inf"), None
        stat_data, d = self.data_prep.make_stationary(data, n_diff=0)
       
        for p in p_values:
            for q in q_values:
                order = (p, d, q)
                self.data_prep.n_diff = 0
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        model = ARIMA(data,order=order)
                        model_fit = model.fit()
                        mse = mean_squared_error(data, model_fit.fittedvalues)
                        if mse < best_score:
                            best_score, best_cfg = mse, order
                except:
                    continue
        return best_cfg

    def find_best_varma_order(self, data, max_p=5, max_q=5, criterion='aic'):
        """
        Find the best (p, q) order for a VARMA model based on the specified information criterion.

        Parameters:
        - data: pandas DataFrame, the time series data.
        - max_p: int, maximum number of lags for the autoregressive part.
        - max_q: int, maximum number of lags for the moving average part.
        - criterion: str, the information criterion to minimize ('aic', 'bic', etc.).

        Returns:
        - best_order: tuple, the best (p, q) order.
        """
        best_order = None
        best_ic = np.inf

        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    model = VARMAX(data, order=(p, q), trend='n').fit(disp=False)
                    ic_value = getattr(model, criterion)
                    if ic_value < best_ic:
                        best_ic = ic_value
                        best_order = (p, q)
                except Exception as e:
                    continue

        return best_order, best_ic

    # def lstm_param_selection(self, train_x,train_y, param_grid, lstm_model_func):
    #     """
    #     Find the best LSTM parameters using grid search.
    #     """
    #     model = KerasRegressor(lstm_model_func, epochs=10, batch_size=16, verbose=0)
    #     scorer = make_scorer(mean_squared_error, greater_is_better=False)
    #     grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=3, scoring=scorer)
    #     grid_result = grid.fit(train_x,train_y)
    #     return grid_result.best_params_

# Example usage:
# tuning = HyperparameterTuning()
# data = pd.read_csv('your_time_series_data.csv')
# best_arima_order = tuning.arima_param_selection(data, p_values=[1,2,3], d_values=1, q_values=[1,2,3])
# print(f'Best ARIMA order: {best_arima_order}')
# best_varma_order, best_ic = tuning.find_best_varma_order(data)
# print(f'Best VARMA order: {best_varma_order} with {criterion.upper()}: {best_ic}')
# param_grid = {'units': [50, 100], 'batch_size': [32, 64], 'epochs': [10, 20]}
# best_lstm_params = tuning.lstm_param_selection(train_data, param_grid, lstm_model_func)
# print(f'Best LSTM params: {best_lstm_params}')
