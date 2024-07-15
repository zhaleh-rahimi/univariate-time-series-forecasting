# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:56:15 2024

@author: Zhaleh
"""

# arima_model.py
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from data_prep.data_preparation import DataPrep
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

class ARIMAModel:
    def __init__(self):
        self.model = None
        
    
    def train(self, train_data):
        raise NotImplementedError("Train method must be implemented by subclass")

    def forecast(self, steps=4):
        raise NotImplementedError("Forecast method must be implemented by subclass")


class AutoARIMA(ARIMAModel):
    def __init__(self):
        super().__init__()

    def train(self, train_data):
        
        self.model = auto_arima(
            train_data,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        print(self.model.order)
        return self.model

    def forecast(self, steps=4):
        forecast = self.model.predict(steps)
        return forecast


class ARIMAA(ARIMAModel):
    def __init__(self , order=(1,1,1)):
        super().__init__()
        self.model_fit=None
        self.order=order
       
        
    
    def train(self, train_data):
             
        self.model = ARIMA(train_data, order=self.order)
        self.model_fit = self.model.fit()
        print(self.order)
        return self.model_fit

    def forecast(self, steps=4):
        start = len(self.model_fit.fittedvalues)
        forecast = self.model_fit.predict(start, start + steps - 1)
      
        return forecast