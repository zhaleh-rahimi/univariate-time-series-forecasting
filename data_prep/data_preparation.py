# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:56:03 2024

@author: Zhaleh
"""

# data_prep.py
import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from statsmodels.tsa.stattools import adfuller

class DataPrep:
    def __init__(self, file_path=None, date_col=None, target_col=None):
        self.file_path = file_path
        self.date_col = date_col
        self.target_col = target_col
        self.data = None
        self.scaler = None
        self.non_stationary_flags= False
        if file_path and target_col:
            self.load_data(file_path, target_col)
        if file_path and target_col and date_col:
            self.load_data(file_path, target_col,date_col)
        
    def load_data(self, file_path, target_col,date_col=None):
        """
        Load time series data from a CSV file.
        """
        self.data = read_csv(file_path, usecols= target_col, engine="python")
        if date_col:
            self.data[date_col]= pd.to_datetime(self.data[date_col])
            # Set 'date' column as the index
            self.data.set_index(date_col, inplace=True)   
        return self.data
    
    def set_data(self, data):
        """
        define the time series without a file 
        """
        self.data=data
        
    def get_data(self):
        return self.data

    def scale_data(self, data, feature_range=(0, 1)):
        """
        Scale the time series data using MinMaxScaler.
        """
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.scaled_data = self.scaler.fit_transform(data)
        return self.scaled_data, self.scaler
    
    def check_stationary(self,data):
        """Check if data is stationary"""
       
        try:
            adf_test = adfuller(data)
            
            if adf_test[1] > 0.05:  # p-value > 0.05, data is non-stationary
                self.non_stationary_flags = True
            else:
                self.non_stationary_flags = False
        except:
            self.non_stationary_flags = False
        return self.non_stationary_flags
    
    def make_stationary(self, data, n_diff=0):
        """if data is not stationary, difference it."""
        stationary_data = data.copy()
        self.non_stationary_flags= self.check_stationary(stationary_data)
      
        if self.non_stationary_flags:  # data is non-stationary
            stationary_data = pd.DataFrame(data).diff().dropna() #difference data
            n_diff += 1
            self.make_stationary(stationary_data,n_diff)
        else:
            self.non_stationary_flags=False      
       
        return stationary_data, n_diff
    
  

    def inverse_transform_forecast(self, forecast_data, n_diff):
        """Invert differencing to obtain the original values."""
        inv_forecast = forecast_data.copy()
        for i in range(n_diff):
            last_ob = pd.DataFrame(self.data).iloc[-1]
            inv_forecast = int(last_ob.iloc[0]) + np.cumsum(forecast_data)
           
        
        return inv_forecast
    
    def split_data(self, train_size):
        """
        Split the data into training and testing sets.
        """
        train_size = int(len(self.data) * train_size)
        train, test = self.data[:train_size], self.data[train_size:]
        return train, test

    def series_to_supervised(self, df, n_in=1, n_out=1, dropnan=True):
        """ convert time series to a dataset for a supervised learning, e.g. LSTM forecasting"""
       
        data = pd.DataFrame(df)
        supervised_data = pd.DataFrame()
        columns = data.columns

        # Create features
        for i in range(n_in, 0, -1):
            shifted_data = data.shift(i)
            shifted_data.columns = [f'var{col}(t-{i})' for col in columns]
            supervised_data = pd.concat([supervised_data, shifted_data], axis=1)
        
        # Create targets
        for i in range(n_out):
            shifted_data = data.shift(-i)
            if i == 0:
                shifted_data.columns = [f'var{col}(t)' for col in columns]
            else:
                shifted_data.columns = [f'var{col}(t+{i})' for col in columns]
            supervised_data = pd.concat([supervised_data, shifted_data], axis=1)
        
        # Drop rows with NaN values (due to shifting)
        supervised_data.dropna(inplace=True)
        
        return supervised_data
