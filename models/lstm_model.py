# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:56:35 2024

@author: Zhaleh
"""

from tensorflow import keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Input, Dropout, Reshape, LSTM
import numpy as np
import os
from keras._tf_keras.keras.callbacks import EarlyStopping
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class LSTMModel:
    def __init__(self, n_window=10, n_features=1, n_forecast=1):
        self.n_window = n_window
        self.n_features = n_features
        self.n_forecast = n_forecast
        self.model = self.create_lstm_model()
        

    def create_lstm_model(self,neurons=50,optimizer='adam',loss='mse'):
        """
        Create and compile an LSTM model.
        """
        input_shape = (self.n_window, self.n_features)
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(units=neurons, return_sequences=True))
        model.add(LSTM(units=neurons, return_sequences=False))
        # model.add(Dropout(0.2))
        model.add(Dense(self.n_forecast * self.n_features))
        model.add(Dense(self.n_forecast * self.n_features))
        model.add(Reshape((self.n_forecast, self.n_features)))
        model.compile(loss=loss, optimizer=optimizer, run_eagerly=True)
        return model

    def train(self, train_x, train_y, val_x, val_y, epochs=100, batch_size=64):
        """
        Train the LSTM model.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='min')
        model=self.model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(val_x,val_y))
                            #  ,callbacks=[early_stopping])

        return model
        

    def forecast(self, input_data):
        """
        Forecast using the LSTM model.
        """
        return self.model.predict(input_data, verbose=0)
        
