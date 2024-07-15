# Univariate Time Series Forecasting

This project provides implementations for univariate time series forecasting using ARIMA, Auto-ARIMA, and LSTM models. The code is modular and designed for easy adaptation to various datasets and forecasting requirements.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Data Preparation](#data-preparation)
6. [Models](#models)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Contributing](#contributing)
10. [License](#license)

## Introduction

This repository contains code for forecasting univariate time series data using ARIMA, Auto-ARIMA, and LSTM models. The project is structured to be easily extensible and reusable for different datasets and forecasting scenarios.


## Features

- Data preparation and scaling
- Hyperparameter tuning for ARIMA and LSTM models
- Rolling and expanding window evaluation methods
- Plotting results and summary metrics

## Installation

### Prerequisites

- Python 3.7 or later
- Required packages listed in `requirements.txt`

### Installing

Clone the repository:
```bash
git clone https://github.com/zhaleh-rahimi/univariate-time-series-forecasting.git
cd univariate-time-series-forecasting
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Run the main script with the necessary arguments:

```bash
python main.py --file_path data/your_data.csv --date_col date --target_col target --model_type ARIMA --evaluation_type rolling --output_file results/output.csv --steps 4
```

### Arguments

- `--file_path`: Path to the CSV file containing the time series data
- `--date_col`: Name of the date column in the CSV file
- `--target_col`: Name of the target column in the CSV file
- `--model_type`: Type of model to use (`ARIMA`, `AUTO-ARIMA`, `LSTM`)
- `--evaluation_type`: Type of evaluation method (`rolling`, `expanding`)
- `--output_file`: Path to save the output CSV file
- `--steps`: Number of steps ahead to forecast

## Data Preparation

### Loading Data

The data is loaded using the `DataPrep` class, which also handles scaling and splitting the data into training and testing sets.It also can prepare a dataset for a supervised learning process, e.g. LSTM forecasting.

### Scaling Data

Data is scaled using the `MinMaxScaler` from `sklearn.preprocessing`. One may change the scaler.

## Models

### ARIMA

Implemented in the `ARIMAModel` class. Hyperparameters are selected using grid search and minimizing mse for imeplemented ARIMA.

### Auto-ARIMA

Implemented in the `AutoARIMA` class using the `pmdarima` library.

### LSTM

Implemented in the `LSTMModel` class. Hyperparameters are selected using grid search.

## Evaluation

### Rolling Window Evaluation

Evaluates the model performance using a rolling window approach.

### Expanding Window Evaluation

Evaluates the model performance using an expanding window approach.

## Results

The results are saved to a CSV file and can be visualized using the provided plotting functions.

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature-foo`)
3. Commit your changes (`git commit -am 'Add some foo'`)
4. Push to the branch (`git push origin feature-foo`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.