"""
Baseline: XGBoost 
Install xgboost: pip install xgboost
Run: python baselines/baseline_XGBoost.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from xgboost import XGBRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from starter_code.load_data import load_split  # noqa: E402
from starter_code.evaluate_validation import evaluate_predictions  # noqa: E402
from starter_code.create_submission import build_submission  # noqa: E402

"""
Function to make create the XGBoost Model 
"""
def make_model() -> XGBRegressor:
        return XGBRegressor(
               objective='reg:squarederror',
               max_depth=6,
               learning_rate=0.1,
               n_estimators=100, 
               random_state=42,
)

"""
Function to train the XGBoost model
"""
def train_model(model:XGBRegressor, X_train: np.array, Y_train: np.array, valid_mask: np.ndarray) -> XGBRegressor:
    
    mask = valid_mask.reshape(-1) # [samples, sensors] is flatten to [samples * sensors]

    X_clean = X_train[mask]
    Y_clean = Y_train[mask]

    model.fit(X_clean, Y_clean)

    return model
       
"""
Function to prepare input data:
XGBoost requires input data to be in the shape of a 2d matrix [rows, features]
Converts:
[num_samples, timesteps, num_sensors]
into:
[num_samples*num_sensors, timesteps]
"""
def prepare_features(X: np.array) -> np.array: 
       
       X = X.transpose(0,2,1) # Converts X to [num_samples, num_sensors, timestep] 

       num_samples = X.shape[0]
       num_sensors= X.shape[1]
       timestep = X.shape[2]

       X = X.reshape(num_samples*num_sensors, timestep) # Reshapes X to [samples * sensors, timesteps]

       return X

"""
Function to prepare our target variable Y
"""
def prepare_targets(Y: np.ndarray) -> np.ndarray:
    
    Y = Y.transpose(0, 2, 1)

    num_samples = Y.shape[0]
    num_sensors = Y.shape[1]
    timesteps = Y.shape[2]

    return Y.reshape(num_samples * num_sensors, timesteps)

"""
Function to covert timestamps into numerical time features
Converts: 
['2012-03-01T00:55:00.000000000']
into
[hours, minutes, total_minutes, AP/PM, day_of_week, is_weekend]
"""
def prepare_time(timestamps: np.ndarray, num_sensors: int) -> np.ndarray: 
      
      timestamps = pd.to_datetime(timestamps) # ['2012-03-01T00:55:00.000000000'] -> ['2012-03-01 00:55:00'] March 1st 2012 at 12:55AM [date(year-month-day) time(HH:MM:SS)]

      hours = timestamps.hour # Get hours 
      minutes = timestamps.minute # Get minute 

      total_minutes = hours * 60 + minutes # Get minutes since 12am (Midnight)

      time_of_day = (hours >= 12).astype(int) # Determine if AM(1) or PM(0) 

      day_of_week = timestamps.day_of_week # Get day of the week Monday = 0 Sunday = 6

      is_weekend = (day_of_week >= 5).astype(int) # Determine if weekend(1) or weekday(0)

      time_features = np.column_stack((hours,minutes,total_minutes,time_of_day,day_of_week,is_weekend))

      time_features = np.repeat(time_features,num_sensors, axis = 0)

      return time_features

"""
Function to restore prediction back to original shape [num_samples, timesteps, num_sensors]
"""
def restore_shape(pred: np.array, samples: int, sensors: int) -> np.array:
      
      timesteps = pred.shape[1] # Reshape from [samples * sensors, timesteps] to [samples, sensors, timesteps]
      pred = pred.reshape(samples, sensors,timesteps)
      pred = pred.transpose(0, 2, 1) # Reorder from [samples, sensors, timesteps] to orginal [num_samples, timesteps, num_sensors]

      return pred

def main() -> None:
        print("Baseline: XGboost\n")

        train = load_split("train") # Load train, val, and test data
        val = load_split("validation")
        test = load_split("test_features")

        X_train = prepare_features(train['X']) # Prepare input data 
        X_val = prepare_features(val['X'])
        X_test = prepare_features(test['X'])

        num_sensors = train['X'].shape[2]

        prepare_time(train["timestamps"],207) # Prepare time features for train, val, and test data 
        time_train = prepare_time(train['timestamps'], num_sensors)
        time_val = prepare_time(val['timestamps'], num_sensors)
        time_test = prepare_time(test['timestamps'],num_sensors)

        # Add new features to data 
        X_train = np.concatenate((X_train, time_train), axis = 1)
        X_val = np.concatenate((X_val, time_val), axis=1)
        X_test = np.concatenate((X_test, time_test),axis=1)

        Y_train = prepare_targets(train['Y']) # Prepare target data (Y)
      
        xgb_model = make_model() # Make the xgboost model

        trained_model = train_model(xgb_model, X_train, Y_train, train["valid_mask"]) # Train the xgboost model 

        print("\nTrained one XGBoost Model")

        pred = trained_model.predict(X_val) # Make predictions
        print("Predictions:", pred.shape) # pred.shape = (samples*sensors, features)

        reshaped_pred = restore_shape(pred, val['X'].shape[0], val['X'].shape[2]) # Reshape predictions 
        print("Predictions reshaped:", reshaped_pred.shape)
        print("Actual values (Y) shape:", val['Y'].shape)

        print("\nValidation performance:") # Evaluation 
        evaluate_predictions(reshaped_pred)

if __name__ == "__main__":
    main()