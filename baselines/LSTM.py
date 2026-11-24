"""
Deep Learning Model: Long Short Term Memory 
Install : pip install torch
Run: python baselines/LSTM.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from starter_code.load_data import load_split  # noqa: E402
from starter_code.evaluate_validation import evaluate_predictions  # noqa: E402
from starter_code.create_submission import build_submission  # noqa: E402

"""
Function to define the LSTM Model
Parameters:
input_size -> [samples, timesteps, sensors]
hidden_size -> Num of hidden units in the LSTM 
num_layers -> Num of stacked LSTM layers 
output_size -> [samples, timesteps, sensors]
"""
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
             input_size,
             hidden_size, 
             num_layers, 
             batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x): # Forward pass
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])

        # Reshape predictions
        out = out.view(-1, 12, 207)
        return out

"""
Function to make the LSTM model 
input_size -> 207 
hidden_size -> ?
num_layers -> ?
output_size -> 207 
"""
def make_model(input_size: int, hidden_size: int, num_layers: int, output_size: int) -> LSTMModel:
     
     return LSTMModel(input_size, hidden_size, num_layers, output_size)

"""
Function to train model 
"""
def train_model(model: LSTMModel, X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor, num_epochs: int) -> LSTMModel:

     model.train() 

     dataset =TensorDataset(X_train, y_train)

     train_loader = DataLoader(dataset, batch_size=32, shuffle=True) # Create batches

     optimizer = optim.Adam(model.parameters(), lr=1e-3) # Define optimizer (Adam)
     loss_fn = nn.MSELoss() # Define loss function 

     for epoch in range(num_epochs):

          total_loss = 0.0
          for X_batch, y_batch in train_loader: 

            predictions = model(X_batch)

            loss = loss_fn(predictions,y_batch) # Calculate loss 

            optimizer.zero_grad() # Zero gradients 
            loss.backward() # Backpropagation
            optimizer.step() # Update model parameters

            total_loss += loss.item()

          avg_loss = total_loss/len(train_loader) 

          print("Epoch:", epoch + 1, "Loss:", avg_loss)

     return model 

def main() -> None:
        print("Deep Learning Model: LSTM\n")

        train = load_split("train") # Load train, val, and test data
        val = load_split("validation")
        test = load_split("test_features")

        X_train = torch.tensor(train['X'], dtype=torch.float32)
        y_train = torch.tensor(train['Y'], dtype=torch.float32)

        X_val = torch.tensor(val['X'], dtype=torch.float32)
        y_val = torch.tensor(val['Y'], dtype=torch.float32)

        X_test = torch.tensor(test['X'], dtype=torch.float32)

        train_mask = torch.tensor(train['valid_mask'], dtype=torch.bool)

        print("X_train:",X_train.shape, X_train.dtype)
        print("y_train:",y_train.shape, y_train.dtype)
        print("X_val:",X_val.shape,X_val.dtype)
        print("y_val:",y_val.shape, y_val.dtype)
        print("X_test:",X_test.shape,X_test.dtype)
        print("train_mask:",train_mask.shape, train_mask.dtype)

        # Create Model 
       # lstm_model = make_model(207,64,2,2484)

       # trained_model = train_model(lstm_model, X_train, y_train, 50) # Train the LSTM 


if __name__ == "__main__":
    main()
    