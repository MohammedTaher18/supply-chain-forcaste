import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from feature_config import LSTM_FEATURES, get_available_features

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

device = torch.device("cpu")
print(f"Using device: {device}")


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def create_sequences(data, seq_length=14):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)


def calculate_wmape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred).sum()
    denominator = np.abs(y_true).sum()
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100


def run_simple_lstm(train_df, test_df, epochs=5):
    print("=" * 50)
    print("Training Simple LSTM")
    print("=" * 50)
    
    features = get_available_features(train_df, LSTM_FEATURES)
    input_size = len(features)
    
    train_data = train_df[features].fillna(0).values
    test_data = test_df[features].fillna(0).values
    
    X_train, y_train = create_sequences(train_data, 14)
    X_test, y_test = create_sequences(test_data, 14)
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    model = SimpleLSTM(input_size=input_size, hidden_size=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    train_loader = DataLoader(list(zip(X_train_t, y_train_t)), batch_size=256, shuffle=True)
    
    print(f"Training {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            pred = model(X_test_t.to(device)).squeeze()
            mae = mean_absolute_error(y_test_t.numpy(), pred.cpu().numpy())
        
        print(f"Epoch {epoch+1}/{epochs} - MAE: {mae:.4f}")
    
    model.eval()
    with torch.no_grad():
        pred = model(X_test_t.to(device)).squeeze()
        pred = pred.cpu().numpy()
    
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    wmape = calculate_wmape(y_test, pred)
    
    print(f"\nLSTM Results: MAE={mae:.4f}, RMSE={rmse:.4f}, WMAPE={wmape:.2f}%")
    
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "lstm_model.pt"))
    
    results = {"model": "LSTM", "MAE": mae, "RMSE": rmse, "WMAPE": wmape}
    pd.DataFrame([results]).to_csv(os.path.join(MODELS_DIR, "lstm_metrics.csv"), index=False)
    
    return results


if __name__ == "__main__":
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    run_simple_lstm(train_df, test_df, epochs=5)