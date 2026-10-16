import os
import torch
import torch.nn as nn
import joblib

class BudgetMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, values):
        return self.network(values)

MODEL_PATH = "../models/mlp.pth" 
X_SCALER_PATH = "../models/mlp_x_scaler.joblib"
Y_SCALER_PATH = "../models/mlp_y_scaler.joblib"
model = BudgetMLP(input_size=10) 

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print(f" Weights loaded successfully. Last sync timestamp: {os.path.getmtime(MODEL_PATH)}")
if os.path.exists(X_SCALER_PATH) and os.path.exists(Y_SCALER_PATH):
    x_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)
else:
    print("Scaler files not detected yet. Check your Desktop Sync status.")

