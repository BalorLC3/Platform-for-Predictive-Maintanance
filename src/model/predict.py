import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from src.preprocessing.processor import DataProcessor, FEATURE_COLS

def get_device(device: torch.device | str = "auto") -> torch.device:
    """
    :param device: One for 'auto', 'cuda', 'cpu'
    :return: supported PyTorch device
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout_prob: float = 0.3,
        device: torch.device | str = "auto"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.device = get_device(device)
        
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout_prob if self.num_layers > 1 else 0.0,
            batch_first=True,
            device = self.device
        )

        self.head = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        last_hidden = h_n[-1]   # (batch, hidden_dim)
        return self.head(last_hidden)
    
class ModelPredictor:
    def __init__(
            self, 
            model_path: Path,
            device: torch.device | str = "auto"
    ):
        self.model_path = model_path
        self.device = get_device(device)
        # load entire checkpoint
        checkpoint = torch.load(
            self.model_path, 
            map_location=self.device, 
            weights_only=True
        )
        
        self.model = LSTMRegressor(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers'],
            dropout_prob=checkpoint.get('dropout_prob', 0.2)
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Load normalization stats
        self.norm_stats = checkpoint['normalization_stats']
        self.mean = self.norm_stats['mean']
        self.std = self.norm_stats['std']

        # Instantiate processor
        self.processor = DataProcessor(
            feature_cols=FEATURE_COLS, 
            seq_len=30
        )
        print("Model and processor loaded successfully.")

    def predict(self, raw_data_df: pd.DataFrame) -> float:
        '''Takes raw data of a single engine's history and returns a RUL prediction'''

        sequence = self.processor.transform(raw_data_df)
        sequence_tensor = torch.tensor(
            sequence,
            dtype=torch.float32).unsqueeze(0).to(self.device)
        
        normalized_tensor = (sequence_tensor - self.mean) / self.std

        with torch.no_grad():
            prediction_raw = self.model(normalized_tensor)

        return prediction_raw.item()

