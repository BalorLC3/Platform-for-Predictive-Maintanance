import torch
import torch.nn as nn
import numpy as np

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