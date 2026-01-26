r"""LSTM Model Training Script for Predictive Maintenance."""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from pathlib import Path
from dataclasses import dataclass
import sys

# Add src directory to path to import our modules
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.preprocessing.processor import DataProcessor, FEATURE_COLS
from src.model.architecture import LSTMRegressor, get_device

# CONFIGURATION 
@dataclass
class TrainConfig:
    data_path: Path
    model_output_path: Path
    seq_len: int = 30
    window_size: int = 20
    rul_clip_value: int = 130
    batch_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 300
    hidden_dim: int = 256
    num_layers: int = 2
    dropout_prob: float = 0.2

class SequenceDataset(Dataset):
    """PyTorch Dataset for loading sequence-based samples.
    
    Wraps normalized sequence data and corresponding RUL targets for use with
    PyTorch DataLoaders. Each sample consists of a fixed-length sequence of
    sensor readings and a corresponding RUL value.
    
    Attributes:
        X (torch.Tensor): Input sequences of shape (num_samples, seq_len, num_features)
        y (torch.Tensor): Target RUL values of shape (num_samples, 1)
    """
    
    def __init__(self, X, y):
        """Initialize dataset with features and targets.
        
        Args:
            X (torch.Tensor): Input sequences of shape (num_samples, seq_len, num_features)
            y (torch.Tensor): Target RUL values of shape (num_samples, 1)
        """
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        """Retrieve a single sequence and target pair.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (X_sample, y_sample) where X_sample is a sequence and y_sample is a scalar RUL
        """
        return self.X[idx], self.y[idx]

def add_rul(df: pd.DataFrame, rul_clip_value: int) -> pd.DataFrame:
    """Calculate and add Remaining Useful Life (RUL) column to dataframe.
    
    For each engine, RUL is computed as the difference between the maximum cycle
    number and the current cycle, then clipped to a maximum value to prevent
    unrealistic targets for early-life engines.
    
    Args:
        df (pd.DataFrame): Input dataframe with 'engine' and 'cycle' columns
        rul_clip_value (int): Maximum RUL value (clips upper bound)
        
    Returns:
        pd.DataFrame: Original dataframe with new 'RUL' column added
    """
    max_cycles = df.groupby('engine')['cycle'].transform('max')
    df['RUL'] = max_cycles - df['cycle']
    df['RUL'] = df['RUL'].clip(upper=rul_clip_value)
    return df

def create_sequences_per_engine(df: pd.DataFrame, processor: type) -> tuple:
    """Create fixed-length sequences from sensor data on a per-engine basis.
    
    This function processes each engine independently to prevent data leakage
    (sequences spanning multiple engines). For each engine:
    1. Applies rolling mean feature engineering
    2. Extracts all features (original + rolling mean variants)
    3. Slides a window to create overlapping sequences
    4. Associates each sequence with the RUL at the end of the sequence
    
    Args:
        df (pd.DataFrame): Training dataframe with engine, cycle, sensor, and RUL columns
        processor (DataProcessor): Processor instance with feature_cols and seq_len attributes
        
    Returns:
        tuple: (X_seq, y_seq) where
            - X_seq: Input sequences of shape (num_sequences, seq_len, num_features)
            - y_seq: Target RUL values of shape (num_sequences, 1)
    
    Note:
        This per-engine approach prevents data leakage by ensuring sequences
        do not span across different engines, maintaining temporal integrity.
    """
    X_seq_list, y_seq_list = [], []
    for engine_id in df['engine'].unique():
        engine_df = df[df['engine'] == engine_id]
        
        # We use our robust, tested processor for feature engineering
        # Note: We pass the full history, the processor will handle it.
        # This processor implementation should not use groupby('engine')
        # as it operates on a single engine's data here.
        
        # Create rolling features first, then sequences.
        rolling_features_df = processor._calculate_rolling_mean(engine_df)
        all_feature_cols = processor.feature_cols + [f"{c}_rolling_mean" for c in processor.feature_cols]
        
        X_engine = rolling_features_df[all_feature_cols].to_numpy()
        y_engine = rolling_features_df['RUL'].to_numpy()

        # Create sequences for this single engine
        if len(X_engine) >= processor.seq_len:
            for i in range(len(X_engine) - processor.seq_len + 1):
                X_seq_list.append(X_engine[i : i + processor.seq_len])
                y_seq_list.append(y_engine[i + processor.seq_len - 1])

    X_seq = torch.tensor(np.array(X_seq_list), dtype=torch.float32)
    y_seq = torch.tensor(np.array(y_seq_list), dtype=torch.float32).unsqueeze(1)
    
    return X_seq, y_seq

def train_model(
        model: nn.Module, 
        train_loader: DataLoader, 
        criterion: nn.Module, 
        optimizer: optim.Optimizer, 
        scheduler: object, 
        epochs: int, 
        device: str
    ) -> tuple:
    """Execute the main training loop with learning rate scheduling.
    
    Trains the LSTM model for the specified number of epochs, printing progress
    every 10 epochs. Uses backpropagation and the specified optimizer. The learning
    rate is adjusted each epoch using the provided scheduler (cosine annealing).
    
    Args:
        model (nn.Module): LSTM regressor model to train
        train_loader (DataLoader): DataLoader providing training batches
        criterion (nn.Module): Loss function (MSELoss expected)
        optimizer (optim.Optimizer): Optimizer for gradient updates (Adam expected)
        scheduler (object): Learning rate scheduler that adjusts LR each epoch
        epochs (int): Number of training epochs
        device (str): Device to train on ('cpu' or 'cuda')
        
    Returns:
        tuple: (trained_model, history) where
            - trained_model: Model with updated weights
            - history: List of mean training loss values per epoch
    """
    print("Starting model training...")
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        mean_loss = total_loss / len(train_loader)
        history.append(mean_loss)
        if (epoch + 1) % 10 == 0:
            print(f"> Epoch {epoch+1}/{epochs}, loss = {mean_loss:.4f}, lr = {scheduler.get_last_lr()[0]:.6f}")

    print("Training complete.")
    return model, history

#  MAIN EXECUTION 
if __name__ == "__main__":
    # Load Raw Training Data
    # Define column names: engine ID, cycle number, 3 operational settings,
    config = TrainConfig(
            data_path=project_root / "data",
            model_output_path=project_root / "checkpoints" / "lstm_model_inference.pth"
    )

    # and 21 sensor measurements
    col_names = ['engine', 'cycle'] + [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
    df_train_raw = pd.read_csv(
        config.data_path / "train_FD001.txt",
        sep=r'\s+', header=None, names=col_names
    )

    # Preprocess Data - Add RUL Target Variable
    # RUL is computed per engine and clipped to prevent unrealistic values
    df_train_rul = add_rul(df_train_raw, config.rul_clip_value)
    
    # Feature Engineering and Sequence Creation
    # Initialize processor with feature columns and sequence parameters
    processor = DataProcessor(
        feature_cols=FEATURE_COLS,
        window_size=config.window_size,
        seq_len=config.seq_len
    )
    
    # Create fixed-length sequences on per-engine basis to prevent data leakage
    # This ensures sequences do not span multiple engines
    X_seq, y_seq = create_sequences_per_engine(df_train_rul, processor)
    print(f"Created sequences. X shape: {X_seq.shape}, y shape: {y_seq.shape}")

    # Split data 80/20 for training and validation
    # Normalization is computed on training set to prevent validation leakage
    X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(
        X_seq, 
        y_seq, 
        test_size=0.2, 
        shuffle=True, 
        random_state=17
    )
    
    mean = X_train_seq.mean(dim=(0, 1), keepdim=True)
    std = X_train_seq.std(dim=(0, 1), keepdim=True) + 1e-8
    
    X_train_norm = (X_train_seq - mean) / std
    X_val_norm = (X_val_seq - mean) / std

    # Wrap normalized training data in PyTorch Dataset for batch loading
    train_ds = SequenceDataset(X_train_norm, y_train_seq)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
    
    # Determine device (GPU if available, else CPU) and input dimension
    device = get_device()
    input_dim = X_train_norm.shape[2]
    print(f"Device: {device}, Input Dimension: {input_dim}")
    
    # Instantiate LSTM model with configured architecture
    model = LSTMRegressor(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout_prob=config.dropout_prob
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)  # Adam optimizer
    
    # Cosine annealing learning rate scheduler: gradually decreases LR from initial
    # value to near zero over training, then restarts. Helps escape local minima.
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # ========================================================================
    # Train the Model
    # ========================================================================
    trained_model, history = train_model(
        model, 
        train_loader, 
        criterion, 
        optimizer, 
        scheduler, 
        config.epochs, 
        device
    )
    
    # ========================================================================
    # Save Model Checkpoint with Normalization Statistics
    # ========================================================================
    # Save both model weights and normalization parameters for inference consistency
    checkpoint = {
        'model_state_dict': trained_model.state_dict(),  # Trained model weights
        'input_dim': input_dim,                          # Input feature dimension
        'hidden_dim': config.hidden_dim,             # Hidden layer dimension
        'num_layers': config.num_layers,             # Number of LSTM layers
        'dropout_prob': config.dropout_prob,         # Dropout probability
        'normalization_stats': {
            'mean': mean.cpu(),  # Feature means (move to CPU for storage)
            'std': std.cpu()     # Feature standard deviations (move to CPU for storage)
        }
    }
    
    # Create output directory if it doesn't exist, then save checkpoint
    config.model_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, config.model_output_path)
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Model checkpoint saved to: {config.model_output_path}")
    print(f"{'='*70}")