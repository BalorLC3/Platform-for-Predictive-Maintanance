import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

# Add src directory to path to import our modules
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.preprocessing.processor import DataProcessor, FEATURE_COLS
from src.model.architecture import LSTMRegressor, get_device

# CONFIGURATION 
CONFIG = {
    "data_path": project_root / "data" / "CMaps", 
    "model_output_path": project_root / "notebooks" / "lstm_model_inference.pth",
    "seq_len": 30,
    "window_size": 20,
    "rul_clip_value": 130,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "epochs": 200,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout_prob": 0.2
}

#  DATASET CLASS 
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#  DATA PREPARATION 
def add_rul(df, rul_clip_value):
    """Adds the Remaining Useful Life (RUL) column to a dataframe."""
    max_cycles = df.groupby('engine')['cycle'].transform('max')
    df['RUL'] = max_cycles - df['cycle']
    df['RUL'] = df['RUL'].clip(upper=rul_clip_value)
    return df

def create_sequences_per_engine(df, processor):
    """
    Correctly creates sequences on a per-engine basis.
    This prevents creating sequences that span across two different engines.
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

# TRAINING LOOP 
def train_model(model, train_loader, criterion, optimizer, epochs, device):
    """Main training loop."""
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
        
        mean_loss = total_loss / len(train_loader)
        history.append(mean_loss)
        if (epoch + 1) % 10 == 0:
            print(f"> Epoch {epoch+1}/{epochs}, Loss: {mean_loss:.4f}")
    print("Training complete.")
    return model, history

#  MAIN EXECUTION 
if __name__ == "__main__":
    # Load Data
    col_names = ['engine', 'cycle'] + [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
    df_train_raw = pd.read_csv(
        CONFIG["data_path"] / "train_FD001.txt",
        sep=r'\s+', header=None, names=col_names
    )

    # Preprocess Data
    df_train_rul = add_rul(df_train_raw, CONFIG["rul_clip_value"])
    
    # Initialize our standard processor
    processor = DataProcessor(
        feature_cols=FEATURE_COLS,
        window_size=CONFIG["window_size"],
        seq_len=CONFIG["seq_len"]
    )
    
    # Create sequences CORRECTLY
    X_seq, y_seq = create_sequences_per_engine(df_train_rul, processor)
    print(f"Created sequences. X shape: {X_seq.shape}, y shape: {y_seq.shape}")

    # 3. Split and Normalize
    X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=True, random_state=17)
    
    mean = X_train_seq.mean(dim=(0, 1), keepdim=True)
    std = X_train_seq.std(dim=(0, 1), keepdim=True) + 1e-8
    
    X_train_norm = (X_train_seq - mean) / std
    X_val_norm = (X_val_seq - mean) / std

    # Create DataLoaders
    train_ds = SequenceDataset(X_train_norm, y_train_seq)
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True)
    
    # Initialize and Train Model
    device = get_device()
    input_dim = X_train_norm.shape[2]
    print(f"Device: {device}, Input Dimension: {input_dim}")
    
    model = LSTMRegressor(
        input_dim=input_dim,
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["num_layers"],
        dropout_prob=CONFIG["dropout_prob"]
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    
    trained_model, history = train_model(model, train_loader, criterion, optimizer, CONFIG["epochs"], device)
    
    # Save the Checkpoint for the API
    checkpoint = {
        'model_state_dict': trained_model.state_dict(),
        'input_dim': input_dim,
        'hidden_dim': CONFIG["hidden_dim"],
        'num_layers': CONFIG["num_layers"],
        'dropout_prob': CONFIG["dropout_prob"],
        'normalization_stats': {
            'mean': mean.cpu(), # Save stats on CPU
            'std': std.cpu()
        }
    }
    
    # Ensure the output directory exists
    CONFIG["model_output_path"].parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, CONFIG["model_output_path"])
    print(f"Model checkpoint saved to: {CONFIG['model_output_path']}")