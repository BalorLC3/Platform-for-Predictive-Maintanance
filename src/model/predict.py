import torch
import pandas as pd
from pathlib import Path
from src.preprocessing.processor import DataProcessor, FEATURE_COLS
from src.model.architecture import get_device, LSTMRegressor
import logging
logger = logging.getLogger(__name__)


class ModelPredictor:
    def __init__(
            self, 
            model_path: Path,
            device: torch.device | str = "auto"
    ):
        self.model_path = model_path
        self.device = get_device(device)
        # Load entire checkpoint
        try:
            checkpoint = torch.load(
                self.model_path, 
                map_location=self.device, 
                weights_only=True
                # Load weights only to avoid executing arbitrary code from checkpoint
            )
        except Exception as e:
            logger.critical("Failed to load model checkpoint", exc_info=True)
            raise RuntimeError("Model initialization failed") from e
        
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
            feature_cols=FEATURE_COLS
        )

    def predict(self, raw_data_df: pd.DataFrame) -> float:
        '''Takes raw data of a single engine's history and returns a RUL prediction'''

        if raw_data_df.empty:
            raise ValueError("Input data is empty")
        # Despite of using schemas and previous validation, this is added as fallback
        if list(raw_data_df.columns) != FEATURE_COLS:
            raise ValueError("Input features do not match expected schema")
        
        sequence = self.processor.transform(raw_data_df)
        sequence_tensor = torch.tensor(
            sequence,
            dtype=torch.float32).unsqueeze(0).to(self.device)
        
        normalized_tensor = (sequence_tensor - self.mean) / self.std

        with torch.no_grad():
            prediction_raw = self.model(normalized_tensor)

        return prediction_raw.item()

