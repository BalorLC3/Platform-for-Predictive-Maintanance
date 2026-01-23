import pandas as pd
import numpy as np
import torch
from typing import List

FEATURE_COLS = [
    'sensor_2',
    'sensor_3',
    'sensor_4',
    'sensor_7',
    'sensor_8',
    'sensor_9',
    'sensor_11',
    'sensor_12',
    'sensor_13',
    'sensor_14',
    'sensor_15',
    'sensor_17',
    'sensor_20',
    'sensor_21',
]

class DataProcessor:
    def __init__(
            self,
            feature_cols: List[str],
            window_size: int = 5, 
            seq_len: int = 30,
    ):
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.seq_len = seq_len

    def _calculate_rolling_mean(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Calculates rolling features'''
        df_out = df.copy()

        if self.window_size > 1:
            for col in self.feature_cols:
                df_out[f"{col}_rolling_mean"] = df_out[col].rolling(
                    window=self.window_size, min_periods=1
                ).mean()

        # Security fill
        df_out.bfill(inplace=True)
        df_out.ffill(inplace=True)
        return df_out
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        # Work only with useful features
        df = df[self.feature_cols].copy()
        
        if df.shape[0] < self.seq_len:
            raise ValueError(f"Input data must have at least {self.seq_len} rows (sequence length)")
        
        df_featured = self._calculate_rolling_mean(df)
        

        # all feature columns (raw + rolling), the model's input_dim must
        # match the number of columns here
        if self.window_size > 1:
            all_feature_cols = self.feature_cols + [f"{c}_rolling_mean" for c in self.feature_cols] 
        else:
            all_feature_cols = self.feature_cols

        # Take last SEQ_LEN rows to form the sequence
        sequence = df_featured[all_feature_cols].tail(self.seq_len).to_numpy()

        return sequence
        
