from pydantic import BaseModel, ConfigDict 
from typing import List
import numpy as np

np.random.seed(17)

class TurbofanDataInput(BaseModel):
    data: List[List[float]]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "data": 
                    # A valid example must have at least 30 rows and exactly 14 columns per row
                    np.round(np.random.randn(30, 14), 2).tolist()
                
            }
        }
    )

class PredictionResponse(BaseModel):
    rul_prediction: float