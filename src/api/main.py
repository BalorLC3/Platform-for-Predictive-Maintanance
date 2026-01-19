from fastapi import FastAPI, HTTPException
import pandas as pd
from pathlib import Path
from src.model.predict import ModelPredictor, FEATURE_COLS
from src.api.schemas import TurbofanDataInput, PredictionResponse

app = FastAPI(
    title="Turbofan Engine RUL Prediction API",
    description="API for predicting the Remaining Useful Life of a turbofan engine.",
    version="1.0.0"
)

# load once when API starts
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "notebooks/lstm_model_inference.pth"
predictor = ModelPredictor(model_path=MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

@app.get("/", tags=['Health Check'])
def read_root():
    return {"status": "API is running."}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_rul(input_data: TurbofanDataInput):
    """
    Predicts the RUL based on a sequence of sensor readings.
    
    - **input_data**: A list of lists, where each inner list represents a cycle's
      sensor readings in the correct order.
    """
    try:
        raw_df = pd.DataFrame(input_data.data, columns=FEATURE_COLS)
        prediction = predictor.predict(raw_df)
    
        return PredictionResponse(rul_prediction=prediction)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
    


