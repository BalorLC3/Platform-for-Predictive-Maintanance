import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from pathlib import Path
from src.model.predict import ModelPredictor, FEATURE_COLS
from src.api.schemas import TurbofanDataInput, PredictionResponse

# This ensures we can see what's happening in Docker/Cloud logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Turbofan Engine RUL Prediction API",
    description="API for predicting the Remaining Useful Life of a turbofan engine.",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "checkpoints/lstm_model_inference.pth"

predictor = None

try:
    if not MODEL_PATH.exists():
        logger.critical(f"Model file not found at {MODEL_PATH}")
        # We don't raise an exception here to allow the app to start (and return health checks),
        # but predictions will fail gracefully.
    else:
        logger.info(f"Loading model from {MODEL_PATH}...")
        predictor = ModelPredictor(model_path=MODEL_PATH)
        logger.info("Model and processor loaded successfully.")

except Exception as e:
    logger.critical(f"Failed to initialize model: {e}", exc_info=True)


@app.get("/", tags=['Health Check'])
def read_root():
    # This endpoint works even if the model fails to load
    return {"status": "API is running. Access to /docs to check."}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_rul(input_data: TurbofanDataInput):
    """
    Predicts the RUL based on a sequence of sensor readings.
    """
    # Check if model loaded correctly on startup
    if predictor is None:
        logger.error("Prediction attempted but model is not loaded.")
        raise HTTPException(status_code=503, detail="Model service is currently unavailable.")

    logger.info("Received prediction request.")
    
    try:
        # Create DataFrame 
        raw_df = pd.DataFrame(input_data.data, columns=FEATURE_COLS)
        
        # Make prediction
        prediction = predictor.predict(raw_df)
        
        logger.info(f"Prediction success: {prediction:.2f}")
        return PredictionResponse(rul_prediction=prediction)

    except ValueError as e:
        # Catches column mismatches or data issues
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        # Catches unexpected server errors
        logger.error(f"Internal prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred.")

@app.get("/monitor_health")
def monitor_equipment():
    ...

@app.get("/within-failure")
def get_time_within_failure():
    ...