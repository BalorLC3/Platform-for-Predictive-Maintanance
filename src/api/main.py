import time
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pathlib import Path
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from src.model.predict import ModelPredictor, FEATURE_COLS
from src.api.schemas import TurbofanDataInput, PredictionResponse
from src.monitoring.state import (
    set_latest_rul,
    get_latest_rul
)
from src.monitoring.metrics import (
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_LATENCY,
    PREDICTION_ERRORS_TOTAL,
    PREDICTIONS_TOTAL
)

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

model_loaded = False

try:
    if MODEL_PATH.exists():
        predictor = ModelPredictor(model_path=MODEL_PATH)
        model_loaded = True
        logger.info("Model loaded successfully")
    else:
        logger.critical(f"Model file not found at {MODEL_PATH}")

except Exception as e:
    logger.critical(f"Failed to initialize model: {e}", exc_info=True)





@app.get("/", tags=['Health Check'])
def read_root():
    # This endpoint works even if the model fails to load
    return {"status": "API is running. Access to /docs to check."}

@app.post("/predict")
def predict_rul(input_data: TurbofanDataInput):
    if predictor is None:
        PREDICTION_ERRORS_TOTAL.inc()
        raise HTTPException(status_code=503, detail="Model unavailable")

    try:
        raw_df = pd.DataFrame(input_data.data, columns=FEATURE_COLS)
        prediction = predictor.predict(raw_df)  

        # Global declaration
        set_latest_rul(prediction)
        PREDICTIONS_TOTAL.inc()

        return PredictionResponse(rul_prediction=prediction)

    except ValueError:
        PREDICTION_ERRORS_TOTAL.inc()
        raise

    except Exception:
        PREDICTION_ERRORS_TOTAL.inc()
        raise

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time

    HTTP_REQUESTS_TOTAL.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    HTTP_REQUEST_LATENCY.labels(
        endpoint=request.url.path
    ).observe(process_time)

    return response

@app.get("/health")
def health():
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded
    }


@app.get("/monitor_health")
def monitor_equipment():
    """
    High-level health status derived from recent RUL predictions
    """
    rul = get_latest_rul()  # cached / stored / last inference

    if rul is None:
        raise HTTPException(503, "No predictions available")

    if rul > 100:
        status = "healthy"
    elif 30 < rul <= 100:
        status = "degrading"
    else:
        status = "critical"

    return {
        "health_status": status,
        "rul_estimate": rul
    }
