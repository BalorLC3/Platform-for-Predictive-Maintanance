import time
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pathlib import Path
from contextlib import asynccontextmanager
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from src.model.predict import ModelPredictor, FEATURE_COLS
from src.api.schemas import TurbofanDataInput, PredictionResponse
from src.monitoring.state import set_latest_rul, get_latest_rul
from src.monitoring.metrics import (
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_LATENCY,
    PREDICTION_ERRORS_TOTAL,
    PREDICTIONS_TOTAL
)

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("rul-api")

class MetricsLogFilter(logging.Filter):
    def filter(self, record):
        return "/metrics" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(MetricsLogFilter())

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "checkpoints" / "lstm_model_inference.pth"

predictor: ModelPredictor | None = None
model_loaded = False

# ---------------- App lifecycle ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, model_loaded
    logger.info("Starting application")

    try:
        if MODEL_PATH.exists():
            predictor = ModelPredictor(model_path=MODEL_PATH)
            model_loaded = True
            logger.info("Model loaded successfully")
        else:
            logger.critical(f"Model not found at {MODEL_PATH}")
    except Exception as e:
        logger.critical("Model initialization failed", exc_info=True)

    yield
    logger.info("Shutting down application")

app = FastAPI(
    title="Turbofan Engine RUL Prediction API",
    description="Predict Remaining Useful Life (RUL) for turbofan engines",
    version="1.0.0",
    lifespan=lifespan
)

# ---------------- Middleware  ----------------
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    HTTP_REQUESTS_TOTAL.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    HTTP_REQUEST_LATENCY.labels(
        endpoint=request.url.path
    ).observe(duration)

    return response

@app.middleware("http")
async def request_logger(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    if request.url.path not in ("/metrics", "/health"):
        logger.info(
            f"{request.method} {request.url.path} "
            f"{response.status_code} {duration:.3f}s"
        )
    return response

# ---------------- Routes ----------------
@app.get("/", tags=["Health"])
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded
    }

@app.post("/predict")
def predict_rul(input_data: TurbofanDataInput):
    if predictor is None:
        PREDICTION_ERRORS_TOTAL.inc()
        raise HTTPException(503, "Model unavailable")

    try:
        raw_df = pd.DataFrame(input_data.data, columns=FEATURE_COLS)
        prediction = predictor.predict(raw_df)

        set_latest_rul(prediction)
        PREDICTIONS_TOTAL.inc()

        return PredictionResponse(rul_prediction=prediction)

    except Exception:
        PREDICTION_ERRORS_TOTAL.inc()
        raise

@app.get("/monitor_health")
def monitor_equipment():
    rul = get_latest_rul()

    if rul is None:
        raise HTTPException(503, "No predictions available")

    status = (
        "healthy" if rul > 100
        else "degrading" if rul > 30
        else "critical"
    )

    return {
        "health_status": status,
        "rul_estimate": rul
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
