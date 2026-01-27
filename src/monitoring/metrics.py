from prometheus_client import Counter, Histogram

# Total HTTP requests
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

# Request latency
HTTP_REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency",
    ["endpoint"]
)

# ML-specific metrics
PREDICTIONS_TOTAL = Counter(
    "model_predictions_total",
    "Total number of model predictions"
)

PREDICTION_ERRORS_TOTAL = Counter(
    "model_prediction_errors_total",
    "Total number of prediction errors"
)
