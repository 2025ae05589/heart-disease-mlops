"""Request logging and Prometheus metrics of the API.

Defines the six metrics that Prometheus reads from GET /metrics, and writes
one log line for every request. api/main.py calls record_request() after
each request and record_prediction() after each prediction.

Calls to /health and /metrics are not counted or logged, as Kubernetes and
Prometheus call them every few seconds.
"""
import json
import logging
import time
import uuid

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

SKIPPED_PATHS = {"/health", "/metrics"}

MAX_INPUT_LENGTH = 1000

# metrics of the service
REQUESTS = Counter(
    "api_requests_total", "Requests, by endpoint and status", ["path", "status"]
)
RESPONSE_TIME = Histogram(
    "api_request_duration_seconds", "Response time of a request", ["path"]
)
MODEL_LOADED = Gauge("api_model_loaded", "1 if the model is loaded, else 0")

# metrics of the model
PREDICTIONS = Counter(
    "api_predictions_total", "Predictions, by class", ["label"]
)
CONFIDENCE = Histogram(
    "api_prediction_confidence",
    "Confidence of the predictions",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
MISSING_FIELDS = Counter(
    "api_missing_fields_total", "Missing values filled by the model, by field", ["field"]
)

log = logging.getLogger("api")


def setup_logging():
    """Configure the "api" logger to write to the console, with the time in
    UTC."""
    if log.handlers:
        return
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S UTC"
    )
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(logging.INFO)


def new_request_id() -> str:
    """Return an 8-character random ID for a request."""
    return uuid.uuid4().hex[:8]


def input_text(values) -> str:
    """Return the input of a request as single-line JSON, truncated after
    MAX_INPUT_LENGTH characters."""
    text = json.dumps(values, separators=(",", ":"), default=str)
    if len(text) > MAX_INPUT_LENGTH:
        text = text[:MAX_INPUT_LENGTH] + "...(cut)"
    return text


def record_prediction(result: dict):
    """Update the model metrics with one prediction: its class, its
    confidence, and the missing fields that the model filled."""
    PREDICTIONS.labels(label=result["label"]).inc()
    CONFIDENCE.observe(result["confidence"])
    for field in result["filled_fields"]:
        MISSING_FIELDS.labels(field=field).inc()


def record_request(method: str, path: str, status: int, seconds: float,
                   request_id: str, details: dict, row_lines: list[str]):
    """Update the service metrics (request count and response time) with one
    completed request, and write its log line, for example

        POST /predict status=200 time_ms=4.1 request_id=7f3a9c21 prediction=1 ...

    details: items written after request_id, such as prediction=1 for a
        prediction or invalid_fields=cp for a rejected request.
    row_lines: for /predict-batch only, one line per patient, written after
        the request line.
    """
    REQUESTS.labels(path=path, status=str(status)).inc()
    RESPONSE_TIME.labels(path=path).observe(seconds)

    line = (f"{method} {path} status={status} time_ms={seconds * 1000:.1f} "
            f"request_id={request_id}")
    for key, value in details.items():
        line += f" {key}={value}"
    log.info(line)
    for row_line in row_lines:
        log.info("request_id=%s %s", request_id, row_line)


def metrics_text() -> tuple[bytes, str]:
    """Return the metrics in the Prometheus text format, and its media type."""
    return generate_latest(), CONTENT_TYPE_LATEST
