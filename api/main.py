"""The prediction API: four endpoints, and /metrics for Prometheus.

    POST /predict        prediction for one patient
    POST /predict-batch  predictions for a list of patients
    GET  /health         whether the service is ready
    GET  /model-info     which model is being served
    GET  /metrics        metrics of the requests and predictions

Related modules:
    api/schemas.py      request and response formats, with input validation
    api/inference.py    loading of the exported model, and prediction
    api/monitoring.py   Prometheus metrics and request logging

The model is loaded once, at start-up.

Run from the project root:
    uvicorn api.main:app --port 8000 --no-access-log
The API can then be tried from http://127.0.0.1:8000/docs
"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from api import monitoring
from api.inference import load_model, predict
from api.monitoring import log
from api.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ModelInfoResponse,
    Patient,
    PredictResponse,
)

monitoring.setup_logging()

# holds the loaded model under the key "loaded"; empty if loading failed
state = {}


@asynccontextmanager
async def lifespan(app):
    """Load the model when the service starts. If it cannot be loaded, the
    service still starts, and /health reports that it is not ready."""
    try:
        state["loaded"] = load_model()
        monitoring.MODEL_LOADED.set(1)
        log.info("Loaded %s version %s", state["loaded"].name, state["loaded"].version)
    except Exception as error:
        monitoring.MODEL_LOADED.set(0)
        log.error("The model could not be loaded: %r", error)
    yield
    state.clear()
    monitoring.MODEL_LOADED.set(0)


app = FastAPI(title="Heart Disease Prediction API", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def monitor(request: Request, call_next):
    """Middleware applied to every request. Assigns a request ID, returned in
    the X-Request-ID response header. After the response, updates the
    metrics and writes the log line."""
    request.state.request_id = monitoring.new_request_id()
    request.state.details = {}
    request.state.row_lines = []
    start = time.perf_counter()

    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request.state.request_id
        return response
    finally:
        path = request.url.path
        # all unknown endpoints share the label "other", so that random URLs
        # do not create new metric series
        if path not in {route.path for route in app.routes}:
            path = "other"
        if path not in monitoring.SKIPPED_PATHS:
            monitoring.record_request(
                request.method,
                path,
                status,
                time.perf_counter() - start,
                request.state.request_id,
                request.state.details,
                request.state.row_lines,
            )


@app.exception_handler(RequestValidationError)
async def refused(request: Request, error: RequestValidationError):
    """Handler for a request that fails validation. Adds the invalid fields
    and the input as received to its log line, then returns FastAPI's
    standard 422 response."""
    fields = []
    for problem in error.errors():
        fields.append(".".join(str(part) for part in problem["loc"][1:]) or "body")
    request.state.details["invalid_fields"] = ",".join(fields)
    request.state.details["input"] = monitoring.input_text(error.body)
    return await request_validation_exception_handler(request, error)


def get_loaded():
    """Return the loaded model, or refuse the request with status 503 if no
    model is loaded."""
    if "loaded" not in state:
        raise HTTPException(status_code=503, detail="The model is not loaded")
    return state["loaded"]


def row_errors(error: ValidationError) -> list[str]:
    """Return the problems of one patient as short lines, for example
    "cp: Value error, should be one of 1, 2, 3, 4"."""
    lines = []
    for problem in error.errors():
        field = ".".join(str(part) for part in problem["loc"]) or "row"
        lines.append(f"{field}: {problem['msg']}")
    return lines


@app.get("/health", response_model=HealthResponse)
def health():
    """Status 200 when the model is loaded, and 503 when it is not."""
    if "loaded" in state:
        return {"status": "ok", "model_loaded": True}
    return JSONResponse(
        status_code=503, content={"status": "unavailable", "model_loaded": False}
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """The model being served, as registered in MLflow."""
    loaded = get_loaded()
    return {
        "model_name": loaded.name,
        "model_version": loaded.version,
        "run_id": loaded.run_id,
        "trained_at": loaded.trained_at,
    }


@app.get("/metrics", include_in_schema=False)
def metrics():
    """Current values of the metrics, in the Prometheus text format."""
    content, media_type = monitoring.metrics_text()
    return Response(content=content, media_type=media_type)


@app.post("/predict", response_model=PredictResponse)
def predict_one(patient: Patient, request: Request):
    """Prediction and confidence for one patient."""
    loaded = get_loaded()
    values = patient.model_dump()
    result = predict(loaded.model, [values])[0]
    result["model_version"] = loaded.version

    monitoring.record_prediction(result)
    request.state.details.update(
        model_version=loaded.version,
        prediction=result["prediction"],
        confidence=result["confidence"],
        input=monitoring.input_text(values),
    )
    return result


@app.post("/predict-batch", response_model=BatchResponse, response_model_exclude_none=True)
def predict_batch(batch: BatchRequest, request: Request):
    """Predictions for a list of patients. A row with a wrong value gets its
    errors in the reply, and the other rows are still predicted."""
    loaded = get_loaded()

    # each row is checked by the same rules as in /predict
    results = []
    valid_rows = []
    for number, row in enumerate(batch.patients, start=1):
        try:
            patient = Patient.model_validate(row)
        except ValidationError as error:
            results.append({"row": number, "status": "error", "errors": row_errors(error)})
        else:
            results.append({"row": number, "status": "ok"})
            valid_rows.append((len(results) - 1, patient.model_dump()))

    if valid_rows:
        predictions = predict(loaded.model, [patient for _, patient in valid_rows])
        for (position, _), prediction in zip(valid_rows, predictions):
            results[position].update(prediction)
            monitoring.record_prediction(prediction)

    # one log line per row: the validated values for a valid row, and the
    # input as received for an invalid row
    checked = dict(valid_rows)
    for position, result in enumerate(results):
        line = f"row={result['row']} status={result['status']}"
        if result["status"] == "ok":
            line += f" prediction={result['prediction']} confidence={result['confidence']}"
            values = checked[position]
        else:
            fields = [error.split(":")[0] for error in result["errors"]]
            line += f" invalid_fields={','.join(fields)}"
            values = batch.patients[position]
        request.state.row_lines.append(f"{line} input={monitoring.input_text(values)}")

    request.state.details.update(
        model_version=loaded.version,
        count=len(results),
        predicted=len(valid_rows),
        failed=len(results) - len(valid_rows),
    )
    return {
        "results": results,
        "count": len(results),
        "predicted": len(valid_rows),
        "failed": len(results) - len(valid_rows),
        "model_version": loaded.version,
    }
