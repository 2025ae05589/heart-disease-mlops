"""The prediction API: the four addresses.

    POST /predict        prediction for one patient
    POST /predict-batch  predictions for a list of patients
    GET  /health         whether the service is ready
    GET  /model-info     which model is being served

The checks on a request are in api/schemas.py and the model is handled in
api/inference.py. The model is loaded once, when the service starts.

Run from the project root:
    uvicorn api.main:app --port 8000
The API can then be tried from http://127.0.0.1:8000/docs
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from api.inference import load_model, predict
from api.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ModelInfoResponse,
    Patient,
    PredictResponse,
)

log = logging.getLogger("api")

# holds the loaded model under the key "loaded"; empty if loading failed
state = {}


@asynccontextmanager
async def lifespan(app):
    """Load the model when the service starts. If it cannot be loaded, the
    service still starts, and /health reports that it is not ready."""
    try:
        state["loaded"] = load_model()
        log.info("Loaded %s version %s", state["loaded"].name, state["loaded"].version)
    except Exception as error:
        log.error("The model could not be loaded: %r", error)
    yield
    state.clear()


app = FastAPI(title="Heart Disease Prediction API", version="1.0", lifespan=lifespan)


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


@app.post("/predict", response_model=PredictResponse)
def predict_one(patient: Patient):
    """Prediction and confidence for one patient."""
    loaded = get_loaded()
    result = predict(loaded.model, [patient.model_dump()])[0]
    result["model_version"] = loaded.version
    return result


@app.post("/predict-batch", response_model=BatchResponse, response_model_exclude_none=True)
def predict_batch(batch: BatchRequest):
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

    return {
        "results": results,
        "count": len(results),
        "predicted": len(valid_rows),
        "failed": len(results) - len(valid_rows),
        "model_version": loaded.version,
    }
