"""Tests for the API in api/: request validation, endpoint responses and
loading of the exported model.

The tests use FastAPI's test client and a stub model with fixed probabilities.
"""
import json

import mlflow.sklearn
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from api import main
from api.inference import LoadedModel, load_model, predict
from api.schemas import EXAMPLE_PATIENT, MAX_BATCH_SIZE, Patient
from src.config import FEATURES, RANDOM_STATE, ROOT
from src.data.preparation import split_data
from src.features.preprocessing import make_pipeline
from src.tracking.experiments import TRUSTED_TYPES, model_requirements


class StubModel:
    """Stub for the trained model. It returns the configured probabilities of
    disease, one for each patient, and records the input data it receives."""

    def __init__(self):
        self.probabilities = [0.87]
        self.received = None

    def predict_proba(self, data):
        self.received = data
        disease = np.array(self.probabilities[:len(data)])
        return np.column_stack([1 - disease, disease])


def patient(**changes):
    """The example patient, with some values changed."""
    values = dict(EXAMPLE_PATIENT)
    values.update(changes)
    return values


@pytest.fixture
def stub_model():
    return StubModel()
    

@pytest.fixture
def client(monkeypatch, stub_model):
    """Test client of the API, with the stub model loaded."""
    loaded = LoadedModel(
        model=stub_model,
        name="heart-disease-classifier",
        version="7",
        run_id="abc123",
        trained_at="2026-10-12 18:19:42 UTC",
    )
    monkeypatch.setattr(main, "load_model", lambda: loaded)
    with TestClient(main.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "heart-disease-classifier",
        "model_version": "7",
        "run_id": "abc123",
        "trained_at": "2026-10-12 18:19:42 UTC",
    }


def test_prediction_of_disease(client, stub_model):
    stub_model.probabilities = [0.87]

    # oldpeak can be 0, unlike the other numerical features
    response = client.post("/predict", json=patient(oldpeak=0))

    assert response.status_code == 200
    assert response.json() == {
        "prediction": 1,
        "label": "disease",
        "confidence": 0.87,
        "filled_fields": [],
        "model_version": "7",
    }


def test_confidence_is_for_the_predicted_class(client, stub_model):
    # a disease probability of 0.20 gives a confidence of 0.80 for "no disease"
    stub_model.probabilities = [0.20]

    result = client.post("/predict", json=patient()).json()

    assert result["prediction"] == 0
    assert result["label"] == "no disease"
    assert result["confidence"] == 0.80


def test_probability_at_the_cut_off_is_predicted_as_disease(client, stub_model):
    stub_model.probabilities = [0.5]

    result = client.post("/predict", json=patient()).json()

    assert result["prediction"] == 1
    assert result["confidence"] == 0.5


def test_features_reach_the_model_in_the_training_order(client, stub_model):
    client.post("/predict", json=patient())

    assert list(stub_model.received.columns) == FEATURES


@pytest.mark.parametrize("feature", ["ca", "thal"])
def test_missing_value_is_passed_to_the_model_and_reported(client, stub_model, feature):
    response = client.post("/predict", json=patient(**{feature: None}))

    assert response.status_code == 200
    assert response.json()["filled_fields"] == [feature]
    assert stub_model.received[feature].isna().all()


@pytest.mark.parametrize("changes, field", [
    ({"age": None}, "age"),             # only "ca" and "thal" may be null
    ({"cp": 9}, "cp"),                  # not a valid code
    ({"sex": 0.5}, "sex"),              # not a valid code
    ({"age": 0}, "age"),                # has to be above 0
    ({"chol": -5}, "chol"),
    ({"oldpeak": -1}, "oldpeak"),       # can be 0, but not below
    ({"age": "fifty"}, "age"),          # not a number
    ({"age": "54"}, "age"),             # a number written as text
    ({"fbs": True}, "fbs"),             # true is not taken as 1
    ({"cholesterol": 250}, "cholesterol"),   # not a known field
])
def test_invalid_request_is_rejected(client, changes, field):
    response = client.post("/predict", json=patient(**changes))

    assert response.status_code == 422
    # the response identifies the invalid field
    assert response.json()["detail"][0]["loc"] == ["body", field]


def test_request_with_a_missing_field_is_rejected(client):
    values = patient()
    del values["chol"]

    response = client.post("/predict", json=values)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "chol"]


def test_batch_gives_a_result_for_each_row(client, stub_model):
    # the stub model receives only the two valid rows
    stub_model.probabilities = [0.87, 0.20]
    rows = [patient(), patient(cp=9), patient(ca=None)]

    response = client.post("/predict-batch", json={"patients": rows})

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {"row": 1, "status": "ok", "prediction": 1, "label": "disease",
             "confidence": 0.87, "filled_fields": []},
            {"row": 2, "status": "error",
             "errors": ["cp: Value error, should be one of 1, 2, 3, 4"]},
            {"row": 3, "status": "ok", "prediction": 0, "label": "no disease",
             "confidence": 0.80, "filled_fields": ["ca"]},
        ],
        "count": 3,
        "predicted": 2,
        "failed": 1,
        "model_version": "7",
    }


@pytest.mark.parametrize("body", [{"patients": []}, {"patients": "all"}, {}])
def test_batch_that_is_not_a_list_of_patients_is_rejected(client, body):
    response = client.post("/predict-batch", json=body)

    assert response.status_code == 422


def test_batch_size_limit(client, stub_model):
    stub_model.probabilities = [0.87] * MAX_BATCH_SIZE

    at_limit = client.post("/predict-batch", json={"patients": [patient()] * MAX_BATCH_SIZE})
    above = client.post("/predict-batch", json={"patients": [patient()] * (MAX_BATCH_SIZE + 1)})

    assert at_limit.status_code == 200
    assert above.status_code == 422


def test_service_without_a_model(monkeypatch):
    def failing_load():
        raise FileNotFoundError("no model folder")

    monkeypatch.setattr(main, "load_model", failing_load)
    with TestClient(main.app) as test_client:
        health = test_client.get("/health")
        prediction = test_client.post("/predict", json=patient())
        info = test_client.get("/model-info")

    assert health.status_code == 503
    assert health.json() == {"status": "unavailable", "model_loaded": False}
    assert prediction.status_code == 503
    assert info.status_code == 503


def test_sample_request_file_is_a_valid_patient():
    sample = json.loads((ROOT / "api" / "sample_request.json").read_text())

    assert Patient.model_validate(sample).model_dump() == EXAMPLE_PATIENT


def test_exported_model_folder_is_loaded(tmp_path):
    # a small model is saved in the same format as the exported champion
    X_train, _, y_train, _ = split_data()
    pipeline = make_pipeline(LogisticRegression(random_state=RANDOM_STATE)).fit(X_train, y_train)
    model_dir = tmp_path / "champion"
    mlflow.sklearn.save_model(
        pipeline, str(model_dir),
        pip_requirements=model_requirements(pipeline), skops_trusted_types=TRUSTED_TYPES,
    )
    (model_dir / "registered_model_meta").write_text(
        "model_name: heart-disease-classifier\nmodel_version: 3\n"
    )
    # a model saved outside an MLflow run has no run ID; the champion has one
    details = yaml.safe_load((model_dir / "MLmodel").read_text())
    details["run_id"] = "run-of-the-test"
    (model_dir / "MLmodel").write_text(yaml.safe_dump(details))

    loaded = load_model(model_dir)
    results = predict(loaded.model, [patient(), patient(ca=None, thal=None)])

    assert loaded.name == "heart-disease-classifier"
    assert loaded.version == "3"
    assert loaded.run_id == "run-of-the-test"
    assert len(loaded.trained_at) == len("2026-10-12 18:19:42 UTC")
    assert loaded.trained_at.endswith(" UTC")
    assert [result["filled_fields"] for result in results] == [[], ["ca", "thal"]]
    for result in results:
        assert result["prediction"] in (0, 1)
        assert 0.5 <= result["confidence"] <= 1
