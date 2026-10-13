"""Tests for the API in api/: the checks on a request, the replies of the
four addresses and the loading of the exported model.

The API is called through FastAPI's test client, so no server is started.
The model is replaced by a fake model that gives a probability fixed in the
test. In this way the reply can be checked exactly. Only the last test uses a
real model, to check that an exported model folder can be loaded.
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
from api.schemas import EXAMPLE_PATIENT, Patient
from src.config import FEATURES, RANDOM_STATE, ROOT
from src.data.preparation import split_data
from src.features.preprocessing import make_pipeline
from src.tracking.experiments import TRUSTED_TYPES, model_requirements


class FakeModel:
    """Stands in for the trained model. It returns the probabilities of
    disease given to it, one for each patient, and keeps the data it was
    asked to predict for."""

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
def fake_model():
    return FakeModel()


@pytest.fixture
def client(monkeypatch, fake_model):
    """The API with the fake model loaded."""
    loaded = LoadedModel(
        model=fake_model,
        name="heart-disease-classifier",
        version="7",
        run_id="abc123",
        trained_at="2026-10-12 18:19:42 UTC",
    )
    monkeypatch.setattr(main, "load_model", lambda: loaded)
    with TestClient(main.app) as test_client:
        yield test_client


def test_health(client):
    reply = client.get("/health")

    assert reply.status_code == 200
    assert reply.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    reply = client.get("/model-info")

    assert reply.status_code == 200
    assert reply.json() == {
        "model_name": "heart-disease-classifier",
        "model_version": "7",
        "run_id": "abc123",
        "trained_at": "2026-10-12 18:19:42 UTC",
    }


def test_prediction_of_disease(client, fake_model):
    fake_model.probabilities = [0.87]

    reply = client.post("/predict", json=patient())

    assert reply.status_code == 200
    assert reply.json() == {
        "prediction": 1,
        "label": "disease",
        "confidence": 0.87,
        "filled_fields": [],
        "model_version": "7",
    }


def test_confidence_is_for_the_predicted_class(client, fake_model):
    # 20% chance of disease means 80% confidence in "no disease"
    fake_model.probabilities = [0.20]

    result = client.post("/predict", json=patient()).json()

    assert result["prediction"] == 0
    assert result["label"] == "no disease"
    assert result["confidence"] == 0.80


def test_probability_at_the_cut_off_is_predicted_as_disease(client, fake_model):
    fake_model.probabilities = [0.5]

    result = client.post("/predict", json=patient()).json()

    assert result["prediction"] == 1
    assert result["confidence"] == 0.5


def test_features_reach_the_model_in_the_training_order(client, fake_model):
    client.post("/predict", json=patient())

    assert list(fake_model.received.columns) == FEATURES


@pytest.mark.parametrize("feature", ["ca", "thal"])
def test_missing_value_is_passed_to_the_model_and_reported(client, fake_model, feature):
    reply = client.post("/predict", json=patient(**{feature: None}))

    assert reply.status_code == 200
    assert reply.json()["filled_fields"] == [feature]
    assert fake_model.received[feature].isna().all()


def test_zero_is_allowed_for_oldpeak(client):
    reply = client.post("/predict", json=patient(oldpeak=0))

    assert reply.status_code == 200


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
def test_wrong_request_is_refused(client, changes, field):
    reply = client.post("/predict", json=patient(**changes))

    assert reply.status_code == 422
    # the reply names the field that is wrong
    assert reply.json()["detail"][0]["loc"] == ["body", field]


def test_request_with_a_field_left_out_is_refused(client):
    values = patient()
    del values["chol"]

    reply = client.post("/predict", json=values)

    assert reply.status_code == 422
    assert reply.json()["detail"][0]["loc"] == ["body", "chol"]


def test_batch_gives_a_result_for_each_row(client, fake_model):
    # the fake model is asked only for the two rows that are valid
    fake_model.probabilities = [0.87, 0.20]
    rows = [patient(), patient(cp=9), patient(ca=None)]

    reply = client.post("/predict-batch", json={"patients": rows})

    assert reply.status_code == 200
    assert reply.json() == {
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


def test_batch_with_only_wrong_rows(client):
    reply = client.post("/predict-batch", json={"patients": [patient(age=0), "not a patient"]})

    assert reply.status_code == 200
    result = reply.json()
    assert result["predicted"] == 0
    assert result["failed"] == 2
    assert [row["status"] for row in result["results"]] == ["error", "error"]


@pytest.mark.parametrize("body", [{"patients": []}, {"patients": "all"}, {}])
def test_batch_that_is_not_a_list_of_patients_is_refused(client, body):
    reply = client.post("/predict-batch", json=body)

    assert reply.status_code == 422


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
    # a small model is saved in the same form as the exported champion
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
