"""Tests for the request logging and metrics in api/monitoring.py: the
/metrics endpoint, the request ID and the log line of each request.

Metric values persist across tests, so each test compares a metric before
and after its own request.
"""
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from api import main, monitoring
from api.inference import LoadedModel
from api.schemas import EXAMPLE_PATIENT


class FakeModel:
    """Replaces the trained model. Returns a probability of disease of 0.87
    for every patient."""

    def predict_proba(self, data):
        disease = np.full(len(data), 0.87)
        return np.column_stack([1 - disease, disease])


def patient(**changes):
    """The example patient, with some values changed."""
    values = dict(EXAMPLE_PATIENT)
    values.update(changes)
    return values


def metric(name, **labels):
    """Present value of a metric, 0 if it has no value yet."""
    return REGISTRY.get_sample_value(name, labels) or 0


@pytest.fixture
def client(monkeypatch):
    """The API with the fake model loaded."""
    loaded = LoadedModel(
        model=FakeModel(),
        name="heart-disease-classifier",
        version="7",
        run_id="abc123",
        trained_at="2026-10-12 18:19:42 UTC",
    )
    monkeypatch.setattr(main, "load_model", lambda: loaded)
    with TestClient(main.app) as test_client:
        yield test_client


def api_lines(caplog):
    """The log lines written by the API."""
    return [record.getMessage() for record in caplog.records if record.name == "api"]


def test_metrics_endpoint_gives_all_six_metrics(client):
    client.post("/predict", json=patient())

    reply = client.get("/metrics")

    assert reply.status_code == 200
    assert reply.headers["content-type"].startswith("text/plain")
    for name in ["api_requests_total", "api_request_duration_seconds", "api_model_loaded",
                 "api_predictions_total", "api_prediction_confidence",
                 "api_missing_fields_total"]:
        assert name in reply.text


def test_model_loaded_is_one_with_a_model(client):
    assert metric("api_model_loaded") == 1


def test_model_loaded_is_zero_without_a_model(monkeypatch):
    def failing_load():
        raise FileNotFoundError("no model folder")

    monkeypatch.setattr(main, "load_model", failing_load)
    with TestClient(main.app):
        assert metric("api_model_loaded") == 0


def test_prediction_is_counted(client):
    requests_before = metric("api_requests_total", path="/predict", status="200")
    disease_before = metric("api_predictions_total", label="disease")
    confidence_before = metric("api_prediction_confidence_bucket", le="0.9")

    client.post("/predict", json=patient())

    assert metric("api_requests_total", path="/predict", status="200") == requests_before + 1
    assert metric("api_predictions_total", label="disease") == disease_before + 1
    # a confidence of 0.87 falls in the bucket "up to 0.9"
    assert metric("api_prediction_confidence_bucket", le="0.9") == confidence_before + 1


def test_missing_field_is_counted(client):
    before = metric("api_missing_fields_total", field="ca")

    client.post("/predict", json=patient(ca=None))

    assert metric("api_missing_fields_total", field="ca") == before + 1


def test_refused_request_is_counted_but_not_predicted(client):
    refused_before = metric("api_requests_total", path="/predict", status="422")
    predictions_before = metric("api_predictions_total", label="disease")

    client.post("/predict", json=patient(cp=9))

    assert metric("api_requests_total", path="/predict", status="422") == refused_before + 1
    assert metric("api_predictions_total", label="disease") == predictions_before


def test_health_and_metrics_are_not_counted_or_logged(client, caplog):
    health_before = metric("api_requests_total", path="/health", status="200")

    with caplog.at_level(logging.INFO, logger="api"):
        client.get("/health")
        client.get("/metrics")

    assert metric("api_requests_total", path="/health", status="200") == health_before
    assert api_lines(caplog) == []


def test_unknown_endpoint_is_counted_as_other(client):
    before = metric("api_requests_total", path="other", status="404")

    client.get("/no-such-endpoint")

    assert metric("api_requests_total", path="other", status="404") == before + 1


def test_each_request_gets_its_own_id(client):
    first = client.post("/predict", json=patient()).headers["X-Request-ID"]
    second = client.post("/predict", json=patient()).headers["X-Request-ID"]

    assert len(first) == 8
    assert first != second


def test_log_line_of_a_prediction(client, caplog):
    with caplog.at_level(logging.INFO, logger="api"):
        reply = client.post("/predict", json=patient())

    [line] = api_lines(caplog)
    assert line.startswith("POST /predict status=200 time_ms=")
    assert f"request_id={reply.headers['X-Request-ID']}" in line
    assert "model_version=7 prediction=1 confidence=0.87" in line
    assert '"age":48.0' in line


def test_log_line_of_a_refused_request(client, caplog):
    with caplog.at_level(logging.INFO, logger="api"):
        client.post("/predict", json=patient(cp=9))

    [line] = api_lines(caplog)
    assert "status=422" in line
    assert "invalid_fields=cp" in line
    assert '"cp":9' in line


def test_log_lines_of_a_batch(client, caplog):
    with caplog.at_level(logging.INFO, logger="api"):
        client.post("/predict-batch", json={"patients": [patient(), patient(cp=9)]})

    summary, first_row, second_row = api_lines(caplog)
    assert "count=2 predicted=1 failed=1" in summary
    assert "row=1 status=ok prediction=1 confidence=0.87" in first_row
    assert "row=2 status=error invalid_fields=cp" in second_row


def test_long_input_is_cut_in_the_log(client, caplog):
    with caplog.at_level(logging.INFO, logger="api"):
        client.post("/predict", json=patient(cp="x" * 5000))

    [line] = api_lines(caplog)
    assert line.endswith("...(cut)")
    assert len(line) < monitoring.MAX_INPUT_LENGTH + 300
