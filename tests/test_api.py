"""Tests for the FastAPI churn prediction service (api/main.py)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["threshold"] == pytest.approx(0.30)
    assert body["model_type"] == "Logistic Regression"


def test_predict_valid_request(client: TestClient, sample_customer: dict) -> None:
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 200
    body = response.json()

    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("Likely to Churn", "Likely to Stay")
    assert body["threshold"] == pytest.approx(0.30)
    assert body["risk_category"] in ("Low", "Medium", "High")


def test_predict_applies_threshold_consistently(client: TestClient, sample_customer: dict) -> None:
    response = client.post("/predict", json=sample_customer)
    body = response.json()
    expected_prediction = int(body["churn_probability"] >= body["threshold"])
    assert body["prediction"] == expected_prediction


def test_predict_missing_field_returns_422(client: TestClient, sample_customer: dict) -> None:
    incomplete = dict(sample_customer)
    del incomplete["Contract"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_invalid_category_returns_422(client: TestClient, sample_customer: dict) -> None:
    invalid = dict(sample_customer)
    invalid["InternetService"] = "Satellite"  # not a category seen during training
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_negative_tenure_returns_422(client: TestClient, sample_customer: dict) -> None:
    invalid = dict(sample_customer)
    invalid["tenure"] = -5
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_docs_available(client: TestClient) -> None:
    response = client.get("/docs")
    assert response.status_code == 200
