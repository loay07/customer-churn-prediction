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


def test_predict_includes_retention_intelligence(client: TestClient, sample_customer: dict) -> None:
    """The /predict response must serialize cleanly through FastAPI and
    include the new explanation/recommendation fields alongside the
    original prediction fields."""
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 200
    body = response.json()

    assert isinstance(body["risk_drivers"], list) and len(body["risk_drivers"]) > 0
    assert isinstance(body["protective_factors"], list)
    assert isinstance(body["suggested_actions"], list)

    for driver in body["risk_drivers"] + body["protective_factors"]:
        assert set(driver) == {"feature", "explanation", "contribution", "actionable"}
        assert isinstance(driver["actionable"], bool)

    for action in body["suggested_actions"]:
        assert set(action) == {"driver", "action"}


def test_predict_risk_drivers_are_sorted_and_positive(client: TestClient, sample_customer: dict) -> None:
    body = client.post("/predict", json=sample_customer).json()
    contributions = [d["contribution"] for d in body["risk_drivers"]]
    assert all(c > 0 for c in contributions)
    assert contributions == sorted(contributions, reverse=True)


def test_predict_protective_factors_are_negative(client: TestClient, sample_customer: dict) -> None:
    body = client.post("/predict", json=sample_customer).json()
    contributions = [d["contribution"] for d in body["protective_factors"]]
    assert all(c < 0 for c in contributions)


def test_predict_tech_support_yes_gets_no_trial_recommendation(client: TestClient, loyal_customer: dict) -> None:
    body = client.post("/predict", json=loyal_customer).json()
    assert all("tech-support trial" not in a["action"] for a in body["suggested_actions"])


def test_predict_two_year_contract_gets_no_contract_upgrade_recommendation(
    client: TestClient, loyal_customer: dict
) -> None:
    body = client.post("/predict", json=loyal_customer).json()
    assert all("longer-term contract" not in a["action"] for a in body["suggested_actions"])


def test_predict_below_average_charges_get_no_pricing_recommendation(
    client: TestClient, sample_customer: dict
) -> None:
    """sample_customer's MonthlyCharges is above average but shows up as a
    protective factor, not a risk driver — no pricing action should appear."""
    body = client.post("/predict", json=sample_customer).json()
    assert all("plan, pricing, and bundle" not in a["action"] for a in body["suggested_actions"])
