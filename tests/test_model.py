"""Tests for the ChurnModelService inference wrapper (src/churn_model.py)."""

from __future__ import annotations

import pytest

from src.churn_model import ChurnModelService

EXPECTED_COLUMNS = {
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
}


@pytest.fixture(scope="module")
def service() -> ChurnModelService:
    return ChurnModelService()


def test_model_loads(service: ChurnModelService) -> None:
    assert service.model is not None
    assert hasattr(service.model, "predict_proba")


def test_metadata_threshold_is_030(service: ChurnModelService) -> None:
    assert service.threshold == pytest.approx(0.30)


def test_required_columns_match_original_schema(service: ChurnModelService) -> None:
    assert set(service.required_columns) == EXPECTED_COLUMNS


def test_predict_returns_probability_between_0_and_1(service, sample_customer) -> None:
    result = service.predict(sample_customer)
    assert 0.0 <= result.churn_probability <= 1.0


def test_threshold_is_applied_correctly(service, sample_customer) -> None:
    result = service.predict(sample_customer)
    expected_prediction = int(result.churn_probability >= service.threshold)
    assert result.prediction == expected_prediction
    expected_label = "Likely to Churn" if expected_prediction == 1 else "Likely to Stay"
    assert result.label == expected_label


def test_high_risk_profile_scores_higher_than_low_risk_profile(
    service, sample_customer, loyal_customer
) -> None:
    high_risk_prob = service.predict(sample_customer).churn_probability
    low_risk_prob = service.predict(loyal_customer).churn_probability
    assert high_risk_prob > low_risk_prob


def test_missing_required_field_raises_value_error(service, sample_customer) -> None:
    incomplete = dict(sample_customer)
    del incomplete["tenure"]
    with pytest.raises(ValueError):
        service.predict(incomplete)
