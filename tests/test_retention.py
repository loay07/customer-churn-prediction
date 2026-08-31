"""Tests for the retention-intelligence layer (src/retention/).

Covers three things separately, matching the module split itself:
explanation math (explain.py), business rules (recommendations.py), and
their combination into one customer analysis (service.py) — including the
filtering/traceability rules from the project brief (recommendations only
from displayed risk drivers, non-actionable features never generate an
action, MonthlyCharges only suggested when genuinely above average).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.retention.explain import (
    compute_feature_contributions,
    get_training_means,
    top_protective_factors,
    top_risk_drivers,
)
from src.retention.recommendations import RETENTION_RULES, is_actionable, suggest_action
from src.retention.service import RetentionIntelligenceService

NON_ACTIONABLE_FEATURES = ["gender", "SeniorCitizen", "Partner", "Dependents", "TotalCharges", "InternetService"]


@pytest.fixture(scope="module")
def service() -> RetentionIntelligenceService:
    return RetentionIntelligenceService()


# ---------------------------------------------------------------------------
# explain.py
# ---------------------------------------------------------------------------


def test_training_means_cover_numeric_columns(service: RetentionIntelligenceService) -> None:
    means = get_training_means(service.model_service.model)
    assert set(means) == {"tenure", "MonthlyCharges", "TotalCharges"}
    assert all(value > 0 for value in means.values())


def test_contributions_cover_every_transformed_feature(service: RetentionIntelligenceService, sample_customer: dict) -> None:
    row = pd.DataFrame([sample_customer], columns=service.model_service.required_columns)
    contributions = compute_feature_contributions(service.model_service.model, row)
    n_transformed = len(service.model_service.model.named_steps["preprocessor"].get_feature_names_out())
    assert len(contributions) == n_transformed


def test_top_risk_drivers_are_sorted_and_positive(service: RetentionIntelligenceService, sample_customer: dict) -> None:
    row = pd.DataFrame([sample_customer], columns=service.model_service.required_columns)
    contributions = compute_feature_contributions(service.model_service.model, row)
    drivers = top_risk_drivers(contributions, top_n=5)

    assert (drivers["contribution"] > 0).all()
    assert list(drivers["contribution"]) == sorted(drivers["contribution"], reverse=True)


def test_top_protective_factors_are_sorted_and_negative(service: RetentionIntelligenceService, sample_customer: dict) -> None:
    row = pd.DataFrame([sample_customer], columns=service.model_service.required_columns)
    contributions = compute_feature_contributions(service.model_service.model, row)
    factors = top_protective_factors(contributions, top_n=5)

    assert (factors["contribution"] < 0).all()
    assert list(factors["contribution"]) == sorted(factors["contribution"])


# ---------------------------------------------------------------------------
# recommendations.py
# ---------------------------------------------------------------------------


def test_actionable_features_match_the_documented_playbook() -> None:
    assert set(RETENTION_RULES) == {
        "Contract", "TechSupport", "PaymentMethod", "tenure",
        "MonthlyCharges", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    }


@pytest.mark.parametrize("feature", NON_ACTIONABLE_FEATURES)
def test_non_actionable_features_are_flagged_correctly(feature: str) -> None:
    assert is_actionable(feature) is False


@pytest.mark.parametrize("feature", list(RETENTION_RULES))
def test_playbook_features_are_flagged_actionable(feature: str) -> None:
    assert is_actionable(feature) is True


def test_month_to_month_contract_suggests_upgrade() -> None:
    customer = pd.DataFrame([{"Contract": "Month-to-month"}])
    assert suggest_action("Contract", customer, {}) == "Offer an incentive to move to a longer-term contract"


@pytest.mark.parametrize("contract", ["One year", "Two year"])
def test_non_month_to_month_contract_suggests_nothing(contract: str) -> None:
    customer = pd.DataFrame([{"Contract": contract}])
    assert suggest_action("Contract", customer, {}) is None


def test_tech_support_yes_never_suggests_a_trial() -> None:
    customer = pd.DataFrame([{"TechSupport": "Yes"}])
    assert suggest_action("TechSupport", customer, {}) is None


def test_tech_support_no_suggests_a_trial() -> None:
    customer = pd.DataFrame([{"TechSupport": "No"}])
    assert suggest_action("TechSupport", customer, {}) == "Offer a free or discounted tech-support trial"


def test_pricing_review_requires_above_average_charges() -> None:
    means = {"MonthlyCharges": 64.81}
    below_average = pd.DataFrame([{"MonthlyCharges": 29.85}])
    above_average = pd.DataFrame([{"MonthlyCharges": 90.00}])

    assert suggest_action("MonthlyCharges", below_average, means) is None
    assert suggest_action("MonthlyCharges", above_average, means) == "Review the customer's current plan, pricing, and bundle"


def test_unknown_feature_suggests_nothing() -> None:
    customer = pd.DataFrame([{"gender": "Female"}])
    assert suggest_action("gender", customer, {}) is None


# ---------------------------------------------------------------------------
# service.py — full customer analysis
# ---------------------------------------------------------------------------


def test_analyze_matches_the_notebooks_worked_example(service: RetentionIntelligenceService) -> None:
    """Pins the exact numbers from notebooks/03_retention-intelligence.ipynb
    (customer 7590-VHVEG) so a future change can't silently alter them."""
    customer = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85,
    }
    result = service.analyze(customer)

    assert result.churn_probability == pytest.approx(0.6413025841448515)
    assert result.prediction == 1
    assert result.label == "Likely to Churn"
    assert [d.explanation for d in result.risk_drivers] == [
        "Tenure: 1 months (below average)",
        "Contract = Month-to-month",
        "Monthly charges: $29.85 (below average)",
        "PaymentMethod = Electronic check",
        "TechSupport = No",
    ]
    # Below-average MonthlyCharges must not produce a pricing recommendation,
    # even though it's a displayed risk driver.
    actions = {a.driver: a.action for a in result.suggested_actions}
    assert "Monthly charges: $29.85 (below average)" not in actions
    assert len(result.suggested_actions) == 4


def test_churn_probability_is_a_valid_probability(service: RetentionIntelligenceService, sample_customer: dict) -> None:
    result = service.analyze(sample_customer)
    assert 0.0 <= result.churn_probability <= 1.0


def test_threshold_is_030(service: RetentionIntelligenceService, sample_customer: dict) -> None:
    result = service.analyze(sample_customer)
    assert result.threshold == pytest.approx(0.30)


def test_loyal_customer_gets_no_inappropriate_actions(service: RetentionIntelligenceService, loyal_customer: dict) -> None:
    """loyal_customer already has tech support, a two-year contract, automatic
    payment, high tenure, and below-average charges — every rule's condition
    should fail, even for features that still show up as (weak) drivers."""
    result = service.analyze(loyal_customer)

    action_text = " ".join(a.action for a in result.suggested_actions)
    assert "trial" not in action_text.lower()
    assert "longer-term contract" not in action_text.lower()
    assert "automatic payment" not in action_text.lower()
    assert "onboarding" not in action_text.lower()
    assert "pricing" not in action_text.lower() and "bundle" not in action_text.lower()


def test_non_actionable_drivers_never_produce_a_suggested_action(
    service: RetentionIntelligenceService, sample_customer: dict, loyal_customer: dict
) -> None:
    for customer in (sample_customer, loyal_customer):
        result = service.analyze(customer)
        actionable_driver_texts = {
            d.explanation for d in (*result.risk_drivers, *result.protective_factors) if d.actionable
        }
        for action in result.suggested_actions:
            assert action.driver in actionable_driver_texts


def test_recommendations_only_come_from_displayed_risk_drivers(
    service: RetentionIntelligenceService, sample_customer: dict
) -> None:
    """sample_customer has 4 rule-triggering signals (tenure, Contract,
    PaymentMethod, TechSupport), but limiting to the single strongest driver
    must suppress the other three even though their conditions are true."""
    full = service.analyze(sample_customer, top_n=5)
    assert len(full.suggested_actions) == 4

    limited = service.analyze(sample_customer, top_n=1)
    assert len(limited.risk_drivers) == 1
    assert len(limited.suggested_actions) == 1
    assert limited.suggested_actions[0].driver == limited.risk_drivers[0].explanation


def test_electronic_check_only_suggested_when_a_displayed_driver(
    service: RetentionIntelligenceService, sample_customer: dict
) -> None:
    limited = service.analyze(sample_customer, top_n=1)  # top driver is tenure, not PaymentMethod
    assert all("automatic payment" not in a.action for a in limited.suggested_actions)
