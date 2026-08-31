"""Orchestrates prediction, explanation, and retention recommendations into
one customer-level "retention intelligence" result.

Keeps three responsibilities separate, on purpose:

    ChurnModelService   decides what this customer looks like to the
                        trained model (probability, threshold, label).
    explain.py          decides which signals drove that prediction.
    recommendations.py  decides what the business could consider doing
                        about those signals.

This module only wires the three together; it contains no modeling math
and no business rules of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.churn_model import ChurnModelService
from src.retention.explain import (
    TOP_N_DEFAULT,
    compute_feature_contributions,
    get_training_means,
    top_protective_factors,
    top_risk_drivers,
)
from src.retention.recommendations import is_actionable, suggest_action


@dataclass(frozen=True)
class Driver:
    feature: str
    explanation: str
    contribution: float
    actionable: bool


@dataclass(frozen=True)
class SuggestedAction:
    driver: str
    action: str


@dataclass(frozen=True)
class RetentionIntelligence:
    churn_probability: float
    prediction: int
    label: str
    threshold: float
    risk_category: str
    risk_drivers: list[Driver]
    protective_factors: list[Driver]
    suggested_actions: list[SuggestedAction]


class RetentionIntelligenceService:
    """Produces a full retention-intelligence analysis for one customer."""

    def __init__(self, model_service: ChurnModelService | None = None) -> None:
        self.model_service = model_service or ChurnModelService()

    def analyze(self, customer: dict[str, Any], top_n: int = TOP_N_DEFAULT) -> RetentionIntelligence:
        """Score, explain, and generate retention suggestions for one customer.

        ``customer`` must contain exactly the raw feature columns the
        pipeline expects (see ``ChurnModelService.required_columns``).
        """
        prediction = self.model_service.predict(customer)

        row = pd.DataFrame([customer], columns=self.model_service.required_columns)
        pipeline = self.model_service.model
        contributions = compute_feature_contributions(pipeline, row)
        training_means = get_training_means(pipeline)

        risk_df = top_risk_drivers(contributions, top_n=top_n)
        protective_df = top_protective_factors(contributions, top_n=top_n)

        risk_drivers = [
            Driver(
                feature=r.base_feature,
                explanation=r.explanation,
                contribution=r.contribution,
                actionable=is_actionable(r.base_feature),
            )
            for r in risk_df.itertuples()
        ]
        protective_factors = [
            Driver(
                feature=r.base_feature,
                explanation=r.explanation,
                contribution=r.contribution,
                actionable=is_actionable(r.base_feature),
            )
            for r in protective_df.itertuples()
        ]

        # Recommendations come only from the displayed risk drivers, never
        # from protective factors or from features outside the top-N shown.
        suggested_actions = []
        for r in risk_df.itertuples():
            action = suggest_action(r.base_feature, row, training_means)
            if action is not None:
                suggested_actions.append(SuggestedAction(driver=r.explanation, action=action))

        return RetentionIntelligence(
            churn_probability=prediction.churn_probability,
            prediction=prediction.prediction,
            label=prediction.label,
            threshold=prediction.threshold,
            risk_category=prediction.risk_category,
            risk_drivers=risk_drivers,
            protective_factors=protective_factors,
            suggested_actions=suggested_actions,
        )
