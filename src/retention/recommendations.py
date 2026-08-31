"""Business-rule retention playbook.

Maps a churn-risk driver (an original feature, e.g. "Contract") to a
retention action a telecom team could consider offering — *if* that
feature is actually one of the customer's displayed risk drivers and its
specific condition applies to them.

These are business-rule suggestions based on what influenced the model's
prediction. They are not causally validated treatments: the dataset has
no record of which offers were ever made to a customer or whether an
offer changed their behavior (see the README's limitations section). A
suggestion here means "this is a lever the business has for this kind of
signal," not "this is proven to retain this customer."

Every rule is a single, readable entry below — a business analyst can
scan this table top to bottom and know exactly which condition produces
which suggestion, without reading any surrounding code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

TrainingMeans = dict[str, float]
Condition = Callable[[pd.DataFrame, TrainingMeans], bool]


@dataclass(frozen=True)
class RetentionRule:
    feature: str
    condition: Condition
    action: str


RETENTION_RULES: dict[str, RetentionRule] = {
    "Contract": RetentionRule(
        feature="Contract",
        condition=lambda customer, means: customer["Contract"].iloc[0] == "Month-to-month",
        action="Offer an incentive to move to a longer-term contract",
    ),
    "TechSupport": RetentionRule(
        feature="TechSupport",
        condition=lambda customer, means: customer["TechSupport"].iloc[0] == "No",
        action="Offer a free or discounted tech-support trial",
    ),
    "PaymentMethod": RetentionRule(
        feature="PaymentMethod",
        condition=lambda customer, means: customer["PaymentMethod"].iloc[0] == "Electronic check",
        action="Offer an incentive to switch to automatic payment",
    ),
    "tenure": RetentionRule(
        feature="tenure",
        condition=lambda customer, means: float(customer["tenure"].iloc[0]) < 12,
        action="Place customer in an early-life retention or onboarding campaign",
    ),
    # Deliberately excludes a customer already paying below-average charges:
    # a positive contribution there means low charges pushed toward churn in
    # this model, not that the customer is overpaying — see README.
    "MonthlyCharges": RetentionRule(
        feature="MonthlyCharges",
        condition=lambda customer, means: float(customer["MonthlyCharges"].iloc[0]) > means["MonthlyCharges"],
        action="Review the customer's current plan, pricing, and bundle",
    ),
    "OnlineSecurity": RetentionRule(
        feature="OnlineSecurity",
        condition=lambda customer, means: customer["OnlineSecurity"].iloc[0] == "No",
        action="Consider an Online Security trial or bundled offer",
    ),
    "OnlineBackup": RetentionRule(
        feature="OnlineBackup",
        condition=lambda customer, means: customer["OnlineBackup"].iloc[0] == "No",
        action="Consider an Online Backup trial or bundled offer",
    ),
    "DeviceProtection": RetentionRule(
        feature="DeviceProtection",
        condition=lambda customer, means: customer["DeviceProtection"].iloc[0] == "No",
        action="Consider a Device Protection trial or bundled offer",
    ),
}
# Features such as gender, SeniorCitizen, Partner, Dependents, InternetService,
# PhoneService, MultipleLines, StreamingTV/Movies, PaperlessBilling, and
# TotalCharges are intentionally absent: they may still appear as contextual
# prediction signals (see explain.py), but the business has no playbook entry
# for them, so they never generate a suggested action.


def is_actionable(base_feature: str) -> bool:
    """Whether this feature type has any retention rule at all.

    This reflects "is this the *kind* of signal the business can act on,"
    independent of whether that rule's condition happens to fire for a
    given customer — e.g. TechSupport is actionable even for a customer
    who already has it (no action fires, but the feature type still is).
    """
    return base_feature in RETENTION_RULES


def suggest_action(base_feature: str, customer: pd.DataFrame, training_means: TrainingMeans) -> str | None:
    """Return the suggested action for this feature and customer, if any.

    Returns ``None`` when there is no rule for this feature, or when there
    is a rule but its condition doesn't apply to this specific customer.
    """
    rule = RETENTION_RULES.get(base_feature)
    if rule is None:
        return None
    if rule.condition(customer, training_means):
        return rule.action
    return None
