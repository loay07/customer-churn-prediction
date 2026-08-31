"""Pydantic request/response models for the churn prediction API.

The ``Literal`` values on ``CustomerFeatures`` are copied from the
categories the ``OneHotEncoder`` inside the saved pipeline was actually
fit on (inspected via ``encoder.categories_``), so a request with a typo
or an unsupported category is rejected with a clear 422 instead of
silently producing a meaningless prediction.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

YesNo = Literal["No", "Yes"]
YesNoNA = Literal["No", "No internet service", "Yes"]

_EXAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 420.75,
}


class CustomerFeatures(BaseModel):
    """Raw, unencoded customer attributes — exactly what a support agent
    or CRM record would hold. The saved pipeline performs one-hot encoding
    and scaling internally, so no preprocessing happens in the API layer.
    """

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1] = Field(description="1 if the customer is a senior citizen")
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months the customer has stayed with the company")
    PhoneService: YesNo
    MultipleLines: Literal["No", "No phone service", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNA
    OnlineBackup: YesNoNA
    DeviceProtection: YesNoNA
    TechSupport: YesNoNA
    StreamingTV: YesNoNA
    StreamingMovies: YesNoNA
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    MonthlyCharges: float = Field(ge=0, description="Current monthly charge in USD")
    TotalCharges: float = Field(ge=0, description="Total amount charged to the customer so far")

    model_config = {"json_schema_extra": {"example": _EXAMPLE_CUSTOMER}}


class RiskDriver(BaseModel):
    """One signal that pushed the model's prediction in some direction.

    This describes what influenced the *model's output* — it is not a
    causal claim about why the customer would actually churn. Use wording
    like "contributed toward this prediction," never "because of."
    """

    feature: str = Field(description="Original feature this signal comes from, e.g. \"Contract\"")
    explanation: str = Field(description="Human-readable description, e.g. \"Contract = Month-to-month\"")
    contribution: float = Field(
        description=(
            "coefficient * transformed_feature_value for this customer. Positive "
            "pushed the prediction toward churn, negative pushed it toward staying. "
            "Shown for transparency/debugging — the business-facing UI should lead "
            "with `explanation`, not this number."
        )
    )
    actionable: bool = Field(
        description=(
            "Whether this feature type has a retention rule at all (see "
            "src/retention/recommendations.py) — not whether an action was "
            "actually suggested for this customer. Non-actionable examples: "
            "gender, SeniorCitizen, Partner, Dependents, TotalCharges."
        )
    )


class RetentionAction(BaseModel):
    """A business-rule retention suggestion — not a causally validated
    treatment. Always traceable to one of the customer's displayed risk
    drivers; never generated from a protective factor or a hidden feature."""

    driver: str = Field(description="The risk driver's explanation text that produced this suggestion")
    action: str = Field(description="Suggested retention action a business team could consider")


class RetentionIntelligenceResponse(BaseModel):
    """Customer Retention Intelligence: a churn prediction plus why the
    model made it and what the business could consider doing about it.

    Three distinct layers, kept separate on purpose:
      - Model prediction: churn_probability / prediction / label / threshold / risk_category
      - Model explanation: risk_drivers / protective_factors
      - Business recommendation: suggested_actions
    """

    churn_probability: float = Field(description="Model-estimated probability of churn, between 0 and 1")
    prediction: Literal[0, 1] = Field(description="1 if churn_probability is at or above the threshold")
    label: Literal["Likely to Churn", "Likely to Stay"]
    threshold: float = Field(description="Decision threshold applied to the probability")
    risk_category: Literal["Low", "Medium", "High"] = Field(
        description=(
            "UI-only interpretation bucket derived from the probability for quick "
            "scanning — a fixed business rule, not a separately trained or "
            "statistically calibrated model."
        )
    )
    risk_drivers: list[RiskDriver] = Field(
        description="Strongest signals that pushed this prediction toward churn, most influential first."
    )
    protective_factors: list[RiskDriver] = Field(
        description="Strongest signals that pushed this prediction toward staying, most influential first."
    )
    suggested_actions: list[RetentionAction] = Field(
        description=(
            "Business-rule retention suggestions derived only from the risk drivers "
            "above. Not causally validated — the dataset has no record of which "
            "offers were made historically or whether they worked."
        )
    )


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_type: str
    threshold: float
