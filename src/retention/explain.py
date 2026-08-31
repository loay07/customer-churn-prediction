"""Per-customer explanation of the churn model's prediction.

This module answers "which signals pushed the model toward this
prediction?" — nothing more. It has no opinion on what the business
should do about those signals (see ``recommendations.py`` for that) and
no opinion on the prediction itself (see ``src/churn_model.py``).

The math mirrors exactly what the fitted pipeline already computes: for a
logistic regression over one-hot-encoded and standard-scaled inputs, the
decision function is a sum of ``coefficient * transformed_feature_value``
terms plus an intercept, passed through a sigmoid. A single term's value
(its "contribution") is therefore a faithful decomposition of *this
specific prediction* into per-feature pieces — a positive contribution
pushed the predicted probability up, a negative one pushed it down.

This describes what influenced the model's output. It is not a causal
claim about why the customer actually would churn — see the README's
"Model prediction / Model explanation / Business recommendation"
distinction.
"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

TOP_N_DEFAULT = 5

# Human-readable labels for numeric features that warrant custom phrasing.
# Any numeric feature not listed here falls back to a generic "<name>: <value>" format.
_NUMERIC_LABELS = {
    "tenure": lambda value, comparison: f"Tenure: {value:.0f} months ({comparison})",
    "MonthlyCharges": lambda value, comparison: f"Monthly charges: ${value:.2f} ({comparison})",
    "TotalCharges": lambda value, comparison: f"Total charges: ${value:.2f} ({comparison})",
}


def get_training_means(pipeline: Pipeline) -> dict[str, float]:
    """Return the fitted StandardScaler's per-feature training means.

    Used both to phrase numeric explanations ("below average" / "above
    average") and, in ``recommendations.py``, to gate the MonthlyCharges
    pricing rule so it only fires for genuinely above-average charges.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    numerical_columns = list(preprocessor.transformers_[1][2])
    scaler = preprocessor.named_transformers_["numerical"]
    return dict(zip(numerical_columns, scaler.mean_))


def _split_transformed_name(feature_name: str, categorical_columns: list[str]) -> tuple[str, str | None]:
    """Recover the original column (and, for one-hot columns, the category)
    behind a ``ColumnTransformer`` output name.

    e.g. ``"categorical__Contract_Month-to-month"`` -> ``("Contract", "Month-to-month")``
         ``"numerical__tenure"`` -> ``("tenure", None)``
    """
    if feature_name.startswith("numerical__"):
        return feature_name.removeprefix("numerical__"), None

    cleaned = feature_name.removeprefix("categorical__")
    for column in categorical_columns:
        prefix = f"{column}_"
        if cleaned.startswith(prefix):
            return column, cleaned[len(prefix) :]
    return cleaned, None


def _describe_numerical(base_feature: str, value: float, mean: float) -> str:
    comparison = "below average" if value < mean else "above average"
    formatter = _NUMERIC_LABELS.get(base_feature, lambda v, c: f"{base_feature}: {v:.2f} ({c})")
    return formatter(value, comparison)


def compute_feature_contributions(pipeline: Pipeline, customer: pd.DataFrame) -> pd.DataFrame:
    """Compute this customer's contribution for every transformed model feature.

    ``customer`` must be a one-row DataFrame with exactly the pipeline's raw
    input columns (see ``ChurnModelService.required_columns``) and already
    correctly typed (e.g. ``TotalCharges`` numeric) — preprocessing itself
    is delegated entirely to the pipeline's own fitted ``ColumnTransformer``,
    so this can never drift from what the model was actually trained on.

    Returns one row per transformed feature with columns:
    ``base_feature`` (the original column, e.g. "Contract"),
    ``explanation`` (readable text, e.g. "Contract = Month-to-month"),
    ``contribution`` (``coefficient * transformed_value``; positive pushes
    toward churn, negative pushes toward staying).
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    categorical_columns = list(preprocessor.transformers_[0][2])
    training_means = get_training_means(pipeline)

    transformed = preprocessor.transform(customer)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    customer_values = transformed[0]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.coef_[0]
    contributions = customer_values * coefficients

    rows = []
    for name, contribution in zip(feature_names, contributions):
        base_feature, category = _split_transformed_name(name, categorical_columns)
        if category is not None:
            explanation = f"{base_feature} = {category}"
        else:
            value = float(customer[base_feature].iloc[0])
            explanation = _describe_numerical(base_feature, value, training_means[base_feature])
        rows.append(
            {
                "base_feature": base_feature,
                "explanation": explanation,
                "contribution": float(contribution),
            }
        )

    return pd.DataFrame(rows)


def top_risk_drivers(contributions: pd.DataFrame, top_n: int = TOP_N_DEFAULT) -> pd.DataFrame:
    """The strongest signals that pushed this customer's prediction toward churn."""
    return (
        contributions[contributions["contribution"] > 0]
        .sort_values("contribution", ascending=False)
        .head(top_n)
    )


def top_protective_factors(contributions: pd.DataFrame, top_n: int = TOP_N_DEFAULT) -> pd.DataFrame:
    """The strongest signals that pushed this customer's prediction toward staying."""
    return (
        contributions[contributions["contribution"] < 0]
        .sort_values("contribution", ascending=True)
        .head(top_n)
    )
