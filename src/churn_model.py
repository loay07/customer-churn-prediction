"""Inference wrapper around the saved churn prediction pipeline.

The saved artifact (``models/churn_logistic_model.joblib``) is a single
scikit-learn ``Pipeline`` containing a ``ColumnTransformer`` (one-hot
encoding for categorical columns, standard scaling for numeric columns)
followed by the trained ``LogisticRegression`` model. Preprocessing lives
inside the pipeline, so this wrapper only has to hand it raw, unencoded
customer data and read back a probability.

The classification threshold (0.30) was chosen during model development
via stratified k-fold cross-validation, not the sklearn default of 0.50,
to prioritize recall (catching churners) over precision. See the README
for the full reasoning.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "churn_logistic_model.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

# Risk-category cut points used only to make the API/UI easier to read.
# This is a fixed business-facing bucketing of the probability output,
# not a second trained model.
MEDIUM_RISK_CUTOFF = 0.60


@dataclass
class ChurnPrediction:
    churn_probability: float
    prediction: int
    label: str
    threshold: float
    risk_category: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _risk_category(probability: float, threshold: float) -> str:
    if probability >= MEDIUM_RISK_CUTOFF:
        return "High"
    if probability >= threshold:
        return "Medium"
    return "Low"


class ChurnModelService:
    """Loads the trained pipeline once and serves predictions from it."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
    ) -> None:
        self.model = joblib.load(model_path)

        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        self.threshold: float = metadata["threshold"]
        self.model_type: str = metadata.get("model_type", "unknown")

        preprocessor = self.model.named_steps["preprocessor"]
        self.required_columns: list[str] = list(preprocessor.feature_names_in_)

    def predict(self, customer: dict[str, Any]) -> ChurnPrediction:
        """Score one raw customer record and apply the decision threshold.

        ``customer`` must contain exactly the raw feature columns the
        pipeline was trained on (see ``self.required_columns``); the
        pipeline's own ``ColumnTransformer`` handles one-hot encoding and
        scaling internally.
        """
        missing = set(self.required_columns) - set(customer)
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")

        row = pd.DataFrame([customer], columns=self.required_columns)
        probability = float(self.model.predict_proba(row)[0, 1])
        prediction = int(probability >= self.threshold)
        label = "Likely to Churn" if prediction == 1 else "Likely to Stay"

        return ChurnPrediction(
            churn_probability=probability,
            prediction=prediction,
            label=label,
            threshold=self.threshold,
            risk_category=_risk_category(probability, self.threshold),
        )
