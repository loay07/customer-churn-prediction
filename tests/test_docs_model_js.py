"""Verifies the static-site JS scorer (docs/model.js + docs/model_params.js)
produces the same predictions as the real sklearn pipeline.

The GitHub Pages demo under docs/ can't call the FastAPI backend, so it
re-implements the trained logistic regression's decision function in
JavaScript using weights exported from the same models/churn_logistic_model.joblib
(see docs/export_model_params.py). This test runs that actual JS file
through Node and cross-checks it against ChurnModelService, so a change to
either the model or the JS that breaks parity is caught here rather than
silently shipping a static demo that disagrees with the real API.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.churn_model import ChurnModelService

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed; skipping JS parity test")


def _predict_via_node(customer: dict) -> dict:
    script = f"""
const {{ predictChurn }} = require({json.dumps(str(DOCS_DIR / "model.js"))});
const params = require({json.dumps(str(DOCS_DIR / "model_params.js"))});
const customer = {json.dumps(customer)};
console.log(JSON.stringify(predictChurn(customer, params)));
"""
    result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"node failed:\n{result.stderr}"
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def service() -> ChurnModelService:
    return ChurnModelService()


def test_js_probability_matches_python_pipeline(service: ChurnModelService, sample_customer: dict) -> None:
    node_result = _predict_via_node(sample_customer)
    real_result = service.predict(sample_customer)

    assert node_result["churn_probability"] == pytest.approx(real_result.churn_probability, abs=1e-9)
    assert node_result["prediction"] == real_result.prediction
    assert node_result["label"] == real_result.label
    assert node_result["risk_category"] == real_result.risk_category


def test_js_probability_matches_python_pipeline_for_loyal_customer(
    service: ChurnModelService, loyal_customer: dict
) -> None:
    node_result = _predict_via_node(loyal_customer)
    real_result = service.predict(loyal_customer)

    assert node_result["churn_probability"] == pytest.approx(real_result.churn_probability, abs=1e-9)
    assert node_result["risk_category"] == real_result.risk_category
