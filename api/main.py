"""FastAPI service that serves predictions from the saved churn pipeline.

Run locally with:

    uvicorn api.main:app --reload

Then open http://127.0.0.1:8000/ for the demo frontend or
http://127.0.0.1:8000/docs for interactive API docs.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.schemas import CustomerFeatures, HealthResponse, RetentionAction, RetentionIntelligenceResponse, RiskDriver
from src.retention.service import RetentionIntelligenceService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"

_service: RetentionIntelligenceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    try:
        _service = RetentionIntelligenceService()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not load the trained model or metadata from models/. "
            "Make sure churn_logistic_model.joblib and model_metadata.json exist."
        ) from exc

    schema_fields = set(CustomerFeatures.model_fields)
    pipeline_fields = set(_service.model_service.required_columns)
    if schema_fields != pipeline_fields:
        raise RuntimeError(
            "CustomerFeatures schema does not match the trained pipeline's "
            f"expected columns. Schema-only: {schema_fields - pipeline_fields}, "
            f"pipeline-only: {pipeline_fields - schema_fields}"
        )

    yield
    _service = None


app = FastAPI(
    title="Customer Retention Intelligence API",
    description=(
        "Serves churn-risk predictions, per-customer prediction explanations, "
        "and business-rule retention suggestions, built around a logistic "
        "regression pipeline trained on the IBM Telco Customer Churn dataset. "
        "The classification threshold (0.30) was tuned for recall so the "
        "business catches more at-risk customers, at the cost of some false "
        "alarms. Explanations describe what influenced the model's prediction, "
        "not why a customer would actually churn; suggested actions are "
        "business-rule heuristics, not causally validated treatments."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_service() -> RetentionIntelligenceService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return _service


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health() -> dict[str, Any]:
    service = get_service()
    return {
        "status": "ok",
        "model_type": service.model_service.model_type,
        "threshold": service.model_service.threshold,
    }


@app.post("/predict", response_model=RetentionIntelligenceResponse, tags=["prediction"])
def predict(customer: CustomerFeatures) -> RetentionIntelligenceResponse:
    """Score a customer and return the full retention-intelligence result:
    churn prediction, the signals that drove it, and business-rule
    retention suggestions traceable to those signals."""
    service = get_service()
    try:
        result = service.analyze(customer.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return RetentionIntelligenceResponse(
        churn_probability=result.churn_probability,
        prediction=result.prediction,
        label=result.label,
        threshold=result.threshold,
        risk_category=result.risk_category,
        risk_drivers=[RiskDriver(**d.__dict__) for d in result.risk_drivers],
        protective_factors=[RiskDriver(**d.__dict__) for d in result.protective_factors],
        suggested_actions=[RetentionAction(**a.__dict__) for a in result.suggested_actions],
    )


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
