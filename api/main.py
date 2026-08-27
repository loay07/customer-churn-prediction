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

from api.schemas import ChurnPredictionResponse, CustomerFeatures, HealthResponse
from src.churn_model import ChurnModelService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"

_service: ChurnModelService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    try:
        _service = ChurnModelService()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not load the trained model or metadata from models/. "
            "Make sure churn_logistic_model.joblib and model_metadata.json exist."
        ) from exc

    schema_fields = set(CustomerFeatures.model_fields)
    pipeline_fields = set(_service.required_columns)
    if schema_fields != pipeline_fields:
        raise RuntimeError(
            "CustomerFeatures schema does not match the trained pipeline's "
            f"expected columns. Schema-only: {schema_fields - pipeline_fields}, "
            f"pipeline-only: {pipeline_fields - schema_fields}"
        )

    yield
    _service = None


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Serves churn-risk predictions from a logistic regression pipeline "
        "trained on the IBM Telco Customer Churn dataset. The classification "
        "threshold (0.30) was tuned for recall so the business catches more "
        "at-risk customers, at the cost of some false alarms."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_service() -> ChurnModelService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return _service


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health() -> dict[str, Any]:
    service = get_service()
    return {"status": "ok", "model_type": service.model_type, "threshold": service.threshold}


@app.post("/predict", response_model=ChurnPredictionResponse, tags=["prediction"])
def predict(customer: CustomerFeatures) -> dict[str, Any]:
    service = get_service()
    try:
        result = service.predict(customer.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.to_dict()


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
