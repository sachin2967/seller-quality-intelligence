"""FastAPI demonstration service for the Seller Quality Intelligence prototype.

Exposes retrospective demo scoring, individual seller lookups, survival estimates,
and merchant operations interventions.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from api.boundary import OperatorBoundary

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Initialize FastAPI App
app = FastAPI(
    title="Seller Quality & Risk Intelligence API",
    description="Offline prototype API for retrospective seller diagnostics; scores are not validated for prospective decisions.",
    version="1.0.0",
)
app.add_middleware(OperatorBoundary)

# Load artifacts into memory. A failed load leaves a visible not-ready state.
champion_model = None
feature_names = []
survival_model = None
explainer = None
seller_db = None
try:
    classifier_artifact = joblib.load(MODELS_DIR / "lgbm_seller_risk.joblib")
    loaded_survival = joblib.load(MODELS_DIR / "seller_survival_cox.joblib")
    loaded_explainer = joblib.load(MODELS_DIR / "seller_explainer.joblib")

    # Load precomputed seller feature database
    loaded_db = pd.read_csv(DATA_DIR / "seller_features.csv")
    if (
        loaded_db.empty
        or loaded_db["seller_id"].isna().any()
        or loaded_db["seller_id"].duplicated().any()
    ):
        raise ValueError("Seller database must have unique non-null seller IDs.")
    loaded_db.set_index("seller_id", inplace=True, drop=False)
    loaded_features = classifier_artifact["feature_names"]
    if not loaded_features or not set(loaded_features).issubset(loaded_db.columns):
        raise ValueError("Seller database does not match the model feature schema.")
    # Publish dependencies together; partial loads must not enable scoring.
    champion_model = classifier_artifact["model"]
    feature_names = loaded_features
    survival_model = loaded_survival
    explainer = loaded_explainer
    seller_db = loaded_db
except (OSError, EOFError, KeyError, ValueError, TypeError, ImportError) as exc:
    logging.getLogger(__name__).warning(
        "Seller artifacts unavailable (%s)", type(exc).__name__
    )


SCORE_SCOPE = {
    "evaluation_scope": "retrospective rule-derived seller label; not validated prospective risk",
    "score_interpretation": "uncalibrated model score; not an intervention effect or realized savings",
    "action_scope": "demonstration recommendation requiring independent operational validation",
}


def require_scoring_artifacts():
    if champion_model is None or explainer is None or not feature_names:
        raise HTTPException(503, "Scoring dependencies unavailable.")


def predict_and_explain(row):
    require_scoring_artifacts()
    try:
        values = pd.DataFrame(
            [{name: row[name] for name in feature_names}], columns=feature_names
        )
        probability = float(champion_model.predict_proba(values)[0, 1])
        if not 0 <= probability <= 1:
            raise ValueError("Invalid score")
        explanation = explainer.explain_seller(pd.Series(row))
        if "drivers" not in explanation or "recommended_action" not in explanation:
            raise ValueError("Invalid explanation")
        return probability, explanation
    except Exception as exc:
        raise HTTPException(503, "Scoring or explanation unavailable.") from exc


@app.get("/live")
def live():
    return {"status": "alive"}


class SellerMetricsInput(BaseModel):
    """Raw seller features for retrospective demonstration scoring."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    total_gmv: FiniteFloat = Field(
        ...,
        ge=0,
        le=1_000_000_000_000,
        json_schema_extra={"example": 4500.0},
        description="Total GMV in currency",
    )
    total_items: int = Field(
        ...,
        ge=0,
        le=100_000_000,
        json_schema_extra={"example": 25},
        description="Total items sold",
    )
    unique_orders: int = Field(
        ...,
        ge=0,
        le=100_000_000,
        json_schema_extra={"example": 20},
        description="Unique orders fulfilled",
    )
    unique_products: int = Field(
        ...,
        ge=0,
        le=100_000_000,
        json_schema_extra={"example": 8},
        description="Distinct product SKUs",
    )
    avg_item_price: FiniteFloat = Field(
        ...,
        ge=0,
        le=1_000_000_000,
        json_schema_extra={"example": 180.0},
        description="Average item selling price",
    )
    std_item_price: FiniteFloat = Field(
        ...,
        ge=0,
        le=1_000_000_000,
        json_schema_extra={"example": 45.0},
        description="Standard deviation of item price",
    )
    avg_freight: FiniteFloat = Field(
        ...,
        ge=0,
        le=1_000_000_000,
        json_schema_extra={"example": 22.0},
        description="Average shipping freight cost",
    )
    freight_ratio: FiniteFloat = Field(
        ...,
        ge=0,
        le=1000,
        json_schema_extra={"example": 0.15},
        description="Freight to GMV ratio",
    )
    avg_delivery_delay: FiniteFloat = Field(
        ...,
        ge=-3650,
        le=3650,
        json_schema_extra={"example": 1.5},
        description="Average delivery delay in days",
    )
    max_delivery_delay: FiniteFloat = Field(
        ...,
        ge=-3650,
        le=3650,
        json_schema_extra={"example": 4.0},
        description="Max delivery delay in days",
    )
    avg_transit_days: FiniteFloat = Field(
        ...,
        ge=0,
        le=3650,
        json_schema_extra={"example": 10.5},
        description="Average total transit days",
    )
    avg_carrier_handoff: FiniteFloat = Field(
        ...,
        ge=0,
        le=3650,
        json_schema_extra={"example": 2.1},
        description="Carrier handoff delay in days",
    )
    has_comment_rate: FiniteFloat = Field(
        ...,
        ge=0,
        le=1,
        json_schema_extra={"example": 0.45},
        description="Proportion of orders with written review",
    )
    tenure_days: FiniteFloat = Field(
        ...,
        ge=0,
        le=100_000,
        json_schema_extra={"example": 180.0},
        description="Days since first seller order",
    )
    orders_per_month: FiniteFloat = Field(
        ...,
        ge=0,
        le=1_000_000,
        json_schema_extra={"example": 3.5},
        description="Fulfillment velocity per month",
    )


@app.get("/health")
@app.get("/ready")
def health_check():
    """Service health check."""
    ready = (
        champion_model is not None
        and seller_db is not None
        and not seller_db.empty
        and explainer is not None
    )
    if not ready:
        raise HTTPException(
            status_code=503,
            detail="Model and seller artifacts are unavailable. Run the pipeline with the required Olist data before serving.",
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "database_size": len(seller_db),
    }


@app.get("/api/summary")
def get_marketplace_summary():
    """Returns macro marketplace seller distribution and risk metrics."""
    if seller_db is None:
        raise HTTPException(status_code=503, detail="Database not ready.")
    if seller_db.empty:
        raise HTTPException(status_code=503, detail="Seller database is empty.")

    total = len(seller_db)
    at_risk = int(seller_db["is_at_risk"].sum())
    avg_score = float(seller_db["quality_score"].mean())
    avg_review = float(seller_db["mean_review_score"].mean())

    return {
        "total_active_sellers": total,
        "at_risk_sellers_count": at_risk,
        "at_risk_percentage": round((at_risk / total) * 100, 2),
        "mean_quality_score": round(avg_score, 1),
        "mean_review_score": round(avg_review, 2),
    }


@app.get("/api/sellers/{seller_id}")
def get_seller_risk_profile(seller_id: str):
    """Fetches diagnostic risk score and recommended action for an existing seller."""
    require_scoring_artifacts()
    if seller_db is None:
        raise HTTPException(503, "Seller database unavailable.")
    if seller_id not in seller_db.index:
        raise HTTPException(status_code=404, detail=f"Seller {seller_id} not found.")

    row = seller_db.loc[seller_id]
    prob, explanation = predict_and_explain(row)
    is_at_risk = bool(prob >= 0.5)

    if prob < 0.25:
        segment = "Elite / Low Risk"
    elif prob < 0.50:
        segment = "Good / Stable"
    elif prob < 0.75:
        segment = "At-Risk / Watchlist"
    else:
        segment = "Critical / Severe Risk"

    return {
        **SCORE_SCOPE,
        "seller_id": seller_id,
        "risk_probability": round(prob, 4),
        "is_at_risk": is_at_risk,
        "segment": segment,
        "quality_score": float(row["quality_score"]),
        "actual_mean_review": float(row["mean_review_score"]),
        "actual_late_delivery_rate": round(float(row["late_delivery_rate"]) * 100, 1),
        "top_drivers": explanation["drivers"],
        "recommended_action": explanation["recommended_action"],
    }


@app.post("/api/score")
def score_seller_realtime(data: SellerMetricsInput):
    """Scores a new or modified seller profile in real time."""
    input_dict = data.model_dump()
    prob, explanation = predict_and_explain(input_dict)

    return {
        **SCORE_SCOPE,
        "risk_probability": round(prob, 4),
        "is_at_risk": bool(prob >= 0.5),
        "top_drivers": explanation["drivers"],
        "recommended_action": explanation["recommended_action"],
    }
