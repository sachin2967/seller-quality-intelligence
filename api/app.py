"""FastAPI demonstration service for the Seller Quality Intelligence prototype.

Exposes retrospective demo scoring, individual seller lookups, survival estimates,
and merchant operations interventions.
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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

# Load artifacts into memory
try:
    classifier_artifact = joblib.load(MODELS_DIR / "lgbm_seller_risk.joblib")
    champion_model = classifier_artifact["model"]
    feature_names = classifier_artifact["feature_names"]

    survival_model = joblib.load(MODELS_DIR / "seller_survival_cox.joblib")
    explainer = joblib.load(MODELS_DIR / "seller_explainer.joblib")

    # Load precomputed seller feature database
    seller_db = pd.read_csv(DATA_DIR / "seller_features.csv")
    seller_db.set_index("seller_id", inplace=True, drop=False)
    print("✅ All ML models and seller database loaded successfully.")
except (FileNotFoundError, OSError, EOFError, KeyError, pd.errors.EmptyDataError) as e:
    print(f"⚠️ Warning during artifact load: {e}")
    champion_model = None
    seller_db = None


class SellerMetricsInput(BaseModel):
    """Raw seller features for retrospective demonstration scoring."""

    total_gmv: float = Field(
        ..., json_schema_extra={"example": 4500.0}, description="Total GMV in currency"
    )
    total_items: int = Field(
        ..., json_schema_extra={"example": 25}, description="Total items sold"
    )
    unique_orders: int = Field(
        ..., json_schema_extra={"example": 20}, description="Unique orders fulfilled"
    )
    unique_products: int = Field(
        ..., json_schema_extra={"example": 8}, description="Distinct product SKUs"
    )
    avg_item_price: float = Field(
        ...,
        json_schema_extra={"example": 180.0},
        description="Average item selling price",
    )
    std_item_price: float = Field(
        ...,
        json_schema_extra={"example": 45.0},
        description="Standard deviation of item price",
    )
    avg_freight: float = Field(
        ...,
        json_schema_extra={"example": 22.0},
        description="Average shipping freight cost",
    )
    freight_ratio: float = Field(
        ..., json_schema_extra={"example": 0.15}, description="Freight to GMV ratio"
    )
    avg_delivery_delay: float = Field(
        ...,
        json_schema_extra={"example": 1.5},
        description="Average delivery delay in days",
    )
    max_delivery_delay: float = Field(
        ...,
        json_schema_extra={"example": 4.0},
        description="Max delivery delay in days",
    )
    avg_transit_days: float = Field(
        ...,
        json_schema_extra={"example": 10.5},
        description="Average total transit days",
    )
    avg_carrier_handoff: float = Field(
        ...,
        json_schema_extra={"example": 2.1},
        description="Carrier handoff delay in days",
    )
    has_comment_rate: float = Field(
        ...,
        json_schema_extra={"example": 0.45},
        description="Proportion of orders with written review",
    )
    tenure_days: float = Field(
        ...,
        json_schema_extra={"example": 180.0},
        description="Days since first seller order",
    )
    orders_per_month: float = Field(
        ...,
        json_schema_extra={"example": 3.5},
        description="Fulfillment velocity per month",
    )


@app.get("/health")
def health_check():
    """Service health check."""
    return {
        "status": "healthy",
        "model_loaded": champion_model is not None,
        "database_size": len(seller_db) if seller_db is not None else 0,
    }


@app.get("/api/summary")
def get_marketplace_summary():
    """Returns macro marketplace seller distribution and risk metrics."""
    if seller_db is None:
        raise HTTPException(status_code=503, detail="Database not ready.")

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
    if seller_db is None or seller_id not in seller_db.index:
        raise HTTPException(status_code=404, detail=f"Seller {seller_id} not found.")

    row = seller_db.loc[seller_id]
    X_row = row[feature_names].values.reshape(1, -1)

    prob = float(champion_model.predict_proba(X_row)[0, 1])
    is_at_risk = bool(prob >= 0.5)

    if prob < 0.25:
        segment = "Elite / Low Risk"
    elif prob < 0.50:
        segment = "Good / Stable"
    elif prob < 0.75:
        segment = "At-Risk / Watchlist"
    else:
        segment = "Critical / Severe Risk"

    explanation = explainer.explain_seller(row)

    return {
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
    if champion_model is None:
        raise HTTPException(status_code=503, detail="Model artifact unavailable.")

    input_dict = data.model_dump()
    X_vals = [input_dict[f] for f in feature_names]
    prob = float(champion_model.predict_proba([X_vals])[0, 1])

    series = pd.Series(input_dict)
    explanation = explainer.explain_seller(series)

    return {
        "risk_probability": round(prob, 4),
        "is_at_risk": bool(prob >= 0.5),
        "top_drivers": explanation["drivers"],
        "recommended_action": explanation["recommended_action"],
    }
