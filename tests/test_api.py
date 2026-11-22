"""API contract tests use isolated doubles and never require private data/artifacts."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.app as api_module

client = TestClient(api_module.app)


class StubModel:
    def predict_proba(self, _values):
        return np.array([[0.35, 0.65]])


class StubExplainer:
    def explain_seller(self, _row):
        return {"drivers": [], "recommended_action": {"action": "review"}}


@pytest.fixture(autouse=True)
def loaded_demo(monkeypatch):
    row = {
        "seller_id": "seller-1",
        "is_at_risk": 1,
        "quality_score": 60.0,
        "mean_review_score": 3.1,
        "late_delivery_rate": 0.2,
        "total_gmv": 4500.0,
        "total_items": 25,
        "unique_orders": 20,
        "unique_products": 8,
        "avg_item_price": 180.0,
        "std_item_price": 45.0,
        "avg_freight": 22.0,
        "freight_ratio": 0.15,
        "avg_delivery_delay": 3.5,
        "max_delivery_delay": 7.0,
        "avg_transit_days": 14.5,
        "avg_carrier_handoff": 4.1,
        "has_comment_rate": 0.45,
        "tenure_days": 180.0,
        "orders_per_month": 3.5,
    }
    db = pd.DataFrame([row]).set_index("seller_id", drop=False)
    monkeypatch.setattr(api_module, "champion_model", StubModel())
    monkeypatch.setattr(
        api_module,
        "feature_names",
        [
            "total_gmv",
            "total_items",
            "unique_orders",
            "unique_products",
            "avg_item_price",
            "std_item_price",
            "avg_freight",
            "freight_ratio",
            "avg_delivery_delay",
            "max_delivery_delay",
            "avg_transit_days",
            "avg_carrier_handoff",
            "has_comment_rate",
            "tenure_days",
            "orders_per_month",
        ],
    )
    monkeypatch.setattr(api_module, "explainer", StubExplainer())
    monkeypatch.setattr(api_module, "seller_db", db)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
        "database_size": 1,
    }


def test_health_is_not_ready_without_artifacts(monkeypatch):
    monkeypatch.setattr(api_module, "explainer", None)
    response = client.get("/health")
    assert response.status_code == 503


def test_marketplace_summary():
    response = client.get("/api/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_active_sellers"] == 1
    assert data["at_risk_percentage"] == 100.0


def test_seller_profile_lookup():
    response = client.get("/api/sellers/seller-1")
    assert response.status_code == 200
    assert response.json()["seller_id"] == "seller-1"


def test_seller_profile_not_found():
    assert client.get("/api/sellers/missing").status_code == 404


def test_score_seller_realtime():
    sample_payload = {
        "total_gmv": 4500.0,
        "total_items": 25,
        "unique_orders": 20,
        "unique_products": 8,
        "avg_item_price": 180.0,
        "std_item_price": 45.0,
        "avg_freight": 22.0,
        "freight_ratio": 0.15,
        "avg_delivery_delay": 3.5,
        "max_delivery_delay": 7.0,
        "avg_transit_days": 14.5,
        "avg_carrier_handoff": 4.1,
        "has_comment_rate": 0.45,
        "tenure_days": 180.0,
        "orders_per_month": 3.5,
    }
    response = client.post("/api/score", json=sample_payload)
    assert response.status_code == 200
    assert response.json()["risk_probability"] == 0.65


def test_score_rejects_extra_and_invalid_values():
    sample = {name: 1 for name in api_module.feature_names}
    sample["total_items"] = -1
    assert client.post("/api/score", json=sample).status_code == 422
    sample["total_items"] = 1
    sample["unexpected"] = 1
    assert client.post("/api/score", json=sample).status_code == 422


def test_partial_dependencies_do_not_serve_scores(monkeypatch):
    monkeypatch.setattr(api_module, "explainer", None)
    sample = {name: 1 for name in api_module.feature_names}
    assert client.post("/api/score", json=sample).status_code == 503
    assert client.get("/api/sellers/seller-1").status_code == 503
    assert client.get("/live").status_code == 200


def test_missing_database_is_unavailable_not_unknown_seller(monkeypatch):
    monkeypatch.setattr(api_module, "seller_db", None)
    assert client.get("/api/sellers/seller-1").status_code == 503


def test_empty_database_does_not_divide_by_zero(monkeypatch):
    monkeypatch.setattr(api_module, "seller_db", pd.DataFrame())
    assert client.get("/api/summary").status_code == 503


def test_bad_model_output_and_broken_explanations_fail_closed(monkeypatch):
    sample = {name: 1 for name in api_module.feature_names}
    monkeypatch.setattr(
        api_module.champion_model,
        "predict_proba",
        lambda _: np.array([[0.0, float("nan")]]),
    )
    assert client.post("/api/score", json=sample).status_code == 503


def test_api_explains_retrospective_scope():
    sample = {name: 1 for name in api_module.feature_names}
    result = client.post("/api/score", json=sample).json()
    assert "retrospective" in result["evaluation_scope"]
    assert "uncalibrated" in result["score_interpretation"]


def test_operator_auth_and_production_key_requirement(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("SELLER_API_KEY", raising=False)
    assert client.get("/api/summary").status_code == 503
    assert client.get("/live").status_code == 200
    monkeypatch.setenv("SELLER_API_KEY", "x" * 32)
    assert client.get("/api/summary").status_code == 401
    assert (
        client.get("/api/summary", headers={"X-API-Key": "x" * 32}).status_code == 200
    )
