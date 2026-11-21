"""FastAPI microservice integration and route tests using TestClient."""

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["database_size"] > 0


def test_marketplace_summary():
    response = client.get("/api/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_active_sellers" in data
    assert "at_risk_percentage" in data
    assert data["total_active_sellers"] > 1000


def test_seller_profile_lookup_not_found():
    response = client.get("/api/sellers/non_existent_seller_id_99999")
    assert response.status_code == 404


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
    data = response.json()
    assert "risk_probability" in data
    assert "is_at_risk" in data
    assert "recommended_action" in data
    assert 0.0 <= data["risk_probability"] <= 1.0


def test_score_seller_validation_error():
    # Incomplete payload
    response = client.post("/api/score", json={"total_gmv": 100.0})
    assert response.status_code == 422
