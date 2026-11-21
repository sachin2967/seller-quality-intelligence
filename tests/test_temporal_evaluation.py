import pandas as pd

from src.evaluation.temporal import build_temporal_snapshots


def test_temporal_features_stop_at_cutoff_and_label_uses_future_orders_only():
    cutoff = pd.Timestamp("2020-02-01")
    rows = [
        # Two completed history orders; a high rating is observed before cutoff.
        {
            "seller_id": "s1",
            "order_id": "h1",
            "order_purchase_timestamp": "2020-01-01",
            "order_delivered_customer_date": "2020-01-03",
            "order_estimated_delivery_date": "2020-01-04",
            "review_answer_timestamp": "2020-01-05",
            "review_score": 5,
            "price": 10,
            "order_status": "delivered",
            "is_late_delivery": 0,
            "transit_days": 2,
        },
        {
            "seller_id": "s1",
            "order_id": "h2",
            "order_purchase_timestamp": "2020-01-10",
            "order_delivered_customer_date": "2020-01-12",
            "order_estimated_delivery_date": "2020-01-14",
            "review_answer_timestamp": "2020-01-15",
            "review_score": 5,
            "price": 20,
            "order_status": "delivered",
            "is_late_delivery": 0,
            "transit_days": 2,
        },
        # Post-cutoff review is hidden from the features but is an outcome signal.
        {
            "seller_id": "s1",
            "order_id": "f1",
            "order_purchase_timestamp": "2020-02-02",
            "order_delivered_customer_date": "2020-02-05",
            "order_estimated_delivery_date": "2020-02-04",
            "review_answer_timestamp": "2020-02-06",
            "review_score": 1,
            "price": 30,
            "order_status": "delivered",
            "is_late_delivery": 1,
            "transit_days": 3,
        },
    ]
    snapshot = build_temporal_snapshots(pd.DataFrame(rows), [cutoff], horizon_days=30)
    assert len(snapshot) == 1
    row = snapshot.iloc[0]
    assert row.orders_lookback == 2
    assert row.mean_review_lookback == 5
    assert row.gmv_lookback == 30
    assert row.future_orders == 1
    assert row.future_at_risk == 1


def test_review_with_unknown_timestamp_is_not_used_as_feature():
    cutoff = pd.Timestamp("2020-02-01")
    rows = []
    for order_id, day in [("h1", "2020-01-01"), ("h2", "2020-01-10")]:
        rows.append(
            {
                "seller_id": "s",
                "order_id": order_id,
                "order_purchase_timestamp": day,
                "order_delivered_customer_date": day,
                "order_estimated_delivery_date": day,
                "review_answer_timestamp": None,
                "review_score": 1,
                "price": 1,
                "order_status": "delivered",
                "is_late_delivery": 0,
                "transit_days": 0,
            }
        )
    rows.append(
        {
            "seller_id": "s",
            "order_id": "f",
            "order_purchase_timestamp": "2020-02-02",
            "order_delivered_customer_date": "2020-02-03",
            "order_estimated_delivery_date": "2020-02-03",
            "review_answer_timestamp": None,
            "review_score": None,
            "price": 1,
            "order_status": "delivered",
            "is_late_delivery": 0,
            "transit_days": 1,
        }
    )
    snapshot = build_temporal_snapshots(pd.DataFrame(rows), [cutoff], horizon_days=30)
    assert snapshot.iloc[0].mean_review_lookback == 3.0
    assert snapshot.iloc[0].future_at_risk == 0


def test_grace_period_includes_delayed_delivery_outcome_after_purchase_window():
    cutoff = pd.Timestamp("2020-02-01")
    rows = []
    for order_id, day in [("h1", "2020-01-01"), ("h2", "2020-01-10")]:
        rows.append(
            {
                "seller_id": "s",
                "order_id": order_id,
                "order_purchase_timestamp": day,
                "order_delivered_customer_date": day,
                "order_estimated_delivery_date": day,
                "review_answer_timestamp": day,
                "review_score": 5,
                "price": 1,
                "order_status": "delivered",
                "is_late_delivery": 0,
                "transit_days": 0,
            }
        )
    rows.append(
        {
            "seller_id": "s",
            "order_id": "f",
            "order_purchase_timestamp": "2020-04-20",
            "order_delivered_customer_date": "2020-05-12",
            "order_estimated_delivery_date": "2020-05-10",
            "review_answer_timestamp": "2020-05-14",
            "review_score": 5,
            "price": 1,
            "order_status": "delivered",
            "is_late_delivery": 1,
            "transit_days": 22,
        }
    )
    snapshot = build_temporal_snapshots(
        pd.DataFrame(rows), [cutoff], horizon_days=90, outcome_grace_days=30
    )
    assert len(snapshot) == 1
    assert snapshot.iloc[0].future_at_risk == 1
