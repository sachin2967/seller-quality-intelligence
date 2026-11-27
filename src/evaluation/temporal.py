"""Point-in-time seller snapshots for a bounded, retrospective temporal study.

This is an observational benchmark on Olist: it is not a deployed seller-risk
system. Features use completed orders available by the cutoff; outcomes use
orders purchased in the following horizon.
"""

from __future__ import annotations

import pandas as pd

FEATURES = [
    "orders_lookback",
    "items_lookback",
    "gmv_lookback",
    "avg_price_lookback",
    "late_rate_lookback",
    "mean_review_lookback",
    "transit_days_lookback",
    "tenure_days_lookback",
]


def build_temporal_snapshots(
    items: pd.DataFrame,
    cutoffs: list[pd.Timestamp],
    horizon_days: int = 90,
    outcome_grace_days: int = 30,
    min_history_orders: int = 2,
) -> pd.DataFrame:
    """Create seller/cutoff rows with no post-cutoff feature information.

    Outcomes use orders purchased in the next ``horizon_days``. Delivery and
    review outcomes may be observed through an additional grace period so late
    purchases near the end of the purchase window are not automatically
    discarded. Only completed deliveries observed by that date contribute.
    Sellers without a future delivered order are omitted because their outcome
    is unknown.
    """
    df = items.copy()
    for col in (
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "review_answer_timestamp",
    ):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    rows: list[dict] = []
    for cutoff in map(pd.Timestamp, cutoffs):
        purchase_window_end = cutoff + pd.Timedelta(days=horizon_days)
        outcome_observation_end = purchase_window_end + pd.Timedelta(
            days=outcome_grace_days
        )
        # Keep transactions with observed delivery by the cutoff. Review fields
        # are gated separately below so unavailable reviews do not hide orders.
        history = df[
            (df.order_purchase_timestamp < cutoff)
            & df.order_delivered_customer_date.notna()
            & (df.order_delivered_customer_date <= cutoff)
        ]
        # A missing review timestamp means its availability time is unknown;
        # retain the transaction but do not expose that review as a feature.
        history = history.copy()
        history.loc[
            history.review_answer_timestamp.isna()
            | (history.review_answer_timestamp > cutoff), "review_score"
        ] = pd.NA
        future = df[
            (df.order_purchase_timestamp >= cutoff)
            & (df.order_purchase_timestamp < purchase_window_end)
            & df.order_delivered_customer_date.notna()
            & (df.order_delivered_customer_date <= outcome_observation_end)
        ]
        future = future.copy()
        future.loc[
            future.review_answer_timestamp > outcome_observation_end, "review_score"
        ] = pd.NA
        future.loc[future.review_answer_timestamp.isna(), "review_score"] = pd.NA
        for seller_id, hist in history.groupby("seller_id", sort=False):
            n_orders = hist.order_id.nunique()
            if n_orders < min_history_orders:
                continue
            outcomes = future[future.seller_id == seller_id]
            n_future = outcomes.order_id.nunique()
            if not n_future:
                continue
            delay = (
                outcomes.order_delivered_customer_date
                - outcomes.order_estimated_delivery_date
            ).dt.total_seconds() / 86400
            rows.append(
                {
                    "seller_id": seller_id,
                    "cutoff": cutoff,
                    "purchase_window_end": purchase_window_end,
                    "outcome_observation_end": outcome_observation_end,
                    "orders_lookback": n_orders,
                    "items_lookback": len(hist),
                    "gmv_lookback": float(hist.price.sum()),
                    "avg_price_lookback": float(hist.price.mean()),
                    "late_rate_lookback": float((hist.is_late_delivery == 1).mean()),
                    "mean_review_lookback": float(hist.review_score.mean())
                    if hist.review_score.notna().any()
                    else 3.0,
                    "transit_days_lookback": float(hist.transit_days.mean()),
                    "tenure_days_lookback": max(
                        float(
                            (
                                cutoff - hist.order_purchase_timestamp.min()
                            ).total_seconds()
                            / 86400
                        ),
                        1.0,
                    ),
                    "future_orders": n_future,
                    "future_at_risk": int(
                        (
                            (delay > 0).fillna(False)
                            | (outcomes.review_score <= 2).fillna(False)
                        ).any()
                    ),
                }
            )
    return pd.DataFrame(rows)
