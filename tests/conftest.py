import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is in sys.path for test discovery
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def data_config(tmp_path_factory):
    """Authored join fixture; never read or overwrite the operator's Olist files."""
    from src.data.loader import RawDataConfig

    directory = tmp_path_factory.mktemp("olist-contract")
    orders = pd.DataFrame(
        [
            ["o1", "delivered", "2020-01-01", "2020-01-02", "2020-01-04", "2020-01-03"],
            ["o2", "delivered", "2020-01-10", "2020-01-11", "2020-01-12", "2020-01-13"],
            ["o3", "delivered", "2020-01-02", "2020-01-03", "2020-01-05", "2020-01-06"],
            ["o4", "canceled", "2020-01-12", None, None, "2020-01-18"],
        ],
        columns=[
            "order_id",
            "order_status",
            "order_purchase_timestamp",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    orders["order_approved_at"] = orders["order_purchase_timestamp"]
    items = pd.DataFrame(
        [
            ["o1", 1, "s1", "p1", 100, 10],
            ["o1", 2, "s1", "p2", 50, 5],
            ["o2", 1, "s1", "p1", 30, 3],
            ["o3", 1, "s2", "p3", 80, 8],
            ["o4", 1, "s2", "p3", 20, 2],
        ],
        columns=[
            "order_id",
            "order_item_id",
            "seller_id",
            "product_id",
            "price",
            "freight_value",
        ],
    )
    items["shipping_limit_date"] = "2020-01-20"
    reviews = pd.DataFrame(
        [
            ["o1", 1, "superseded fixture review", "2020-01-05"],
            ["o1", 5, "latest fixture review", "2020-01-06"],
            ["o2", 5, None, "2020-01-14"],
            ["o3", 2, "fixture review", "2020-01-07"],
        ],
        columns=[
            "order_id",
            "review_score",
            "review_comment_message",
            "review_answer_timestamp",
        ],
    )
    reviews["review_creation_date"] = reviews["review_answer_timestamp"]
    sellers = pd.DataFrame({"seller_id": ["s1", "s2"], "seller_state": ["SP", "RJ"]})
    for frame, name in [
        (orders, "orders"),
        (items, "order_items"),
        (reviews, "order_reviews"),
        (sellers, "sellers"),
    ]:
        frame.to_csv(directory / f"olist_{name}_dataset.csv", index=False)
    return RawDataConfig.from_directory(directory)
