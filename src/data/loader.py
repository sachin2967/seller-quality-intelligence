"""Data loading and validation module for the Seller Quality Intelligence platform.

Ensures proper join validation across orders, order items, reviews, and sellers
while preventing multi-item order double-counting.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd


@dataclass(frozen=True)
class RawDataConfig:
    """Paths to raw e-commerce CSV files."""
    orders_path: Path
    items_path: Path
    reviews_path: Path
    sellers_path: Path

    @classmethod
    def from_directory(cls, dir_path: str | Path) -> "RawDataConfig":
        base = Path(dir_path)
        return cls(
            orders_path=base / "olist_orders_dataset.csv",
            items_path=base / "olist_order_items_dataset.csv",
            reviews_path=base / "olist_order_reviews_dataset.csv",
            sellers_path=base / "olist_sellers_dataset.csv",
        )


class DataLoader:
    """Loads and validates raw marketplace tables."""

    def __init__(self, config: RawDataConfig):
        self.config = config

    def load_raw(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads all four raw datasets with proper date parsing."""
        orders = pd.read_csv(
            self.config.orders_path,
            parse_dates=[
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ],
        )

        items = pd.read_csv(
            self.config.items_path,
            parse_dates=["shipping_limit_date"],
        )

        reviews = pd.read_csv(
            self.config.reviews_path,
            parse_dates=["review_creation_date", "review_answer_timestamp"],
        )

        sellers = pd.read_csv(self.config.sellers_path)

        return orders, items, reviews, sellers

    def build_joined_item_level_dataset(self) -> pd.DataFrame:
        """Constructs an item-level joined dataset with strict grain controls.

        Grain: One row per order_item (identified by order_id + order_item_id).
        Review score is deduplicated per order before joining to avoid inflating
        review counts for multi-item orders.
        """
        orders, items, reviews, sellers = self.load_raw()

        # Step 1: Filter to delivered or cancelled orders only for clean outcome analysis
        valid_orders = orders[orders["order_status"].isin(["delivered", "canceled", "shipped"])].copy()

        # Step 2: Deduplicate reviews at order level (take the first review per order_id)
        dedup_reviews = reviews.sort_values("review_answer_timestamp").drop_duplicates(
            subset=["order_id"], keep="last"
        )[["order_id", "review_score", "review_comment_message", "review_answer_timestamp"]]

        # Step 3: Join items with orders (Inner join: we need order metadata for every item)
        items_with_orders = items.merge(
            valid_orders,
            on="order_id",
            how="inner",
        )

        # Step 4: Join reviews (Left join: some orders may lack explicit reviews)
        items_with_reviews = items_with_orders.merge(
            dedup_reviews,
            on="order_id",
            how="left",
        )

        # Step 5: Join seller geographic metadata
        full_df = items_with_reviews.merge(
            sellers,
            on="seller_id",
            how="left",
        )

        # Step 6: Derive item-level delivery performance metrics
        # Is order delivered later than estimated date?
        full_df["is_late_delivery"] = (
            full_df["order_delivered_customer_date"] > full_df["order_estimated_delivery_date"]
        ).astype(float)

        # Delivery delay in days (positive = late, negative = early)
        full_df["delivery_delay_days"] = (
            full_df["order_delivered_customer_date"] - full_df["order_estimated_delivery_date"]
        ).dt.total_seconds() / 86400.0

        # Transit duration from purchase to delivery
        full_df["transit_days"] = (
            full_df["order_delivered_customer_date"] - full_df["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400.0

        # Carrier handoff delay (carrier date vs purchase timestamp)
        full_df["carrier_handoff_days"] = (
            full_df["order_delivered_carrier_date"] - full_df["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400.0

        return full_df
