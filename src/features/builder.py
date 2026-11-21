"""Feature engineering module for Seller Quality Intelligence.

Aggregates item-level transactional records to seller-level behavioral profiles,
computes RFM velocity, SLA breach rates, sentiment indicators, and constructs
both classification and survival analysis target labels.
"""

import numpy as np
import pandas as pd


class SellerFeatureBuilder:
    """Builds comprehensive seller-level feature vectors and target outcomes."""

    def __init__(self, min_orders_threshold: int = 2):
        """
        Args:
            min_orders_threshold: Minimum orders required to evaluate a seller reliably.
        """
        self.min_orders_threshold = min_orders_threshold

    def build_features(self, item_df: pd.DataFrame) -> pd.DataFrame:
        """Transforms item-level records into seller-level features.

        Args:
            item_df: Joined DataFrame output from DataLoader.build_joined_item_level_dataset()

        Returns:
            DataFrame with one row per seller and full-history aggregate features + retrospective targets.
        """
        # Ensure chronological ordering
        df = item_df.sort_values("order_purchase_timestamp").copy()

        # Group by seller_id
        seller_groups = df.groupby("seller_id")

        features_list = []

        for seller_id, group in seller_groups:
            total_items = len(group)
            unique_orders = group["order_id"].nunique()

            if unique_orders < self.min_orders_threshold:
                continue

            # Financial & Volume features
            total_gmv = group["price"].sum()
            avg_item_price = group["price"].mean()
            std_item_price = group["price"].std(ddof=0)
            avg_freight = group["freight_value"].mean()
            freight_ratio = group["freight_value"].sum() / max(total_gmv, 1.0)
            unique_products = group["product_id"].nunique()

            # Delivery & SLA features
            late_mask = group["is_late_delivery"] == 1.0
            late_delivery_rate = late_mask.mean()
            avg_delivery_delay = (
                group["delivery_delay_days"].dropna().mean()
                if not group["delivery_delay_days"].dropna().empty
                else 0.0
            )
            max_delivery_delay = (
                group["delivery_delay_days"].dropna().max()
                if not group["delivery_delay_days"].dropna().empty
                else 0.0
            )
            avg_transit_days = (
                group["transit_days"].dropna().mean()
                if not group["transit_days"].dropna().empty
                else 0.0
            )
            avg_carrier_handoff = (
                group["carrier_handoff_days"].dropna().mean()
                if not group["carrier_handoff_days"].dropna().empty
                else 0.0
            )

            # Quality & Review features
            reviews = group["review_score"].dropna()
            mean_review = reviews.mean() if not reviews.empty else 3.0
            one_star_rate = (reviews == 1.0).mean() if not reviews.empty else 0.0
            five_star_rate = (reviews == 5.0).mean() if not reviews.empty else 0.0
            has_comment_rate = group["review_comment_message"].notna().mean()

            # Cancellation & Status features
            canceled_orders = (group["order_status"] == "canceled").sum()
            cancellation_rate = canceled_orders / max(total_items, 1)

            # Temporal / Tenure features
            first_order = group["order_purchase_timestamp"].min()
            last_order = group["order_purchase_timestamp"].max()
            tenure_days = max((last_order - first_order).total_seconds() / 86400.0, 1.0)
            orders_per_month = (unique_orders / max(tenure_days, 1.0)) * 30.0

            # Survival Analysis targets: Time to first severe degradation (1-star review or >5d delay)
            breach_events = group[
                (group["review_score"] == 1.0) | (group["delivery_delay_days"] > 5.0)
            ]
            if not breach_events.empty:
                event_occurred = 1
                first_breach_time = breach_events["order_purchase_timestamp"].min()
                time_to_breach_days = max(
                    (first_breach_time - first_order).total_seconds() / 86400.0, 0.5
                )
            else:
                event_occurred = 0
                time_to_breach_days = tenure_days

            # Classification target: At-Risk seller definition
            # A seller is classified as 'At-Risk' if their mean review < 3.8 OR late rate > 15% OR cancellation > 5%
            is_at_risk = int(
                (mean_review < 3.8)
                or (late_delivery_rate > 0.15)
                or (cancellation_rate > 0.05)
            )

            # Composite Seller Quality Score (0-100)
            # 40% Review performance + 40% On-time delivery + 20% Fulfillment volume
            review_component = np.clip((mean_review - 1.0) / 4.0, 0.0, 1.0) * 40.0
            ontime_component = np.clip(1.0 - late_delivery_rate, 0.0, 1.0) * 40.0
            volume_component = np.clip(np.log1p(total_gmv) / 10.0, 0.0, 1.0) * 20.0
            quality_score = round(
                review_component + ontime_component + volume_component, 1
            )

            features_list.append(
                {
                    "seller_id": seller_id,
                    "total_gmv": total_gmv,
                    "total_items": total_items,
                    "unique_orders": unique_orders,
                    "unique_products": unique_products,
                    "avg_item_price": avg_item_price,
                    "std_item_price": std_item_price,
                    "avg_freight": avg_freight,
                    "freight_ratio": freight_ratio,
                    "late_delivery_rate": late_delivery_rate,
                    "avg_delivery_delay": avg_delivery_delay,
                    "max_delivery_delay": max_delivery_delay,
                    "avg_transit_days": avg_transit_days,
                    "avg_carrier_handoff": avg_carrier_handoff,
                    "mean_review_score": mean_review,
                    "one_star_rate": one_star_rate,
                    "five_star_rate": five_star_rate,
                    "has_comment_rate": has_comment_rate,
                    "cancellation_rate": cancellation_rate,
                    "tenure_days": tenure_days,
                    "orders_per_month": orders_per_month,
                    "quality_score": quality_score,
                    # Survival targets
                    "survival_duration": time_to_breach_days,
                    "survival_event": event_occurred,
                    # Classification target
                    "is_at_risk": is_at_risk,
                }
            )

        seller_df = pd.DataFrame(features_list)
        return seller_df

    def get_feature_matrix(
        self, seller_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Separates features, classification target, and survival target.

        Returns:
            X: Feature matrix
            y: Binary target (is_at_risk)
            survival_df: DataFrame with ['survival_duration', 'survival_event']
        """
        exclude_cols = [
            "seller_id",
            "is_at_risk",
            "quality_score",
            "survival_duration",
            "survival_event",
            # Exclude direct leakage columns that form the target definition
            "mean_review_score",
            "one_star_rate",
            "five_star_rate",
            "late_delivery_rate",
            "cancellation_rate",
        ]

        feature_cols = [c for c in seller_df.columns if c not in exclude_cols]
        X = seller_df[feature_cols].copy()
        y = seller_df["is_at_risk"].copy()
        survival_df = seller_df[["survival_duration", "survival_event"]].copy()

        return X, y, survival_df
