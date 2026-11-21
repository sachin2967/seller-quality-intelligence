"""Unit tests for the seller feature builder and target definition."""

from pathlib import Path
import pytest
from src.data.loader import DataLoader, RawDataConfig
from src.features.builder import SellerFeatureBuilder


@pytest.fixture
def item_dataset():
    root = Path(__file__).resolve().parent.parent / "data" / "raw"
    config = RawDataConfig.from_directory(root)
    loader = DataLoader(config)
    return loader.build_joined_item_level_dataset()


def test_feature_builder_output(item_dataset):
    builder = SellerFeatureBuilder(min_orders_threshold=2)
    seller_df = builder.build_features(item_dataset)

    assert not seller_df.empty
    assert "seller_id" in seller_df.columns
    assert "is_at_risk" in seller_df.columns
    assert "quality_score" in seller_df.columns
    assert "survival_duration" in seller_df.columns
    assert "survival_event" in seller_df.columns

    # Verify target is binary
    assert set(seller_df["is_at_risk"].unique()).issubset({0, 1})

    # Verify feature matrix separation
    X, y, surv = builder.get_feature_matrix(seller_df)
    assert len(X) == len(y) == len(surv)
    assert "is_at_risk" not in X.columns
    assert "mean_review_score" not in X.columns  # No direct target leakage
