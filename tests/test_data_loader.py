"""Unit tests for the data loading and join validation module."""

from pathlib import Path

import pytest

from src.data.loader import DataLoader, RawDataConfig


@pytest.fixture
def data_config():
    root = Path(__file__).resolve().parent.parent / "data" / "raw"
    return RawDataConfig.from_directory(root)


def test_data_loader_raw_files_exist(data_config):
    loader = DataLoader(data_config)
    orders, items, reviews, sellers = loader.load_raw()

    assert not orders.empty
    assert not items.empty
    assert not reviews.empty
    assert not sellers.empty
    assert "order_id" in orders.columns
    assert "seller_id" in items.columns


def test_joined_item_level_dataset(data_config):
    loader = DataLoader(data_config)
    joined_df = loader.build_joined_item_level_dataset()

    assert not joined_df.empty
    assert "is_late_delivery" in joined_df.columns
    assert "delivery_delay_days" in joined_df.columns
    assert "carrier_handoff_days" in joined_df.columns
    # Check that is_late_delivery is binary (0.0 or 1.0)
    assert set(joined_df["is_late_delivery"].dropna().unique()).issubset({0.0, 1.0})
