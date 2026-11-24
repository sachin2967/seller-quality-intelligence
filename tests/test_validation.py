import numpy as np
import pandas as pd
import pytest

from src.evaluation.validation import (
    cluster_bootstrap_ap,
    matured_training_rows,
    ranking_metrics,
)


def test_training_purges_unavailable_labels_even_with_earlier_cutoffs():
    data = pd.DataFrame(
        {
            "cutoff": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-05-01"]),
            "outcome_observation_end": pd.to_datetime(
                ["2020-04-30", "2020-05-31", "2020-05-01"]
            ),
        }
    )
    assert matured_training_rows(data, pd.Timestamp("2020-05-01")).index.tolist() == [0]


def test_capacity_has_exact_budget_and_uses_scores():
    result = ranking_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], fraction=0.25)
    assert result["review_count"] == 1
    assert result["precision_at_capacity"] == 1
    assert result["recall_at_capacity"] == 0.5


def test_bootstrap_keeps_whole_seller_groups_and_handles_single_class():
    result = cluster_bootstrap_ap(
        [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], ["a", "a", "b", "b"], repetitions=100
    )
    assert 0 < result["valid_repetitions"] < 100
    assert result["ap_95_percentile_interval"] == [1.0, 1.0]
    assert (
        cluster_bootstrap_ap([1, 1], [0.1, 0.9], ["a", "b"], repetitions=10)[
            "ap_95_percentile_interval"
        ]
        is None
    )


@pytest.mark.parametrize("scores", [[np.nan, 0.1], [0.1]])
def test_invalid_scores_do_not_produce_metrics(scores):
    with pytest.raises(ValueError):
        ranking_metrics([0, 1], scores)


def test_fractional_labels_are_rejected_before_coercion():
    with pytest.raises(ValueError):
        ranking_metrics([0, 1.5], [0.1, 0.9])
