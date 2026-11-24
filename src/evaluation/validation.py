"""Availability-aware validation and seller-clustered uncertainty estimates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def matured_training_rows(snapshot: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Purge labels whose full observation window was unavailable at prediction."""
    as_of = pd.Timestamp(as_of)
    return snapshot[
        (snapshot.cutoff < as_of) & (snapshot.outcome_observation_end <= as_of)
    ].copy()


def ranking_metrics(y, scores, fraction: float = 0.1) -> dict:
    """A fixed-capacity policy; ties follow stable input order, with no label access."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if (
        y.ndim != 1
        or scores.shape != y.shape
        or not len(y)
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(scores).all()
        or not 0 < fraction <= 1
    ):
        raise ValueError(
            "Expected binary labels, aligned finite scores and a valid capacity."
        )
    count = max(1, int(np.ceil(len(y) * fraction)))
    chosen = np.argsort(-scores, kind="stable")[:count]
    both = len(np.unique(y)) == 2
    return {
        "rows": len(y),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, scores))
        if both
        else None,
        "roc_auc": float(roc_auc_score(y, scores)) if both else None,
        "review_capacity_fraction": fraction,
        "review_count": count,
        "precision_at_capacity": float(y[chosen].mean()),
        "recall_at_capacity": float(y[chosen].sum() / y.sum()) if y.sum() else None,
    }


def cluster_bootstrap_ap(
    y, scores, sellers, repetitions: int = 500, seed: int = 42
) -> dict:
    """Resample whole sellers, retaining multiplicity and all their snapshots."""
    y, scores, sellers = np.asarray(y), np.asarray(scores), np.asarray(sellers)
    ranking_metrics(y, scores)  # Validate before resampling.
    if sellers.shape != y.shape or repetitions < 2 or pd.isna(sellers).any():
        raise ValueError(
            "Expected aligned nonmissing seller IDs and at least two replicates."
        )
    unique = np.unique(sellers)
    groups = [np.flatnonzero(sellers == seller) for seller in unique]
    rng, values = np.random.default_rng(seed), []
    for _ in range(repetitions):
        indices = np.concatenate(
            [groups[i] for i in rng.integers(len(groups), size=len(groups))]
        )
        if len(np.unique(y[indices])) == 2:
            values.append(average_precision_score(y[indices], scores[indices]))
    return {
        "method": "seller-cluster percentile bootstrap; conditional on this historical period",
        "seed": seed,
        "requested_repetitions": repetitions,
        "valid_repetitions": len(values),
        "unique_sellers": len(unique),
        "ap_95_percentile_interval": np.quantile(values, [0.025, 0.975]).tolist()
        if len(values) >= 2
        else None,
        "limitations": "Not uncertainty over new time periods, missing outcomes or marketplace shifts.",
    }
