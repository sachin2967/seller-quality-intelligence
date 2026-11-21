"""Evaluation metrics module for Seller Quality Intelligence.

Computes decision-focused business metrics including Precision@K,
Financial Cost-Benefit analysis, and probability calibration diagnostics.
"""

from typing import Dict, List
import numpy as np
import pandas as pd


def compute_precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k_pct: float = 0.1) -> float:
    """Computes Precision among the top k% highest-risk predicted sellers."""
    n = len(y_true)
    k_count = max(int(n * k_pct), 1)
    top_k_indices = np.argsort(y_prob)[::-1][:k_count]
    return float(np.mean(y_true[top_k_indices]))


def compute_cost_benefit_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    churn_loss_saved: float = 500.0,
    inspection_cost: float = 80.0,
    missed_breach_penalty: float = 750.0,
) -> Dict[str, float]:
    """Evaluates the financial net value of the model vs naive baseline.

    Args:
        y_true: Binary ground truth labels (1 = at-risk, 0 = safe)
        y_prob: Predicted risk probabilities
        threshold: Decision threshold for intervention
        churn_loss_saved: Net savings when an at-risk seller is caught and fixed
        inspection_cost: Operational cost of reviewing a flagged seller
        missed_breach_penalty: Financial damage of an unmitigated seller SLA breach

    Returns:
        Dictionary with confusion counts and net business savings.
    """
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    # Net value = (Savings from caught sellers) - (Ops inspection costs) - (Penalties from missed breaches)
    model_net_value = (tp * churn_loss_saved) - ((tp + fp) * inspection_cost) - (fn * missed_breach_penalty)
    # Naive baseline: No model, all breaches missed
    naive_net_value = -(np.sum(y_true) * missed_breach_penalty)
    incremental_value = model_net_value - naive_net_value

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "model_net_value": float(model_net_value),
        "naive_net_value": float(naive_net_value),
        "incremental_roi_value": float(incremental_value),
        "precision_at_top_10pct": compute_precision_at_k(y_true, y_prob, 0.10),
        "precision_at_top_20pct": compute_precision_at_k(y_true, y_prob, 0.20),
    }
