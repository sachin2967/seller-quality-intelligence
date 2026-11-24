"""Prespecified rolling study with matured labels and fixed operational baselines."""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import DataLoader, RawDataConfig
from src.evaluation.temporal import FEATURES, build_temporal_snapshots
from src.evaluation.validation import (
    cluster_bootstrap_ap,
    matured_training_rows,
    ranking_metrics,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "rolling_benchmark_report.json"
    )
    args = parser.parse_args()
    config = RawDataConfig.from_directory(args.data_dir)
    items = DataLoader(config).build_joined_item_level_dataset()
    cutoffs = list(pd.date_range("2017-01-01", "2018-04-01", freq="MS"))
    snapshots = build_temporal_snapshots(items, cutoffs)
    windows, skipped, final_predictions = [], [], []
    for cutoff in cutoffs:
        train = matured_training_rows(snapshots, cutoff)
        test = snapshots[snapshots.cutoff == cutoff].sort_values("seller_id")
        if len(train) < 50 or train.future_at_risk.nunique() < 2 or len(test) < 25:
            skipped.append(
                {
                    "cutoff": str(cutoff.date()),
                    "train_rows": len(train),
                    "test_rows": len(test),
                }
            )
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        )
        model.fit(train[FEATURES], train.future_at_risk)
        scores = model.predict_proba(test[FEATURES])[:, 1]
        methods = {
            "fixed_logistic": scores,
            "historical_late_rate": test.late_rate_lookback.to_numpy(),
            "historical_low_review": -test.mean_review_lookback.to_numpy(),
            "training_prevalence": [train.future_at_risk.mean()] * len(test),
        }
        future_orders = items[
            (items.order_purchase_timestamp >= cutoff)
            & (items.order_purchase_timestamp < cutoff + pd.Timedelta(days=90))
        ]
        windows.append(
            {
                "cutoff": str(cutoff.date()),
                "train_rows": len(train),
                "latest_training_label_date": str(
                    train.outcome_observation_end.max().date()
                ),
                "train_unique_sellers": int(train.seller_id.nunique()),
                "test_unique_sellers": int(test.seller_id.nunique()),
                "sellers_with_future_purchases": int(future_orders.seller_id.nunique()),
                "evaluated_sellers": len(test),
                "methods": {
                    name: ranking_metrics(test.future_at_risk, values)
                    for name, values in methods.items()
                },
                "logistic_ap_uncertainty": cluster_bootstrap_ap(
                    test.future_at_risk, scores, test.seller_id
                ),
            }
        )
        final_predictions = [
            {"seller": seller, "label": int(label), "score": float(score)}
            for seller, label, score in zip(test.seller_id, test.future_at_risk, scores)
        ]
    report = {
        "protocol_version": "rolling-matured-v1",
        "python": platform.python_version(),
        "raw_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in vars(config).values()
        },
        "protocol": "Monthly expanding training; 90-day purchase + 30-day grace; train labels mature by each cutoff; fixed logistic and baselines, no test tuning; 10% review capacity.",
        "scope": "Historical observational study; all windows are exploratory after prior dataset inspection.",
        "windows": windows,
        "skipped_windows": skipped,
        "latest_window_predictions_count": len(final_predictions),
        "limitations": [
            "Overlapping windows and repeated sellers make pooled row counts non-independent; no pooled significance claim.",
            "Sellers need two completed historical orders and an observed future delivery; missing and censored outcomes remain excluded.",
            "Bootstrap intervals quantify seller sampling within each window, not temporal generalization.",
            "Scores are uncalibrated; no intervention, realized savings, prospective validation or automatic model promotion.",
            "Constant-score baseline capacity uses seller-ID tie order; expected random precision equals prevalence.",
        ],
    }
    output = args.output
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "windows": len(windows),
                "output": str(output),
                "latest": windows[-1] if windows else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
