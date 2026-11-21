"""Build and evaluate a point-in-time 90-day seller outcome benchmark."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loader import DataLoader, RawDataConfig
from src.evaluation.temporal import FEATURES, build_temporal_snapshots


def main():
    root = Path(__file__).resolve().parent.parent
    items = DataLoader(
        RawDataConfig.from_directory(root / "data" / "raw")
    ).build_joined_item_level_dataset()
    cutoffs = list(pd.date_range("2017-01-01", "2018-05-01", freq="90D"))
    purchase_window_days = 90
    outcome_grace_days = 30
    snapshot = build_temporal_snapshots(
        items,
        cutoffs,
        horizon_days=purchase_window_days,
        outcome_grace_days=outcome_grace_days,
    )
    if snapshot.empty:
        raise RuntimeError("No eligible temporal snapshots; check raw data coverage.")
    train = snapshot[snapshot.cutoff < pd.Timestamp("2018-01-01")]
    test = snapshot[snapshot.cutoff >= pd.Timestamp("2018-01-01")]
    if train.future_at_risk.nunique() < 2 or test.future_at_risk.nunique() < 2:
        raise RuntimeError(
            "Temporal split lacks both outcome classes; do not report classification metrics."
        )
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    )
    model.fit(train[FEATURES], train.future_at_risk)
    prob = model.predict_proba(test[FEATURES])[:, 1]
    y = test.future_at_risk
    unseen = test[~test.seller_id.isin(train.seller_id)]
    unseen_prob = model.predict_proba(unseen[FEATURES])[:, 1] if len(unseen) else []
    report = {
        "dataset_source": "Olist Brazilian E-Commerce public dataset (local raw CSVs); observational, not synthetic",
        "outcome": "among orders purchased in the next 90 days, any observed delivery >0 days later than estimate or review score <=2; outcomes are observed for up to 30 additional days",
        "feature_cutoff": "only orders delivered by cutoff; reviews are available only when review answer timestamp <= cutoff; missing review timestamps are treated as unavailable",
        "cutoff_schedule": [str(pd.Timestamp(c).date()) for c in cutoffs],
        "train_cutoff_before": "2018-01-01",
        "test_cutoff_from": "2018-01-01",
        "purchase_window_days": purchase_window_days,
        "outcome_grace_days": outcome_grace_days,
        "outcome_observation_days_from_cutoff": purchase_window_days
        + outcome_grace_days,
        "feature_columns": FEATURES,
        "snapshot_count": len(snapshot),
        "unique_sellers": int(snapshot.seller_id.nunique()),
        "train_rows": len(train),
        "train_positive_rate": float(train.future_at_risk.mean()),
        "test_rows": len(test),
        "test_positive_rate": float(y.mean()),
        "test_roc_auc": float(roc_auc_score(y, prob)),
        "test_average_precision": float(average_precision_score(y, prob)),
        "test_brier_score": float(brier_score_loss(y, prob)),
        "test_prevalence_baseline_average_precision": float(y.mean()),
        "test_unique_sellers": int(test.seller_id.nunique()),
        "test_sellers_seen_in_train": int(test.seller_id.isin(train.seller_id).sum()),
        "unseen_seller_test_rows": len(unseen),
        "unseen_seller_metrics": (
            {
                "positive_rate": float(unseen.future_at_risk.mean()),
                "roc_auc": float(roc_auc_score(unseen.future_at_risk, unseen_prob)),
                "average_precision": float(
                    average_precision_score(unseen.future_at_risk, unseen_prob)
                ),
            }
            if len(unseen) and unseen.future_at_risk.nunique() == 2
            else None
        ),
        "scope": "retrospective temporal observational benchmark; repeated seller snapshots may occur across periods; no causal or ROI claim",
        "limitations": [
            "small local Olist extract, finite historical period, and sellers/orders are not independent",
            "outcomes include only orders with delivery recorded by the end of the 30-day grace period; incomplete or undelivered orders do not contribute, which can bias observed risk",
            "sellers with no delivered future order in the purchase window are omitted because their outcome is unknown",
            "review answer timestamp is used to gate historical review features; missing timestamps are treated as unavailable",
            "no calibration, decision threshold selection, prospective deployment, or intervention evaluation",
        ],
    }
    (root / "temporal_benchmark_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(processed_dir / "seller_temporal_snapshots.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
