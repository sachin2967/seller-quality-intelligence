"""End-to-end training and evaluation script for Seller Quality Intelligence.

Executes data loading, retrospective seller-state classification CV, and an
exploratory in-sample survival fit. This dataset has no point-in-time snapshots,
so these results are not prospective prediction or business-impact evidence.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import joblib

from src.data.loader import DataLoader, RawDataConfig
from src.features.builder import SellerFeatureBuilder
from src.models.classifier import SellerRiskClassifier
from src.models.explainability import SellerExplainer
from src.models.survival import SellerSurvivalModel


def run_pipeline():
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("🚀 1. LOADING & VALIDATING MARKETPLACE DATA")
    print("==================================================")
    config = RawDataConfig.from_directory(raw_dir)
    loader = DataLoader(config)
    item_df = loader.build_joined_item_level_dataset()
    print(
        f"Joined item-level dataset: {item_df.shape[0]} rows across {item_df['seller_id'].nunique()} unique sellers."
    )

    print("\n==================================================")
    print("🛠️ 2. COMPUTING SELLER FEATURES & TARGET OUTCOMES")
    print("==================================================")
    builder = SellerFeatureBuilder(min_orders_threshold=2)
    seller_df = builder.build_features(item_df)
    processed_path = processed_dir / "seller_features.csv"
    seller_df.to_csv(processed_path, index=False)
    print(
        f"Aggregated seller profiles: {seller_df.shape[0]} sellers with {seller_df.shape[1]} metrics."
    )
    print(
        f"At-Risk sellers count: {seller_df['is_at_risk'].sum()} / {len(seller_df)} ({seller_df['is_at_risk'].mean() * 100:.1f}%)"
    )

    X, y, survival_df = builder.get_feature_matrix(seller_df)
    print(f"Feature matrix X shape: {X.shape}, features: {list(X.columns)}")

    print("\n==================================================")
    print("📊 3. EVALUATING BASELINE (LOGISTIC REGRESSION 5-FOLD CV)")
    print("==================================================")
    classifier = SellerRiskClassifier(random_state=42)
    lr_res = classifier.evaluate_baseline_lr(X, y, n_splits=5)
    print(f"Baseline LR ROC-AUC: {lr_res.mean_roc_auc:.4f} ± {lr_res.std_roc_auc:.4f}")
    print(f"Baseline LR PR-AUC:  {lr_res.mean_pr_auc:.4f} ± {lr_res.std_pr_auc:.4f}")
    print(f"Baseline LR F1:      {lr_res.mean_f1:.4f}")
    print(f"Baseline LR Brier:   {lr_res.mean_brier:.4f}")

    print("\n==================================================")
    print("🏆 4. TRAINING CHAMPION (LIGHTGBM 5-FOLD CV)")
    print("==================================================")
    lgbm_res, _final_model = classifier.train_champion_lgbm(X, y, n_splits=5)
    print(
        f"Champion LightGBM ROC-AUC: {lgbm_res.mean_roc_auc:.4f} ± {lgbm_res.std_roc_auc:.4f}"
    )
    print(
        f"Champion LightGBM PR-AUC:  {lgbm_res.mean_pr_auc:.4f} ± {lgbm_res.std_pr_auc:.4f}"
    )
    print(f"Champion LightGBM F1:      {lgbm_res.mean_f1:.4f}")
    print(f"Champion LightGBM Brier:   {lgbm_res.mean_brier:.4f}")

    # Save classifier
    classifier.save(models_dir / "lgbm_seller_risk.joblib")

    print("\nTop 5 Most Important Features in LightGBM:")
    for _, row in lgbm_res.feature_importances.head(5).iterrows():
        print(f"  - {row['feature']}: {row['importance']:.2f}")

    print("\n==================================================")
    print("⏱️ 5. FITTING SURVIVAL ANALYSIS (COX PROPORTIONAL HAZARDS)")
    print("==================================================")
    survival_model = SellerSurvivalModel(penalizer=0.1)
    surv_summary = survival_model.fit_cox_ph(
        X,
        durations=survival_df["survival_duration"],
        events=survival_df["survival_event"],
    )
    survival_model.save(models_dir / "seller_survival_cox.joblib")
    print(
        f"Cox PH Concordance Index (C-Index): {surv_summary['concordance_index']:.4f}"
    )
    print(
        f"Sample size: {surv_summary['sample_size']}, SLA breach events: {surv_summary['events_observed']}"
    )

    hazard_df = survival_model.get_hazard_ratios()
    print(
        "\nExploratory apparent in-sample hazard associations (not held-out effects):"
    )
    for idx, row in hazard_df.head(4).iterrows():
        print(
            f"  - {idx}: HR = {row['hazard_ratio']:.3f} (apparent p={row['p_value']:.4f})"
        )

    print("\n==================================================")
    print("💰 6. BUSINESS IMPACT")
    print("==================================================")
    print(
        "Skipped: no temporal holdout or intervention study supports a financial ROI estimate."
    )

    # Fit and save explainability baseline
    explainer = SellerExplainer()
    explainer.fit_reference(X)
    joblib.dump(explainer, models_dir / "seller_explainer.joblib")

    # Record benchmark results
    benchmark_report = {
        "dataset": {
            "total_sellers": len(seller_df),
            "at_risk_sellers": int(seller_df["is_at_risk"].sum()),
            "at_risk_prevalence": float(seller_df["is_at_risk"].mean()),
        },
        "models": {
            "baseline_logistic_regression": {
                "roc_auc": lr_res.mean_roc_auc,
                "roc_auc_std": lr_res.std_roc_auc,
                "pr_auc": lr_res.mean_pr_auc,
                "f1_score": lr_res.mean_f1,
                "brier_score": lr_res.mean_brier,
            },
            "champion_lightgbm": {
                "roc_auc": lgbm_res.mean_roc_auc,
                "roc_auc_std": lgbm_res.std_roc_auc,
                "pr_auc": lgbm_res.mean_pr_auc,
                "f1_score": lgbm_res.mean_f1,
                "brier_score": lgbm_res.mean_brier,
            },
            "survival_cox_ph": {
                "concordance_index": surv_summary["concordance_index"],
                "events_observed": surv_summary["events_observed"],
                "evaluation_scope": "apparent_in_sample_full_history_association_only",
            },
        },
        "evaluation_scope": {
            "classification": "stratified_random_seller_cv_on_retrospective_full_history_rule_label",
            "prospective_prediction_validated": False,
            "financial_impact_validated": False,
            "limitations": [
                "features and labels summarize overlapping seller histories",
                "cross-validation does not simulate prediction at a historical cutoff",
                "survival concordance is apparent in-sample and is not a held-out estimate",
                "no intervention experiment supports causal savings or ROI claims",
            ],
        },
    }

    report_path = project_root / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(benchmark_report, f, indent=2)

    print(
        f"\n✅ All artifacts and benchmarks saved to:\n  - {models_dir}\n  - {report_path}"
    )
    print("==================================================")


if __name__ == "__main__":
    run_pipeline()
