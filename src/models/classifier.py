"""Model training and evaluation module for Seller Quality Risk Classification.

Implements both a Logistic Regression baseline and a tuned LightGBM classifier
with Stratified 5-Fold Cross-Validation, complete metric reporting (ROC-AUC, PR-AUC, Brier score),
and model persistence.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


@dataclass
class CVResult:
    """Holds cross-validation results across folds."""

    mean_roc_auc: float
    std_roc_auc: float
    mean_pr_auc: float
    std_pr_auc: float
    mean_f1: float
    mean_brier: float
    fold_results: list[dict[str, float]]
    feature_importances: pd.DataFrame


class SellerRiskClassifier:
    """Trains and cross-validates models against the retrospective seller-state label."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.baseline_model = None
        self.champion_model = None
        self.scaler = StandardScaler()
        self.feature_names = []

    def evaluate_baseline_lr(
        self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
    ) -> CVResult:
        """Evaluates Logistic Regression baseline using Stratified K-Fold."""
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        roc_aucs, pr_aucs, f1s, briers = [], [], [], []
        fold_details = []

        X_mat = X.values
        y_mat = y.values
        self.feature_names = list(X.columns)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_mat, y_mat)):
            X_tr, y_tr = X_mat[train_idx], y_mat[train_idx]
            X_va, y_val = X_mat[val_idx], y_mat[val_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)

            lr = LogisticRegression(
                class_weight="balanced", random_state=self.random_state, max_iter=1000
            )
            lr.fit(X_tr_s, y_tr)

            probs = lr.predict_proba(X_va_s)[:, 1]
            preds = (probs >= 0.5).astype(int)

            roc = roc_auc_score(y_val, probs)
            pr = average_precision_score(y_val, probs)
            f1 = f1_score(y_val, preds, zero_division=0)
            brier = brier_score_loss(y_val, probs)

            roc_aucs.append(roc)
            pr_aucs.append(pr)
            f1s.append(f1)
            briers.append(brier)
            fold_details.append(
                {
                    "fold": fold + 1,
                    "roc_auc": roc,
                    "pr_auc": pr,
                    "f1": f1,
                    "brier": brier,
                }
            )

        coef_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": np.abs(lr.coef_[0]),
            }
        ).sort_values("importance", ascending=False)

        return CVResult(
            mean_roc_auc=float(np.mean(roc_aucs)),
            std_roc_auc=float(np.std(roc_aucs)),
            mean_pr_auc=float(np.mean(pr_aucs)),
            std_pr_auc=float(np.std(pr_aucs)),
            mean_f1=float(np.mean(f1s)),
            mean_brier=float(np.mean(briers)),
            fold_results=fold_details,
            feature_importances=coef_df,
        )

    def train_champion_lgbm(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        params: dict[str, Any] | None = None,
    ) -> tuple[CVResult, lgb.LGBMClassifier]:
        """Trains and evaluates Champion LightGBM with Stratified K-Fold."""
        if params is None:
            params = {
                "n_estimators": 150,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": 5,
                "min_child_samples": 15,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "class_weight": "balanced",
                "random_state": self.random_state,
                "verbose": -1,
            }

        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        roc_aucs, pr_aucs, f1s, briers = [], [], [], []
        fold_details = []
        importances = np.zeros(X.shape[1])

        self.feature_names = list(X.columns)
        X_mat = X.values
        y_mat = y.values

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_mat, y_mat)):
            X_tr, y_tr = X_mat[train_idx], y_mat[train_idx]
            X_va, y_val = X_mat[val_idx], y_mat[val_idx]

            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_tr, y_tr)

            probs = clf.predict_proba(X_va)[:, 1]
            preds = (probs >= 0.5).astype(int)

            roc = roc_auc_score(y_val, probs)
            pr = average_precision_score(y_val, probs)
            f1 = f1_score(y_val, preds, zero_division=0)
            brier = brier_score_loss(y_val, probs)

            roc_aucs.append(roc)
            pr_aucs.append(pr)
            f1s.append(f1)
            briers.append(brier)
            importances += clf.feature_importances_ / n_splits
            fold_details.append(
                {
                    "fold": fold + 1,
                    "roc_auc": roc,
                    "pr_auc": pr,
                    "f1": f1,
                    "brier": brier,
                }
            )

        # Fit final model on full dataset
        final_clf = lgb.LGBMClassifier(**params)
        final_clf.fit(X_mat, y_mat)
        self.champion_model = final_clf

        imp_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)

        cv_res = CVResult(
            mean_roc_auc=float(np.mean(roc_aucs)),
            std_roc_auc=float(np.std(roc_aucs)),
            mean_pr_auc=float(np.mean(pr_aucs)),
            std_pr_auc=float(np.std(pr_aucs)),
            mean_f1=float(np.mean(f1s)),
            mean_brier=float(np.mean(briers)),
            fold_results=fold_details,
            feature_importances=imp_df,
        )

        return cv_res, final_clf

    def save(self, filepath: str | Path):
        """Persists trained model and metadata."""
        artifact = {
            "model": self.champion_model,
            "feature_names": self.feature_names,
            "random_state": self.random_state,
        }
        joblib.dump(artifact, filepath)

    @classmethod
    def load(cls, filepath: str | Path) -> "SellerRiskClassifier":
        """Loads persisted classifier."""
        artifact = joblib.load(filepath)
        instance = cls(random_state=artifact["random_state"])
        instance.champion_model = artifact["model"]
        instance.feature_names = artifact["feature_names"]
        return instance
