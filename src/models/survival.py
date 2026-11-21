"""Exploratory survival analysis for retrospective seller time-to-breach associations.

Implements Kaplan-Meier survival curves and Cox Proportional Hazards regression
using `lifelines`. The current full-history aggregates do not support prospective
time-to-degradation evaluation; reported fit statistics are apparent/in-sample.
"""

import joblib
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index


class SellerSurvivalModel:
    """Models the duration until a seller experiences their first major SLA breach."""

    def __init__(self, penalizer: float = 0.1):
        self.penalizer = penalizer
        self.cph = CoxPHFitter(penalizer=penalizer)
        self.kmf = KaplanMeierFitter()
        self.feature_columns = []

    def fit_kaplan_meier(
        self, durations: pd.Series, events: pd.Series
    ) -> KaplanMeierFitter:
        """Fits non-parametric baseline survival curve."""
        self.kmf.fit(durations, event_observed=events, label="All Marketplace Sellers")
        return self.kmf

    def fit_cox_ph(
        self,
        X: pd.DataFrame,
        durations: pd.Series,
        events: pd.Series,
    ) -> dict[str, float]:
        """Fits Cox Proportional Hazards regression model.

        Returns:
            Dictionary with C-index, log-likelihood, and summary statistics.
        """
        # Assemble survival dataframe
        df = X.copy()
        df["duration"] = durations.values
        df["event"] = events.values

        # Select low-collinearity features for robust CPH estimation
        key_survival_cols = [
            col
            for col in [
                "avg_delivery_delay",
                "avg_carrier_handoff",
                "freight_ratio",
                "avg_item_price",
                "orders_per_month",
                "has_comment_rate",
                "duration",
                "event",
            ]
            if col in df.columns
        ]

        cph_data = df[key_survival_cols].dropna().copy()
        # Drop zero or negative durations
        cph_data = cph_data[cph_data["duration"] > 0]

        self.cph.fit(cph_data, duration_col="duration", event_col="event")
        self.feature_columns = [
            c for c in key_survival_cols if c not in ["duration", "event"]
        ]

        # Predict partial hazard & calculate C-index
        predicted_hazard = self.cph.predict_partial_hazard(cph_data)
        c_idx = concordance_index(
            cph_data["duration"], -predicted_hazard.values, cph_data["event"]
        )

        summary = {
            "concordance_index": float(c_idx),
            "log_likelihood": float(self.cph.log_likelihood_),
            "sample_size": len(cph_data),
            "events_observed": int(cph_data["event"].sum()),
        }

        return summary

    def get_hazard_ratios(self) -> pd.DataFrame:
        """Returns feature hazard ratios (exp(coef)) with 95% confidence intervals."""
        if not hasattr(self.cph, "hazard_ratios_"):
            raise ValueError("Model must be fitted before extracting hazard ratios.")

        summary_df = self.cph.summary[
            ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]
        ]
        summary_df.columns = ["hazard_ratio", "lower_95", "upper_95", "p_value"]
        return summary_df.sort_values("hazard_ratio", ascending=False)

    def predict_survival_probability(
        self,
        seller_features: pd.DataFrame,
        times: list[int] | None = None,
    ) -> pd.DataFrame:
        """Predicts probability of surviving without an SLA breach at specific day horizons."""
        if times is None:
            times = [30, 60, 90, 180]
        surv_curves = self.cph.predict_survival_function(seller_features)
        predictions = {}
        for t in times:
            # Interpolate or find nearest time point
            idx = surv_curves.index.get_indexer([t], method="nearest")[0]
            predictions[f"surv_prob_{t}d"] = surv_curves.iloc[idx].values

        return pd.DataFrame(predictions, index=seller_features.index)

    def save(self, filepath: str):
        """Persists trained survival model."""
        joblib.dump(
            {"cph": self.cph, "kmf": self.kmf, "features": self.feature_columns},
            filepath,
        )

    @classmethod
    def load(cls, filepath: str) -> "SellerSurvivalModel":
        """Loads persisted survival model."""
        data = joblib.load(filepath)
        instance = cls()
        instance.cph = data["cph"]
        instance.kmf = data["kmf"]
        instance.feature_columns = data["features"]
        return instance
