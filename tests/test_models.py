"""Real serialization round trips on small authored synthetic training fixtures."""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.classifier import SellerRiskClassifier
from src.models.explainability import SellerExplainer
from src.models.survival import SellerSurvivalModel


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    directory = tmp_path_factory.mktemp("seller-model-contracts")
    rng = np.random.default_rng(42)
    features = pd.DataFrame(
        {
            "avg_delivery_delay": rng.normal(2, 1, 60),
            "freight_ratio": rng.uniform(0.1, 0.5, 60),
            "orders_per_month": rng.uniform(1, 10, 60),
        }
    )
    labels = (
        features["avg_delivery_delay"] > features["avg_delivery_delay"].median()
    ).astype(int)
    classifier = SellerRiskClassifier()
    classifier.train_champion_lgbm(
        features,
        labels,
        n_splits=2,
        params={
            "n_estimators": 8,
            "num_leaves": 4,
            "min_child_samples": 2,
            "random_state": 42,
            "n_jobs": 1,
            "verbosity": -1,
        },
    )
    classifier.save(directory / "classifier.joblib")
    survival = SellerSurvivalModel(penalizer=0.2)
    durations = pd.Series(rng.exponential(20, 60) + 1)
    events = pd.Series(rng.binomial(1, 0.7, 60))
    survival.fit_cox_ph(features, durations, events)
    survival.save(str(directory / "survival.joblib"))
    explainer = SellerExplainer()
    explainer.fit_reference(features)
    joblib.dump(explainer, directory / "explainer.joblib")
    return {
        "directory": directory,
        "features": features,
        "scores": classifier.champion_model.predict_proba(features),
        "survival": survival.predict_survival_probability(features, times=[5, 10]),
    }


def test_classifier_artifact_loading(artifacts):
    classifier = SellerRiskClassifier.load(artifacts["directory"] / "classifier.joblib")
    assert classifier.feature_names == list(artifacts["features"].columns)
    np.testing.assert_allclose(
        classifier.champion_model.predict_proba(artifacts["features"]),
        artifacts["scores"],
    )


def test_survival_model_artifact_loading(artifacts):
    survival = SellerSurvivalModel.load(str(artifacts["directory"] / "survival.joblib"))
    hazard_df = survival.get_hazard_ratios()
    assert set(hazard_df.index) == set(artifacts["features"].columns)
    assert "hazard_ratio" in hazard_df.columns
    pd.testing.assert_frame_equal(
        survival.predict_survival_probability(artifacts["features"], times=[5, 10]),
        artifacts["survival"],
    )


def test_explainer_prescriptions(artifacts):
    explainer = joblib.load(artifacts["directory"] / "explainer.joblib")
    explanation = explainer.explain_seller(
        {
            "seller_id": "test_seller_1",
            "avg_delivery_delay": 8.5,
            "avg_carrier_handoff": 4.2,
            "freight_ratio": 0.40,
            "orders_per_month": 1.0,
        }
    )
    assert [driver["factor"] for driver in explanation["drivers"]] == [
        "High Delivery Delay",
        "Slow Carrier Handoff",
        "Excessive Shipping Cost Ratio",
    ]
    assert explanation["recommended_action"]["urgency"] == "CRITICAL"
