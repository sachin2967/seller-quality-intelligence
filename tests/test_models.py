"""Unit tests for model inference, survival prediction, and explainability."""

from pathlib import Path
import joblib
import pytest
from src.models.classifier import SellerRiskClassifier
from src.models.survival import SellerSurvivalModel
from src.models.explainability import SellerExplainer


@pytest.fixture
def models_dir():
    return Path(__file__).resolve().parent.parent / "models"


def test_classifier_artifact_loading(models_dir):
    clf_path = models_dir / "lgbm_seller_risk.joblib"
    assert clf_path.exists()

    clf = SellerRiskClassifier.load(clf_path)
    assert clf.champion_model is not None
    assert len(clf.feature_names) > 0


def test_survival_model_artifact_loading(models_dir):
    surv_path = models_dir / "seller_survival_cox.joblib"
    assert surv_path.exists()

    surv = SellerSurvivalModel.load(str(surv_path))
    assert surv.cph is not None
    hazard_df = surv.get_hazard_ratios()
    assert not hazard_df.empty
    assert "hazard_ratio" in hazard_df.columns


def test_explainer_prescriptions(models_dir):
    exp_path = models_dir / "seller_explainer.joblib"
    assert exp_path.exists()

    explainer = joblib.load(exp_path)
    sample_seller = {
        "seller_id": "test_seller_1",
        "avg_delivery_delay": 8.5,
        "avg_carrier_handoff": 4.2,
        "freight_ratio": 0.40,
        "orders_per_month": 1.0,
    }
    explanation = explainer.explain_seller(sample_seller)
    assert "drivers" in explanation
    assert "recommended_action" in explanation
    assert explanation["recommended_action"]["urgency"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
