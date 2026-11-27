"""Interactive Streamlit dashboard for Category Managers and Merchant Operations teams."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Seller Quality Intelligence Platform",
    page_icon="🛒",
    layout="wide",
)

# Root directory & paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "seller_features.csv"
MODELS_DIR = PROJECT_ROOT / "models"


@st.cache_data
def load_seller_data():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_resource
def load_models():
    clf_artifact = joblib.load(MODELS_DIR / "lgbm_seller_risk.joblib")
    explainer = joblib.load(MODELS_DIR / "seller_explainer.joblib")
    return clf_artifact["model"], clf_artifact["feature_names"], explainer


try:
    df = load_seller_data()
    model, feature_names, explainer = load_models()
except (FileNotFoundError, OSError, EOFError, KeyError, pd.errors.EmptyDataError) as e:
    st.error(
        f"Error loading artifacts: {e}. Please ensure `scripts/train_and_evaluate.py` has run."
    )
    st.stop()

# Compute live risk probabilities
X = df[feature_names].values
probs = model.predict_proba(X)[:, 1]
df["risk_prob"] = probs


def assign_segment(p):
    if p < 0.25:
        return "Elite / Low Risk"
    elif p < 0.50:
        return "Good / Stable"
    elif p < 0.75:
        return "At-Risk / Watchlist"
    else:
        return "Critical / Severe Risk"


df["segment"] = [assign_segment(p) for p in probs]

# --- UI HEADER ---
st.title("🛒 E-Commerce Seller Quality & Performance Intelligence")
st.markdown(
    "**Offline prototype:** Explore retrospective seller-condition scores, logistics diagnostics, "
    "and heuristic operations playbooks. Scores are not validated for future-risk decisions."
)
st.warning(
    "Evaluation limitation: seller features and rule-derived labels use overlapping full histories. "
    "Cross-validation measures retrospective label reconstruction, not early-warning performance. "
    "No causal or financial impact has been established."
)

st.divider()

# --- TOP METRIC CARDS ---
col1, col2, col3, col4, col5 = st.columns(5)
total_sellers = len(df)
at_risk_count = (df["risk_prob"] >= 0.5).sum()
at_risk_pct = (at_risk_count / total_sellers) * 100
avg_gmv = df["total_gmv"].mean()
avg_review = df["mean_review_score"].mean()
avg_late = df["late_delivery_rate"].mean() * 100

col1.metric("Active Sellers", f"{total_sellers:,}")
col2.metric(
    "At-Risk Sellers",
    f"{at_risk_count:,}",
    f"{at_risk_pct:.1f}% of total",
    delta_color="inverse",
)
col3.metric("Avg Seller GMV", f"R$ {avg_gmv:,.0f}")
col4.metric("Avg Review Score", f"{avg_review:.2f} / 5.0")
col5.metric("Avg Late Delivery Rate", f"{avg_late:.1f}%")

st.divider()

# --- MAIN DASHBOARD LAYOUT ---
tab1, tab2, tab3 = st.tabs(
    [
        "📊 Marketplace Portfolio Overview",
        "🔍 Seller Deep Dive & Interventions",
        "📈 Model Benchmarks",
    ]
)

with tab1:
    st.subheader("Marketplace Risk & Commercial Value Matrix")

    # Segment filter
    selected_segments = st.multiselect(
        "Filter by Risk Segment:",
        options=[
            "Elite / Low Risk",
            "Good / Stable",
            "At-Risk / Watchlist",
            "Critical / Severe Risk",
        ],
        default=["Critical / Severe Risk", "At-Risk / Watchlist"],
    )

    filtered_df = df[df["segment"].isin(selected_segments)] if selected_segments else df

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = {
        "Elite / Low Risk": "#2ecc71",
        "Good / Stable": "#3498db",
        "At-Risk / Watchlist": "#f39c12",
        "Critical / Severe Risk": "#e74c3c",
    }

    for seg in filtered_df["segment"].unique():
        sub = filtered_df[filtered_df["segment"] == seg]
        ax.scatter(
            sub["total_gmv"],
            sub["quality_score"],
            label=seg,
            color=colors.get(seg, "gray"),
            alpha=0.7,
            edgecolors="none",
            s=40,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Total GMV (BRL, log scale)")
    ax.set_ylabel("Quality Score (0 - 100)")
    ax.set_title("Seller Quality Score vs GMV by Risk Segment")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

    st.dataframe(
        filtered_df[
            [
                "seller_id",
                "segment",
                "risk_prob",
                "quality_score",
                "total_gmv",
                "late_delivery_rate",
                "mean_review_score",
            ]
        ]
        .rename(
            columns={
                "seller_id": "Seller ID",
                "segment": "Risk Segment",
                "risk_prob": "Risk Prob",
                "quality_score": "Score",
                "total_gmv": "GMV (BRL)",
                "late_delivery_rate": "Late Rate",
                "mean_review_score": "Review",
            }
        )
        .sort_values("Risk Prob", ascending=False),
        use_container_width=True,
    )

with tab2:
    st.subheader("Individual Merchant Diagnosis & Prescribed Playbook")

    seller_ids = df["seller_id"].tolist()
    selected_seller = st.selectbox(
        "Select Seller ID to Inspect:", options=seller_ids, index=0
    )

    s_row = df[df["seller_id"] == selected_seller].iloc[0]
    explanation = explainer.explain_seller(s_row)

    d_col1, d_col2 = st.columns([1, 1.2])

    with d_col1:
        st.markdown(f"### Seller Profile: `{selected_seller[:12]}...`")
        st.write(f"**Segment:** {s_row['segment']}")
        st.progress(
            float(s_row["risk_prob"]),
            text=f"Risk Probability: {s_row['risk_prob'] * 100:.1f}%",
        )

        m1, m2 = st.columns(2)
        m1.metric("Historical Review", f"{s_row['mean_review_score']:.2f} / 5.0")
        m2.metric("Late Delivery %", f"{s_row['late_delivery_rate'] * 100:.1f}%")

        m3, m4 = st.columns(2)
        m3.metric("Carrier Handoff Delay", f"{s_row['avg_carrier_handoff']:.1f} days")
        m4.metric("Avg Delivery Delay", f"{s_row['avg_delivery_delay']:.1f} days")

    with d_col2:
        st.markdown("### 🛠️ Prescribed Category Manager Action")
        action_info = explanation["recommended_action"]
        urgency = action_info.get("urgency", "LOW")

        if urgency == "CRITICAL":
            st.error(f"🚨 **Action: {action_info['action']}** (Urgency: {urgency})")
        elif urgency == "HIGH":
            st.warning(f"⚠️ **Action: {action_info['action']}** (Urgency: {urgency})")
        else:
            st.info(f"ℹ️ **Action: {action_info['action']}** (Urgency: {urgency})")

        st.write(f"**Operations Playbook:** {action_info['playbook']}")

        st.markdown("#### Top Root-Cause Risk Drivers:")
        if explanation["drivers"]:
            for d in explanation["drivers"]:
                st.markdown(
                    f"- **{d['factor']}**: `{d['value']}` (Benchmark: *{d['benchmark']}*) [Severity: **{d['severity']}**]"
                )
        else:
            st.write(
                "No severe negative drivers detected. Healthy merchant operations."
            )

with tab3:
    st.subheader("Retrospective Offline Evaluation")

    benchmark_json = PROJECT_ROOT / "benchmark_report.json"
    if benchmark_json.exists():
        import json

        with open(benchmark_json) as f:
            b_data = json.load(f)

        m_data = b_data.get("models", {})
        lr = m_data.get("baseline_logistic_regression", {})
        lgb = m_data.get("champion_lightgbm", {})

        scope = b_data.get("evaluation_scope", {})
        st.info(f"Classification scope: {scope.get('classification', 'not specified')}")
        st.markdown("### 5-Fold Stratified Seller Cross-Validation")
        comp_df = pd.DataFrame(
            {
                "Metric": [
                    "ROC-AUC",
                    "PR-AUC",
                    "F1-Score",
                    "Brier Score (Calibration)",
                ],
                "Baseline (Logistic Regression)": [
                    f"{lr.get('roc_auc', 0):.4f}",
                    f"{lr.get('pr_auc', 0):.4f}",
                    f"{lr.get('f1_score', 0):.4f}",
                    f"{lr.get('brier_score', 0):.4f}",
                ],
                "Champion (LightGBM)": [
                    f"{lgb.get('roc_auc', 0):.4f}",
                    f"{lgb.get('pr_auc', 0):.4f}",
                    f"{lgb.get('f1_score', 0):.4f}",
                    f"{lgb.get('brier_score', 0):.4f}",
                ],
                "Improvement": [
                    f"+{(lgb.get('roc_auc', 0) - lr.get('roc_auc', 0)) * 100:.2f}%",
                    f"+{(lgb.get('pr_auc', 0) - lr.get('pr_auc', 0)) * 100:.2f}%",
                    f"+{(lgb.get('f1_score', 0) - lr.get('f1_score', 0)) * 100:.2f}%",
                    f"{(lgb.get('brier_score', 0) - lr.get('brier_score', 0)) * 100:.2f}% (better)",
                ],
            }
        )
        st.table(comp_df)

        st.caption(
            "These metrics describe retrospective reconstruction of a rule-derived label. "
            "They do not estimate future prediction quality, intervention benefit, or ROI."
        )
