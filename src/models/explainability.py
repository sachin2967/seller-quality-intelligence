"""Explainability and actionable business intervention module.

Generates human-interpretable risk driver attributions and prescribes
concrete merchant interventions for Category Managers and Operations teams.
"""

from typing import Any, Dict, List
import pandas as pd


class SellerExplainer:
    """Interprets model predictions and prescribes actionable business interventions."""

    def __init__(self, feature_means: Dict[str, float] = None):
        self.feature_means = feature_means or {}

    def fit_reference(self, X: pd.DataFrame):
        """Records marketplace baseline averages for relative comparisons."""
        self.feature_means = X.mean().to_dict()

    def explain_seller(self, seller_row: pd.Series) -> Dict[str, Any]:
        """Provides a structured risk diagnosis and recommended operational action.

        Args:
            seller_row: Series containing a single seller's feature values.

        Returns:
            Dictionary with risk level, top risk drivers, and recommended intervention.
        """
        drivers = []

        # Driver 1: Delivery Delay
        avg_delay = seller_row.get("avg_delivery_delay", 0.0)
        baseline_delay = self.feature_means.get("avg_delivery_delay", 0.0)
        if avg_delay > baseline_delay + 2.0:
            drivers.append({
                "factor": "High Delivery Delay",
                "value": f"{avg_delay:.1f} days",
                "benchmark": f"{baseline_delay:.1f} days avg",
                "severity": "HIGH" if avg_delay > baseline_delay + 5.0 else "MEDIUM",
            })

        # Driver 2: Carrier Handoff Latency
        handoff = seller_row.get("avg_carrier_handoff", 0.0)
        baseline_handoff = self.feature_means.get("avg_carrier_handoff", 2.0)
        if handoff > baseline_handoff + 1.5:
            drivers.append({
                "factor": "Slow Carrier Handoff",
                "value": f"{handoff:.1f} days to dispatch",
                "benchmark": f"{baseline_handoff:.1f} days avg",
                "severity": "HIGH",
            })

        # Driver 3: High Freight Ratio
        freight_ratio = seller_row.get("freight_ratio", 0.0)
        if freight_ratio > 0.35:
            drivers.append({
                "factor": "Excessive Shipping Cost Ratio",
                "value": f"{freight_ratio * 100:.1f}% of GMV",
                "benchmark": "Under 20% typical",
                "severity": "MEDIUM",
            })

        # Driver 4: Low Volume / Fragile Velocity
        orders_month = seller_row.get("orders_per_month", 0.0)
        if orders_month < 2.0:
            drivers.append({
                "factor": "Low Fulfillment Velocity",
                "value": f"{orders_month:.1f} orders/month",
                "benchmark": "> 5 orders/mo healthy",
                "severity": "LOW",
            })

        # Prescribe concrete business action
        intervention = self._prescribe_intervention(drivers)

        return {
            "seller_id": seller_row.get("seller_id", "Unknown"),
            "drivers": drivers[:3],  # Top 3 drivers
            "recommended_action": intervention,
        }

    def _prescribe_intervention(self, drivers: List[Dict[str, Any]]) -> Dict[str, str]:
        """Translates technical drivers into actionable ops playbook steps."""
        if not drivers:
            return {
                "action": "Maintain Elite Status",
                "playbook": "Eligible for Marketplace Preferred Seller badge and priority search ranking.",
                "urgency": "LOW",
            }

        high_factors = [d["factor"] for d in drivers if d.get("severity") == "HIGH"]

        if "Slow Carrier Handoff" in high_factors:
            return {
                "action": "Logistics Dispatch Audit",
                "playbook": "Seller takes >3 days to hand off parcels. Require automated pick-up scheduling or enforce 48-hour fulfillment SLA limit.",
                "urgency": "CRITICAL",
            }

        if "High Delivery Delay" in high_factors:
            return {
                "action": "Transit Corridor Re-routing",
                "playbook": "Carrier SLA is failing on customer delivery. Reassign default courier partner for seller's zip code corridor.",
                "urgency": "HIGH",
            }

        return {
            "action": "Commercial Health Check",
            "playbook": "High freight-to-price ratio detected. Encourage product bundling or regional warehouse inventory placement.",
            "urgency": "MEDIUM",
        }
