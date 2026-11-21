"""Guardrails against presenting retrospective metrics as prospective evidence."""

import json
from pathlib import Path


def test_benchmark_report_discloses_non_prospective_scope():
    report_path = Path(__file__).resolve().parent.parent / "benchmark_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    scope = report["evaluation_scope"]
    assert scope["prospective_prediction_validated"] is False
    assert scope["financial_impact_validated"] is False
    assert "retrospective_full_history" in scope["classification"]
    assert report["models"]["survival_cox_ph"]["evaluation_scope"].startswith(
        "apparent_in_sample"
    )
    assert "financial_impact" not in report
