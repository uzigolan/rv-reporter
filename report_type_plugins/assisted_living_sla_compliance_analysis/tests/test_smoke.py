from __future__ import annotations

import pandas as pd

from rv_reporter.report_types.plugins import ReportPluginManager


def test_assisted_living_sla_compliance_analysis_smoke_build() -> None:
    manager = ReportPluginManager(plugin_dir="report_type_plugins")
    df = pd.DataFrame({
        "timestamp": [
            "2026-03-01T06:51:00Z", "2026-03-01T06:57:00Z",
            "2026-03-01T07:04:00Z", "2026-03-01T07:05:00Z",
        ],
        "device_name": [
            "Rm 101 Alice Brown", "Caregiver Maria Lopez",
            "Rm 101 Alice Brown", "Caregiver Maria Lopez",
        ],
        "property": [
            "Call Button Pressed", "Call Accepted",
            "Resolution Confirmed", "Resolution Confirmed",
        ],
    })
    result = manager.compute_metrics(
        metrics_profile="assisted_living_sla_compliance_analysis",
        df=df,
        prefs={},
        report_type_id="assisted_living_sla_compliance_analysis",
    )
    assert isinstance(result, dict)
    assert "summary" in result
    assert "alerts" in result
