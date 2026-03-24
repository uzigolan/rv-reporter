from __future__ import annotations

import pandas as pd

from rv_reporter.report_types.plugins import ReportPluginManager


def test_telecom_session_sla_report_smoke_build() -> None:
    manager = ReportPluginManager(plugin_dir="report_type_plugins")
    result = manager.compute_metrics(
        metrics_profile="telecom_session_sla_report",
        df=pd.DataFrame({"sample": [1, 2, 3]}),
        prefs={},
        report_type_id="telecom_session_sla_report",
    )
    assert isinstance(result, dict)
