from __future__ import annotations

import pandas as pd

from rv_reporter.report_types.plugins import ReportPluginManager


def test_telecom_session_health_sla_compliance_v4_smoke_build() -> None:
    manager = ReportPluginManager(plugin_dir="report_type_plugins")
    result = manager.compute_metrics(
        metrics_profile="telecom_session_health_sla_compliance_v4",
        df=pd.DataFrame({"sample": [1, 2, 3]}),
        prefs={},
        report_type_id="telecom_session_health_sla_compliance_v4",
    )
    assert isinstance(result, dict)
