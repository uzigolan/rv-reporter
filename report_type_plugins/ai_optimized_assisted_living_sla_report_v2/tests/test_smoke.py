from __future__ import annotations

import pandas as pd

from rv_reporter.report_types.plugins import ReportPluginManager


def test_ai_optimized_assisted_living_sla_report_v2_smoke_build() -> None:
    manager = ReportPluginManager(plugin_dir="report_type_plugins")
    result = manager.compute_metrics(
        metrics_profile="ai_optimized_assisted_living_sla_report_v2",
        df=pd.DataFrame({"sample": [1, 2, 3]}),
        prefs={},
        report_type_id="ai_optimized_assisted_living_sla_report_v2",
    )
    assert isinstance(result, dict)
