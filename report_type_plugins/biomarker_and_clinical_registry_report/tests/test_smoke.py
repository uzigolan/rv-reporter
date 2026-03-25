from __future__ import annotations

import pandas as pd

from rv_reporter.report_types.plugins import ReportPluginManager


def test_biomarker_and_clinical_registry_report_smoke_build() -> None:
    manager = ReportPluginManager(plugin_dir="report_type_plugins")
    result = manager.compute_metrics(
        metrics_profile="biomarker_and_clinical_registry_report",
        df=pd.DataFrame({"sample": [1, 2, 3]}),
        prefs={},
        report_type_id="biomarker_and_clinical_registry_report",
    )
    assert isinstance(result, dict)
