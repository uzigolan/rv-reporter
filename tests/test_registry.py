from rv_reporter.report_types.registry import ReportTypeRegistry


def test_registry_lists_report_types() -> None:
    registry = ReportTypeRegistry(config_dir="configs/report_types")
    assert "network_queue_congestion" in registry.list_report_types()
    assert "twamp_session_health" in registry.list_report_types()
    assert "pm_export_health" in registry.list_report_types()


def test_registry_loads_definition() -> None:
    registry = ReportTypeRegistry(config_dir="configs/report_types")
    definition = registry.get("network_queue_congestion")
    assert definition.report_type_id == "network_queue_congestion"
    assert "Time" in definition.required_columns
