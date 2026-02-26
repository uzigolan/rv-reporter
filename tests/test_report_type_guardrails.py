from __future__ import annotations

from pathlib import Path

import yaml

from rv_reporter.report_types.plugins import ReportPluginManager


def test_all_plugin_manifests_load_with_schema() -> None:
    manager = ReportPluginManager(plugin_dir=Path("report_type_plugins"))
    profiles = manager.list_supported_profiles()
    assert "wireshark_capture_health" in profiles


def test_all_report_type_configs_reference_supported_metrics_profiles() -> None:
    manager = ReportPluginManager(plugin_dir=Path("report_type_plugins"))
    supported = manager.list_supported_profiles()

    config_dir = Path("configs/report_types")
    for yaml_path in sorted(config_dir.glob("*.yaml")):
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert isinstance(raw, dict)
        profile = str(raw.get("metrics_profile", "")).strip()
        assert profile in supported, f"{yaml_path.name} uses unsupported metrics_profile '{profile}'"
