from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from rv_reporter.report_types.plugins import ReportPluginManager, compute_report_metrics


def _write_manifest_schema(root: Path) -> None:
    schema_src = Path("report_type_plugins/manifest.schema.yaml")
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.schema.yaml").write_text(schema_src.read_text(encoding="utf-8"), encoding="utf-8")


def test_plugin_manager_loads_filesystem_plugin(tmp_path: Path) -> None:
    root = tmp_path / "plugins"
    _write_manifest_schema(root)
    plugin_dir = root / "demo_profile"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "def get_spec():",
                "    return {'metrics_profile': 'demo_profile', 'api_version': 1, 'title': 'Demo'}",
                "def build(df, prefs, ctx):",
                "    return {'summary': {'report_type_id': ctx.report_type_id, 'rows': len(df)}, 'alerts': []}",
            ]
        ),
        encoding="utf-8",
    )
    (plugin_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "plugin_id": "demo_profile",
                "metrics_profile": "demo_profile",
                "api_version": 1,
                "version": "1.0.0",
                "title": "Demo",
                "description": "Demo plugin",
                "family": "tabular_statistical",
                "domain": "operations",
                "mode": "health_score",
                "entrypoint": "plugin.py",
                "status": "active",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    manager = ReportPluginManager(plugin_dir=root)
    assert "demo_profile" in manager.list_supported_profiles()

    result = manager.compute_metrics(
        metrics_profile="demo_profile",
        df=pd.DataFrame({"x": [1, 2]}),
        prefs={},
        report_type_id="demo_type",
    )
    assert result["summary"]["report_type_id"] == "demo_type"
    assert result["summary"]["rows"] == 2


def test_plugin_manager_falls_back_to_legacy_metrics() -> None:
    result = compute_report_metrics(
        metrics_profile="ops_kpi",
        df=pd.DataFrame(
            {
                "timestamp": ["2025-01-01T00:00:00Z"],
                "service": ["svc-a"],
                "requests": [100],
                "errors": [0],
                "latency_ms": [25.0],
            }
        ),
        prefs={},
        report_type_id="ops_report",
    )
    assert result["totals"]["requests"] == 100


def test_plugin_manager_rejects_invalid_api_version(tmp_path: Path) -> None:
    root = tmp_path / "plugins"
    _write_manifest_schema(root)
    plugin_dir = root / "bad_profile"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.py").write_text(
        "\n".join(
            [
                "def get_spec():",
                "    return {'metrics_profile': 'bad_profile', 'api_version': 2, 'title': 'Bad'}",
                "def build(df, prefs, ctx):",
                "    return {}",
            ]
        ),
        encoding="utf-8",
    )
    (plugin_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "plugin_id": "bad_profile",
                "metrics_profile": "bad_profile",
                "api_version": 1,
                "version": "1.0.0",
                "title": "Bad",
                "description": "Invalid by get_spec version",
                "family": "tabular_statistical",
                "domain": "operations",
                "mode": "health_score",
                "entrypoint": "plugin.py",
                "status": "active",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    manager = ReportPluginManager(plugin_dir=root)
    with pytest.raises(ValueError, match="api_version"):
        manager.list_supported_profiles()


def test_plugin_manager_normalizes_ai_style_get_spec_from_manifest(tmp_path: Path) -> None:
    root = tmp_path / "plugins"
    _write_manifest_schema(root)
    plugin_dir = root / "snmp_performance_trend"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.py").write_text(
        "\n".join(
            [
                "def get_spec():",
                "    return {'id': 'snmp_performance_trend', 'title': 'SNMP performance trend'}",
                "def build(df, prefs, ctx):",
                "    return {'summary': {'rows': len(df), 'report_type_id': ctx.report_type_id}}",
            ]
        ),
        encoding="utf-8",
    )
    (plugin_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "plugin_id": "snmp_performance_trend",
                "metrics_profile": "snmp_performance_trend",
                "api_version": 1,
                "version": "1.0.0",
                "title": "SNMP performance trend",
                "description": "Trend plugin",
                "family": "time_series",
                "domain": "networking",
                "mode": "trend_analysis",
                "entrypoint": "plugin.py",
                "status": "draft",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    manager = ReportPluginManager(plugin_dir=root)
    result = manager.compute_metrics(
        metrics_profile="snmp_performance_trend",
        df=pd.DataFrame({"x": [1, 2]}),
        prefs={},
        report_type_id="snmp_performance_trend",
    )
    assert result["summary"]["rows"] == 2


def test_builtin_wireshark_profile_is_available_via_plugin_folder() -> None:
    manager = ReportPluginManager(plugin_dir=Path("report_type_plugins"))
    assert "wireshark_capture_health" in manager.list_supported_profiles()


def test_plugin_manager_rejects_missing_manifest(tmp_path: Path) -> None:
    root = tmp_path / "plugins"
    _write_manifest_schema(root)
    plugin_dir = root / "missing_manifest"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.py").write_text(
        "\n".join(
            [
                "def get_spec():",
                "    return {'metrics_profile': 'missing_manifest', 'api_version': 1, 'title': 'x'}",
                "def build(df, prefs, ctx):",
                "    return {}",
            ]
        ),
        encoding="utf-8",
    )
    manager = ReportPluginManager(plugin_dir=root)
    with pytest.raises(ValueError, match="missing manifest.yaml"):
        manager.list_supported_profiles()


def test_builtin_telecom_threshold_sla_v2_plugin_loads_and_runs_with_pm_sample() -> None:
    manager = ReportPluginManager(plugin_dir=Path("report_type_plugins"))
    df = pd.read_csv("samples/pm-csv-es.csv")

    result = manager.compute_metrics(
        metrics_profile="telecom_session_health_threshold_sla_v2",
        df=df,
        prefs={
            "sla_thresholds": {
                "latency_good": 10,
                "latency_critical": 50,
                "jitter_good": 5,
                "jitter_critical": 20,
                "packet_loss_good": 0,
                "packet_loss_critical": 1,
            },
        },
        report_type_id="telecom_session_health_threshold_sla_v2",
    )

    assert "summary" in result
    assert "compliance" in result
    assert "per_interval" in result
    assert "charts" in result
    assert "alerts" in result
    assert result["summary"]["total_intervals"] > 0
    assert set(result["compliance"].keys()) == {"latency", "jitter", "packet_loss"}


def test_builtin_telecom_health_report_v3_plugin_runs_with_pm_sample() -> None:
    manager = ReportPluginManager(plugin_dir=Path("report_type_plugins"))
    df = pd.read_csv("samples/pm-csv-es.csv")

    result = manager.compute_metrics(
        metrics_profile="telecom_session_health_report_v3",
        df=df,
        prefs={
            "sla_thresholds": {
                "latency_good": 10,
                "latency_critical": 50,
                "jitter_good": 5,
                "jitter_critical": 20,
                "packet_loss_good": 0,
                "packet_loss_critical": 1,
            },
        },
        report_type_id="telecom_session_health_report_v3",
    )

    assert "summary" in result
    assert result["summary"]["total_intervals"] > 0
    assert "charts" in result
    assert "alerts" in result
