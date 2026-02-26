from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rv_reporter.report_types.scaffold import scaffold_report_type


def test_scaffold_report_type_creates_expected_files(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs" / "report_types"
    plugin_root = tmp_path / "report_type_plugins"
    plugin_root.mkdir(parents=True, exist_ok=True)
    (plugin_root / "manifest.schema.yaml").write_text(
        Path("report_type_plugins/manifest.schema.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    result = scaffold_report_type(
        report_type_id="net_usage",
        title="Network Usage",
        family="time_series",
        domain="networking",
        mode="trend_analysis",
        required_columns=["timestamp", "interface", "bytes_in", "bytes_out"],
        generator="openai_sdk",
        create_report_type_yaml=True,
        config_dir=config_dir,
        plugin_root=plugin_root,
    )

    assert result.report_type_yaml is not None and result.report_type_yaml.exists()
    assert result.plugin_manifest.exists()
    assert result.plugin_code.exists()
    assert result.plugin_smoke_test.exists()

    manifest = yaml.safe_load(result.plugin_manifest.read_text(encoding="utf-8"))
    assert manifest["plugin_id"] == "net_usage"
    assert manifest["metrics_profile"] == "net_usage"
    assert manifest["family"] == "time_series"
    assert manifest["generator"] == "openai_sdk"


def test_scaffold_report_type_refuses_overwrite_without_force(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs" / "report_types"
    plugin_root = tmp_path / "report_type_plugins"
    plugin_root.mkdir(parents=True, exist_ok=True)
    (plugin_root / "manifest.schema.yaml").write_text(
        Path("report_type_plugins/manifest.schema.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    scaffold_report_type(
        report_type_id="net_usage",
        title="Network Usage",
        family="time_series",
        domain="networking",
        mode="trend_analysis",
        required_columns=["timestamp"],
        config_dir=config_dir,
        plugin_root=plugin_root,
    )

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        scaffold_report_type(
            report_type_id="net_usage",
            title="Network Usage",
            family="time_series",
            domain="networking",
            mode="trend_analysis",
            required_columns=["timestamp"],
            config_dir=config_dir,
            plugin_root=plugin_root,
        )
