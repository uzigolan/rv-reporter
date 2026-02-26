from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

FAMILIES = {"time_series", "tabular_statistical", "event", "log_text", "hybrid"}
DOMAINS = {"networking", "project_management", "healthcare", "operations", "security", "finance"}
MODES = {
    "health_score",
    "anomaly_detection",
    "trend_analysis",
    "threshold_sla",
    "burst_detection",
    "correlation_analysis",
    "distribution_analysis",
    "top_n_hotspots",
    "flow_bottleneck",
}


@dataclass(frozen=True, slots=True)
class ScaffoldResult:
    report_type_yaml: Path | None
    plugin_manifest: Path
    plugin_code: Path
    plugin_smoke_test: Path


def scaffold_report_type(
    *,
    report_type_id: str,
    title: str,
    family: str,
    domain: str,
    mode: str,
    required_columns: list[str],
    version: str = "1.0.0",
    description: str = "",
    owner: str = "platform",
    generator: str = "manual",
    inherits_from: str = "",
    status: str = "draft",
    create_report_type_yaml: bool = True,
    config_dir: str | Path = "configs/report_types",
    plugin_root: str | Path = "report_type_plugins",
    force: bool = False,
) -> ScaffoldResult:
    _validate_scaffold_inputs(
        report_type_id=report_type_id,
        family=family,
        domain=domain,
        mode=mode,
        status=status,
        required_columns=required_columns,
    )

    plugin_root_path = Path(plugin_root)
    plugin_dir = plugin_root_path / report_type_id
    manifest_path = plugin_dir / "manifest.yaml"
    plugin_path = plugin_dir / "plugin.py"
    tests_dir = plugin_dir / "tests"
    smoke_test_path = tests_dir / "test_smoke.py"
    report_type_yaml_path = Path(config_dir) / f"{report_type_id}.yaml"

    for path in (manifest_path, plugin_path, smoke_test_path):
        if path.exists() and not force:
            raise ValueError(f"Refusing to overwrite existing file: '{path}'. Use force=True to overwrite.")
    if create_report_type_yaml and report_type_yaml_path.exists() and not force:
        raise ValueError(f"Refusing to overwrite existing file: '{report_type_yaml_path}'. Use force=True to overwrite.")

    plugin_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "plugin_id": report_type_id,
        "metrics_profile": report_type_id,
        "api_version": 1,
        "version": version,
        "title": title,
        "description": description or f"{title} plugin.",
        "family": family,
        "domain": domain,
        "mode": mode,
        "entrypoint": "plugin.py",
        "owner": owner,
        "generator": generator,
        "status": status,
    }
    if inherits_from.strip():
        manifest["inherits_from"] = inherits_from.strip()

    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    plugin_path.write_text(_plugin_template(report_type_id=report_type_id, title=title), encoding="utf-8")
    smoke_test_path.write_text(_smoke_test_template(report_type_id=report_type_id), encoding="utf-8")

    created_report_yaml: Path | None = None
    if create_report_type_yaml:
        report_type_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_payload = {
            "report_type_id": report_type_id,
            "version": version,
            "title": title,
            "required_columns": required_columns,
            "metrics_profile": report_type_id,
            "default_prefs": {},
            "output_schema": {"type": "object"},
            "prompt_instructions": "Summarize key findings and recommended actions.",
        }
        report_type_yaml_path.write_text(yaml.safe_dump(yaml_payload, sort_keys=False), encoding="utf-8")
        created_report_yaml = report_type_yaml_path

    return ScaffoldResult(
        report_type_yaml=created_report_yaml,
        plugin_manifest=manifest_path,
        plugin_code=plugin_path,
        plugin_smoke_test=smoke_test_path,
    )


def _validate_scaffold_inputs(
    *,
    report_type_id: str,
    family: str,
    domain: str,
    mode: str,
    status: str,
    required_columns: list[str],
) -> None:
    if not report_type_id or any(c for c in report_type_id if c not in "abcdefghijklmnopqrstuvwxyz0123456789_"):
        raise ValueError("report_type_id must match [a-z0-9_]+.")
    if family not in FAMILIES:
        raise ValueError(f"Unsupported family '{family}'. Expected one of: {sorted(FAMILIES)}")
    if domain not in DOMAINS:
        raise ValueError(f"Unsupported domain '{domain}'. Expected one of: {sorted(DOMAINS)}")
    if mode not in MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of: {sorted(MODES)}")
    if status not in {"draft", "active", "deprecated"}:
        raise ValueError("status must be one of: draft, active, deprecated.")
    if not required_columns:
        raise ValueError("At least one required column is required.")


def _plugin_template(*, report_type_id: str, title: str) -> str:
    return f"""from __future__ import annotations

from typing import Any

import pandas as pd


def get_spec() -> dict[str, Any]:
    return {{
        "metrics_profile": "{report_type_id}",
        "api_version": 1,
        "title": "{title}",
        "description": "{title} plugin.",
    }}


def build(df: pd.DataFrame, prefs: dict[str, Any], ctx: Any) -> dict[str, Any]:
    # TODO: implement report-specific computation logic.
    return {{
        "summary": {{
            "rows": int(len(df)),
            "report_type_id": getattr(ctx, "report_type_id", "{report_type_id}"),
        }},
        "rows": [],
        "alerts": [],
    }}
"""


def _smoke_test_template(*, report_type_id: str) -> str:
    return f"""from __future__ import annotations

import pandas as pd

from rv_reporter.report_types.plugins import ReportPluginManager


def test_{report_type_id}_smoke_build() -> None:
    manager = ReportPluginManager(plugin_dir="report_type_plugins")
    result = manager.compute_metrics(
        metrics_profile="{report_type_id}",
        df=pd.DataFrame({{"sample": [1, 2, 3]}}),
        prefs={{}},
        report_type_id="{report_type_id}",
    )
    assert isinstance(result, dict)
"""
