from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import pandas as pd
import yaml
from jsonschema import ValidationError, validate

from rv_reporter.services.metrics import compute_legacy_metrics

PLUGIN_API_VERSION = 1

_LEGACY_METRICS_PROFILES = {
    "ops_kpi",
    "finance_variance",
    "network_queue_congestion",
    "twamp_session_health",
    "pm_export_health",
    "jira_issue_portfolio",
    "ms_biomarker_registry_health",
    "wireshark_capture_health",
}


@dataclass(frozen=True, slots=True)
class ReportPluginSpec:
    metrics_profile: str
    api_version: int
    title: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class ReportPluginContext:
    report_type_id: str


@dataclass(frozen=True, slots=True)
class _LoadedPlugin:
    spec: ReportPluginSpec
    manifest: dict[str, Any]
    build_fn: Callable[[pd.DataFrame, dict[str, Any], ReportPluginContext], dict[str, Any]]


class ReportPluginManager:
    def __init__(self, plugin_dir: str | Path | None = None) -> None:
        env_dir = os.getenv("RV_REPORT_PLUGINS_DIR", "").strip()
        configured_dir = plugin_dir or env_dir or "report_type_plugins"
        self._plugin_dir = Path(configured_dir)
        self._manifest_schema_path = self._plugin_dir / "manifest.schema.yaml"
        self._plugin_cache: dict[str, _LoadedPlugin] | None = None

    def list_supported_profiles(self) -> set[str]:
        profiles = set(_LEGACY_METRICS_PROFILES)
        profiles.update(self._load_plugins().keys())
        return profiles

    def compute_metrics(
        self,
        *,
        metrics_profile: str,
        df: pd.DataFrame,
        prefs: dict[str, Any],
        report_type_id: str,
    ) -> dict[str, Any]:
        loaded = self._load_plugins().get(metrics_profile)
        if loaded is not None:
            context = ReportPluginContext(report_type_id=report_type_id)
            return loaded.build_fn(df.copy(), dict(prefs), context)
        return compute_legacy_metrics(metrics_profile, df, prefs)

    def _load_plugins(self) -> dict[str, _LoadedPlugin]:
        if self._plugin_cache is not None:
            return self._plugin_cache

        plugins: dict[str, _LoadedPlugin] = {}
        if not self._plugin_dir.exists():
            self._plugin_cache = plugins
            return plugins

        manifest_schema = self._load_manifest_schema()
        for plugin_path in sorted(self._plugin_dir.glob("*/plugin.py")):
            manifest = self._load_plugin_manifest(plugin_path.parent, manifest_schema)
            loaded = self._load_plugin_file(plugin_path, manifest)
            if loaded.spec.metrics_profile in plugins:
                raise ValueError(f"Duplicate plugin metrics_profile '{loaded.spec.metrics_profile}'.")
            plugins[loaded.spec.metrics_profile] = loaded

        self._plugin_cache = plugins
        return plugins

    def _load_manifest_schema(self) -> dict[str, Any]:
        if not self._manifest_schema_path.exists():
            raise ValueError(f"Missing plugin manifest schema: '{self._manifest_schema_path}'.")
        raw = yaml.safe_load(self._manifest_schema_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid manifest schema at '{self._manifest_schema_path}'.")
        return raw

    def _load_plugin_manifest(self, plugin_folder: Path, schema: dict[str, Any]) -> dict[str, Any]:
        manifest_path = plugin_folder / "manifest.yaml"
        if not manifest_path.exists():
            raise ValueError(f"Plugin '{plugin_folder.name}' missing manifest.yaml.")
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"Plugin '{plugin_folder.name}' manifest.yaml must be a mapping.")
        try:
            validate(instance=manifest, schema=schema)
        except ValidationError as exc:
            raise ValueError(f"Plugin '{plugin_folder.name}' manifest validation failed: {exc.message}") from exc

        plugin_id = str(manifest.get("plugin_id", "")).strip()
        if plugin_id != plugin_folder.name:
            raise ValueError(
                f"Plugin '{plugin_folder.name}' manifest plugin_id must match folder name '{plugin_folder.name}'."
            )
        return manifest

    def _load_plugin_file(self, plugin_path: Path, manifest: dict[str, Any]) -> _LoadedPlugin:
        module_name = f"rv_reporter_plugin_{plugin_path.parent.name}"
        spec = importlib.util.spec_from_file_location(module_name, plugin_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Failed to load plugin module from '{plugin_path}'.")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return self._validate_plugin_module(module, plugin_path, manifest)

    def _validate_plugin_module(self, module: ModuleType, plugin_path: Path, manifest: dict[str, Any]) -> _LoadedPlugin:
        get_spec = getattr(module, "get_spec", None)
        build = getattr(module, "build", None)
        if not callable(get_spec) or not callable(build):
            raise ValueError(
                f"Plugin '{plugin_path.parent.name}' must expose callable get_spec() and build(df, prefs, ctx)."
            )

        raw_spec = get_spec()
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Plugin '{plugin_path.parent.name}' get_spec() must return a dict.")

        metrics_profile = str(raw_spec.get("metrics_profile", "")).strip()
        if not metrics_profile:
            raise ValueError(f"Plugin '{plugin_path.parent.name}' missing metrics_profile in get_spec().")

        api_version = int(raw_spec.get("api_version", 0) or 0)
        if api_version != PLUGIN_API_VERSION:
            raise ValueError(
                f"Plugin '{plugin_path.parent.name}' api_version {api_version} is not supported. "
                f"Expected {PLUGIN_API_VERSION}."
            )

        if metrics_profile != str(manifest["metrics_profile"]).strip():
            raise ValueError(
                f"Plugin '{plugin_path.parent.name}' metrics_profile mismatch between manifest and get_spec()."
            )
        if api_version != int(manifest["api_version"]):
            raise ValueError(f"Plugin '{plugin_path.parent.name}' api_version mismatch between manifest and get_spec().")

        title = str(raw_spec.get("title", "")).strip() or metrics_profile
        description = str(raw_spec.get("description", "")).strip()
        validated_spec = ReportPluginSpec(
            metrics_profile=metrics_profile,
            api_version=api_version,
            title=title,
            description=description,
        )
        return _LoadedPlugin(spec=validated_spec, manifest=manifest, build_fn=build)


_DEFAULT_MANAGER = ReportPluginManager()


def list_supported_metrics_profiles() -> set[str]:
    return _DEFAULT_MANAGER.list_supported_profiles()


def compute_report_metrics(
    *,
    metrics_profile: str,
    df: pd.DataFrame,
    prefs: dict[str, Any],
    report_type_id: str,
) -> dict[str, Any]:
    return _DEFAULT_MANAGER.compute_metrics(
        metrics_profile=metrics_profile,
        df=df,
        prefs=prefs,
        report_type_id=report_type_id,
    )
