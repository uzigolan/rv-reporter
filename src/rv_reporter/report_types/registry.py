from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ReportTypeDefinition:
    report_type_id: str
    version: str
    title: str
    required_columns: list[str]
    metrics_profile: str
    default_prefs: dict[str, Any]
    output_schema: dict[str, Any]
    prompt_instructions: str


class ReportTypeRegistry:
    def __init__(self, config_dir: str | Path = "configs/report_types") -> None:
        self._config_dir = Path(config_dir)
        self._cache: dict[str, ReportTypeDefinition] = {}

    def list_report_types(self) -> list[str]:
        return sorted(path.stem for path in self._config_dir.glob("*.yaml"))

    def get(self, report_type_id: str) -> ReportTypeDefinition:
        if report_type_id in self._cache:
            return self._cache[report_type_id]

        path = self._config_dir / f"{report_type_id}.yaml"
        if not path.exists():
            raise ValueError(f"Unknown report_type_id '{report_type_id}'.")

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        definition = ReportTypeDefinition(
            report_type_id=raw["report_type_id"],
            version=raw["version"],
            title=raw["title"],
            required_columns=raw["required_columns"],
            metrics_profile=raw["metrics_profile"],
            default_prefs=raw.get("default_prefs", {}),
            output_schema=raw["output_schema"],
            prompt_instructions=raw.get("prompt_instructions", ""),
        )
        self._cache[report_type_id] = definition
        return definition
