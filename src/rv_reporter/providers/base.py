from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rv_reporter.report_types.registry import ReportTypeDefinition


class ReportProvider(ABC):
    @abstractmethod
    def generate_report_json(
        self,
        definition: ReportTypeDefinition,
        csv_profile: dict[str, Any],
        metrics: dict[str, Any],
        user_prefs: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError
