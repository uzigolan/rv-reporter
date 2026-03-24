from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from rv_reporter.report_types.registry import ReportTypeDefinition


@dataclass(slots=True)
class AgentTraceStep:
    agent: str
    status: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IntentResolution:
    report_type_id: str
    csv_path: str
    sheet_name: str | None
    row_limit: int | None
    user_prefs: dict[str, Any]
    confidence: float = 1.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReportSectionPlan:
    section_id: str
    title: str
    purpose: str
    source: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class ReportExecutionPlan:
    report_type_id: str
    report_title: str
    metrics_profile: str
    provider_name: str
    model: str
    prompt_instructions: str
    resolved_prefs: dict[str, Any]
    sections: list[ReportSectionPlan] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    trace: list[AgentTraceStep] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "report_type_id": self.report_type_id,
            "report_title": self.report_title,
            "metrics_profile": self.metrics_profile,
            "provider_name": self.provider_name,
            "model": self.model,
            "prompt_instructions": self.prompt_instructions,
            "resolved_prefs": dict(self.resolved_prefs),
            "sections": [section.as_dict() for section in self.sections],
            "notes": list(self.notes),
            "trace": [step.as_dict() for step in self.trace],
        }


@dataclass(slots=True)
class PreparedFacts:
    definition: ReportTypeDefinition
    prefs: dict[str, Any]
    csv_profile: dict[str, Any]
    metrics: dict[str, Any]
