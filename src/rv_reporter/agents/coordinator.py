from __future__ import annotations

from typing import Any

from rv_reporter.agents.execution_agent import DeterministicExecutionAgent
from rv_reporter.agents.intent_agent import IntentRoutingAgent
from rv_reporter.agents.models import AgentTraceStep, PreparedFacts, ReportExecutionPlan
from rv_reporter.agents.planning_agent import ReportPlanningAgent
from rv_reporter.agents.writer_agent import NarrativeWriterAgent
from rv_reporter.providers.base import ReportProvider
from rv_reporter.providers.prompt_builder import PromptBuilder
from rv_reporter.report_types.registry import ReportTypeRegistry


class MultiAgentReportCoordinator:
    """Coordinates specialized agents while preserving the deterministic pipeline as source of truth."""

    def __init__(
        self,
        *,
        registry: ReportTypeRegistry | None = None,
        intent_agent: IntentRoutingAgent | None = None,
        planning_agent: ReportPlanningAgent | None = None,
        execution_agent: DeterministicExecutionAgent | None = None,
        writer_agent: NarrativeWriterAgent | None = None,
    ) -> None:
        self._registry = registry or ReportTypeRegistry()
        self._intent_agent = intent_agent or IntentRoutingAgent()
        self._planning_agent = planning_agent or ReportPlanningAgent()
        self._execution_agent = execution_agent or DeterministicExecutionAgent()
        self._writer_agent = writer_agent or NarrativeWriterAgent()

    def prepare_generation(
        self,
        *,
        report_type_id: str = "",
        csv_path: str,
        user_prefs: dict[str, Any] | None,
        provider_name: str,
        model: str,
        user_description: str = "",
        row_limit: int | None = None,
        sheet_name: str | None = None,
        ignore_columns: list[str] | None = None,
    ) -> tuple[PreparedFacts, ReportExecutionPlan]:
        intent = self._intent_agent.resolve(
            report_type_id=report_type_id,
            csv_path=csv_path,
            user_description=user_description,
            user_prefs=user_prefs,
            sheet_name=sheet_name,
            row_limit=row_limit,
        )
        definition = self._registry.get(intent.report_type_id)
        plan = self._planning_agent.build_plan(
            intent=intent,
            definition=definition,
            provider_name=provider_name,
            model=model,
        )
        plan.trace.append(
            AgentTraceStep(
                agent="intent_router",
                status="completed",
                summary=f"Resolved explicit request to report type '{intent.report_type_id}'.",
                details=intent.as_dict(),
            )
        )
        plan.trace.append(
            AgentTraceStep(
                agent="report_planner",
                status="completed",
                summary=f"Planned {len(plan.sections)} report sections for metrics profile '{definition.metrics_profile}'.",
                details={"sections": [section.as_dict() for section in plan.sections]},
            )
        )
        prepared = self._execution_agent.prepare_facts(
            intent=intent,
            definition=definition,
            ignore_columns=ignore_columns,
        )
        plan.trace.append(
            AgentTraceStep(
                agent="deterministic_executor",
                status="completed",
                summary=(
                    f"Prepared facts from {prepared.csv_profile.get('row_count', 0)} rows and "
                    f"computed metrics profile '{definition.metrics_profile}'."
                ),
                details={
                    "row_count": prepared.csv_profile.get("row_count", 0),
                    "column_count": prepared.csv_profile.get("column_count", 0),
                },
            )
        )
        return prepared, plan

    def generate_report_json(
        self,
        *,
        report_type_id: str,
        csv_path: str,
        user_prefs: dict[str, Any] | None,
        provider: ReportProvider,
        provider_name: str,
        model: str,
        row_limit: int | None = None,
        sheet_name: str | None = None,
        ignore_columns: list[str] | None = None,
    ) -> tuple[PreparedFacts, ReportExecutionPlan, dict]:
        prepared, plan = self.prepare_generation(
            report_type_id=report_type_id,
            csv_path=csv_path,
            user_prefs=user_prefs,
            provider_name=provider_name,
            model=model,
            row_limit=row_limit,
            sheet_name=sheet_name,
            ignore_columns=ignore_columns,
        )
        report_json = self._writer_agent.generate(provider=provider, prepared=prepared, plan=plan)
        plan.trace.append(
            AgentTraceStep(
                agent="narrative_writer",
                status="completed",
                summary=f"Provider '{provider_name}' produced report JSON.",
                details={"provider_name": provider_name, "model": model},
            )
        )
        return prepared, plan, report_json
