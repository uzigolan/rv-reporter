from __future__ import annotations

from rv_reporter.agents.models import IntentResolution, ReportExecutionPlan, ReportSectionPlan
from rv_reporter.report_types.registry import ReportTypeDefinition


class ReportPlanningAgent:
    """Builds a structured section plan before any narrative generation occurs."""

    def build_plan(
        self,
        *,
        intent: IntentResolution,
        definition: ReportTypeDefinition,
        provider_name: str,
        model: str,
    ) -> ReportExecutionPlan:
        sections = [
            ReportSectionPlan(
                section_id="data_profile",
                title="Data Profile",
                purpose="Summarize source size, shape, and overall coverage.",
                source="csv_profile",
            ),
            ReportSectionPlan(
                section_id="key_findings",
                title="Key Findings",
                purpose="Summarize the most important computed metrics and risk signals.",
                source="metrics",
            ),
        ]

        properties = definition.output_schema.get("properties", {}) if isinstance(definition.output_schema, dict) else {}
        if "alerts" in properties:
            sections.append(
                ReportSectionPlan(
                    section_id="alerts",
                    title="Alerts",
                    purpose="Highlight threshold breaches, anomalies, and critical risks.",
                    source="metrics.alerts",
                )
            )
        if "tables" in properties:
            sections.append(
                ReportSectionPlan(
                    section_id="evidence_table",
                    title="Evidence Table",
                    purpose="Expose the structured facts that justify the narrative.",
                    source="metrics",
                )
            )
        sections.append(
            ReportSectionPlan(
                section_id="recommendations",
                title="Recommendations",
                purpose="Turn validated findings into next-step actions for the target audience.",
                source="metrics+prompt_instructions",
            )
        )

        notes = [
            "Plan built from report schema and prompt instructions.",
            f"Provider runtime selected: {provider_name}:{model}",
        ]
        if definition.prompt_instructions:
            notes.append("Report-type prompt instructions included in plan context.")

        return ReportExecutionPlan(
            report_type_id=definition.report_type_id,
            report_title=definition.title,
            metrics_profile=definition.metrics_profile,
            provider_name=provider_name,
            model=model,
            prompt_instructions=definition.prompt_instructions,
            resolved_prefs=dict(intent.user_prefs),
            sections=sections,
            notes=notes,
        )
