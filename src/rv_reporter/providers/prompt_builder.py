"""Prompt building utilities separated from provider logic."""

from __future__ import annotations

from typing import Any

from rv_reporter.report_types.registry import ReportTypeDefinition


class PromptBuilder:
    """Builds structured prompts for report generation."""

    @staticmethod
    def build_generation_prompt(
        *,
        definition: ReportTypeDefinition,
        csv_profile: dict[str, Any],
        metrics: dict[str, Any],
        user_prefs: dict[str, Any],
        agent_plan: dict[str, Any] | None = None,
    ) -> str:
        """Build a prompt instructing the model to generate a report JSON."""
        prompt_parts: list[str] = []

        # Preamble
        prompt_parts.append("You are an expert data analyst generating a structured report.\n")

        # Report type context
        prompt_parts.append(f"Report Type: {definition.title}")
        if definition.required_columns:
            prompt_parts.append(f"Required Columns: {', '.join(definition.required_columns)}")
        if definition.prompt_instructions:
            prompt_parts.append(f"Special Instructions: {definition.prompt_instructions[:100]}...")
        prompt_parts.append("")

        # Data context
        prompt_parts.append("## Data Summary")
        prompt_parts.append(f"Rows analyzed: {csv_profile.get('row_count', '?')}")
        prompt_parts.append(f"Columns: {csv_profile.get('column_count', '?')}")
        prompt_parts.append(f"Data types: {', '.join(csv_profile.get('dtypes', []))}\n")

        # Metrics context
        if metrics:
            prompt_parts.append("## Computed Metrics")
            for metric_key, metric_value in metrics.items():
                if isinstance(metric_value, dict):
                    prompt_parts.append(f"{metric_key}:")
                    for k, v in metric_value.items():
                        prompt_parts.append(f"  {k}: {v}")
                else:
                    prompt_parts.append(f"{metric_key}: {metric_value}")
            prompt_parts.append("")

        # Agent plan context
        if agent_plan and "sections" in agent_plan:
            prompt_parts.append("## Report Structure (from planning agent)")
            for section in agent_plan["sections"]:
                prompt_parts.append(f"- {section.get('title', 'Untitled')}: {section.get('purpose', 'N/A')}")
            prompt_parts.append("")

        # User preferences
        if user_prefs:
            prompt_parts.append("## Preferences")
            if user_prefs.get("tone"):
                prompt_parts.append(f"Tone: {user_prefs['tone']}")
            if user_prefs.get("audience"):
                prompt_parts.append(f"Audience: {user_prefs['audience']}")
            if user_prefs.get("focus"):
                prompt_parts.append(f"Focus: {user_prefs['focus']}")
            prompt_parts.append("")

        # Generation instructions
        prompt_parts.append("## Instructions")
        prompt_parts.append(definition.prompt_instructions or "Generate a comprehensive data report.")
        prompt_parts.append("")

        # Output format
        prompt_parts.append("## Output Format")
        prompt_parts.append(
            "Return a valid JSON object matching the report schema with keys: "
            "report_type_id, report_title, summary, sections, alerts, recommendations, tables, charts, metadata."
        )

        return "\n".join(prompt_parts)

    @staticmethod
    def build_estimation_prompt(
        *,
        definition: ReportTypeDefinition,
        csv_profile: dict[str, Any],
        metrics: dict[str, Any],
        user_prefs: dict[str, Any],
        agent_plan: dict[str, Any] | None = None,
    ) -> str:
        """Build a prompt for token cost estimation (same as generation but may be simplified)."""
        return PromptBuilder.build_generation_prompt(
            definition=definition,
            csv_profile=csv_profile,
            metrics=metrics,
            user_prefs=user_prefs,
            agent_plan=agent_plan,
        )
