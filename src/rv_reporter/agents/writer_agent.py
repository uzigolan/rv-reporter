from __future__ import annotations

from rv_reporter.agents.models import PreparedFacts, ReportExecutionPlan
from rv_reporter.providers.base import ReportProvider
from rv_reporter.providers.prompt_builder import PromptBuilder


class NarrativeWriterAgent:
    """Delegates narrative generation to the selected provider using a validated execution plan."""

    def generate(
        self,
        *,
        provider: ReportProvider,
        prepared: PreparedFacts,
        plan: ReportExecutionPlan,
    ) -> dict:
        """Generate narrative using provider with structured prompt from PromptBuilder."""
        # Build structured prompt using the PromptBuilder
        prompt = PromptBuilder.build_generation_prompt(
            definition=prepared.definition,
            csv_profile=prepared.csv_profile,
            metrics=prepared.metrics,
            user_prefs=prepared.prefs,
            agent_plan=plan.as_dict(),
        )

        # Delegate to provider with prompt and plan
        return provider.generate_report_json(
            prepared.definition,
            prepared.csv_profile,
            prepared.metrics,
            prepared.prefs,
            agent_plan=plan.as_dict(),
            prompt=prompt,
        )
