from __future__ import annotations

from typing import Any

from rv_reporter.agents.models import IntentResolution
from rv_reporter.agents.nli_extractor import NaturalLanguageIntentExtractor
from rv_reporter.report_types.registry import ReportTypeRegistry


class IntentRoutingAgent:
    """Normalizes report requests via explicit ID or natural language description into structured intent."""

    def __init__(self, registry: ReportTypeRegistry | None = None) -> None:
        self._registry = registry or ReportTypeRegistry()
        self._nli_extractor = NaturalLanguageIntentExtractor(registry=self._registry)

    def resolve(
        self,
        *,
        report_type_id: str = "",
        csv_path: str = "",
        user_description: str = "",
        user_prefs: dict[str, Any] | None = None,
        sheet_name: str | None = None,
        row_limit: int | None = None,
    ) -> IntentResolution:
        """
        Resolve intent from explicit report type ID or natural language description.

        Args:
            report_type_id: Explicit report type ID (preferred)
            csv_path: Path to CSV/Excel data file
            user_description: Natural language request if report_type_id not provided
            user_prefs: User preferences (tone, audience, focus)
            sheet_name: Excel sheet name (optional)
            row_limit: Maximum rows to process (optional)

        Returns:
            IntentResolution with normalized report_type_id and prefs
        """
        normalized_prefs = dict(user_prefs or {})
        notes: list[str] = []

        # If explicit report_type_id provided, use it
        if report_type_id and report_type_id.strip():
            notes.append("Intent resolved from explicit report type ID selection.")
        # Otherwise, try to extract from natural language description
        elif user_description and user_description.strip():
            nli_result = self._nli_extractor.extract(user_description=user_description)
            if nli_result.get("report_type_id"):
                report_type_id = nli_result["report_type_id"]
                notes.append(f"NLI extraction (confidence: {nli_result.get('confidence', 0):.2f}): {nli_result.get('reasoning', '')}")
                # Merge NLI-extracted prefs with user-provided prefs
                for key in ("tone", "audience", "focus"):
                    if key in nli_result and not normalized_prefs.get(key):
                        normalized_prefs[key] = nli_result[key]
            else:
                notes.append("NLI extraction returned no match; please provide explicit report type ID.")
        else:
            notes.append("No report type ID or description provided.")

        # Add any explicit pref notes
        if normalized_prefs.get("tone"):
            notes.append("Tone preference supplied by caller.")
        if normalized_prefs.get("audience"):
            notes.append("Audience preference supplied by caller.")
        if normalized_prefs.get("focus"):
            notes.append("Focus preference supplied by caller.")

        return IntentResolution(
            report_type_id=str(report_type_id).strip(),
            csv_path=str(csv_path),
            sheet_name=sheet_name or None,
            row_limit=row_limit,
            user_prefs=normalized_prefs,
            confidence=1.0,
            notes=notes,
        )
