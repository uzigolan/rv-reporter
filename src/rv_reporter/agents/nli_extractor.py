"""Natural Language Intent (NLI) extractor for report type selection."""

from __future__ import annotations

import json
from typing import Any

from rv_reporter.report_types.registry import ReportTypeRegistry


class NaturalLanguageIntentExtractor:
    """Extracts structured intent from natural language descriptions using an LLM."""

    def __init__(self, registry: ReportTypeRegistry | None = None) -> None:
        self._registry = registry or ReportTypeRegistry()

    def extract(
        self,
        *,
        user_description: str,
        provider: Any = None,
        model: str = "gpt-4o",
        available_report_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Extract structured intent (report_type_id, prefs) from natural language description.

        Args:
            user_description: User's natural language request (e.g., "analyze network latency issues")
            provider: Optional provider instance for LLM extraction (defaults to mock if not provided)
            model: LLM model name
            available_report_types: List of available report type IDs; if None, uses all registered types

        Returns:
            Dict with keys:
              - report_type_id: chosen report type ID (best match inferred)
              - user_prefs: dict with tone, audience, focus preferences inferred from description
              - confidence: float 0.0-1.0 indicating extraction confidence
              - reasoning: str explaining why this report type was chosen
              - notes: list of any extraction notes/warnings
        """
        available = available_report_types or self._registry.list_report_types()

        # Mock extraction: return empty state if user_description is empty or no provider
        if not user_description.strip():
            return {
                "report_type_id": "",
                "user_prefs": {},
                "confidence": 0.0,
                "reasoning": "No user description provided.",
                "notes": ["NLI extraction skipped due to empty input."],
            }

        if provider is None:
            # Fallback: simple heuristic matching
            return self._heuristic_extract(
                description=user_description,
                available_report_types=available,
            )

        # LLM-driven extraction via provider
        return self._llm_extract(
            description=user_description,
            provider=provider,
            model=model,
            available_report_types=available,
        )

    def _heuristic_extract(
        self,
        description: str,
        available_report_types: list[str],
    ) -> dict[str, Any]:
        """Simple keyword-based fallback extraction."""
        desc_lower = description.lower()
        keyword_map = {
            "network": ["network_queue_congestion", "wireshark_capture_health"],
            "queue": ["network_queue_congestion"],
            "latency": ["twamp_session_health", "wireshark_capture_health"],
            "health": ["twamp_session_health", "ms_biomarker_registry_health"],
            "biomarker": ["ms_biomarker_registry_health"],
            "performance": ["pm_export_health"],
            "pm": ["pm_export_health"],
            "jira": ["jira_issue_portfolio"],
            "issue": ["jira_issue_portfolio"],
        }

        matches = {}
        for keyword, types in keyword_map.items():
            if keyword in desc_lower:
                for rid in types:
                    if rid in available_report_types:
                        matches[rid] = matches.get(rid, 0) + 1

        if matches:
            best_match = max([(k, v) for k, v in matches.items()], key=lambda x: x[1])[0]
        else:
            best_match = available_report_types[0] if available_report_types else ""
        confidence = (matches.get(best_match, 0) / max(len(keyword_map), 1)) if best_match else 0.0

        # Extract prefs from description
        prefs = {}
        if "aggressive" in desc_lower or "strict" in desc_lower:
            prefs["tone"] = "assertive"
        if "executive" in desc_lower or "high-level" in desc_lower:
            prefs["audience"] = "executive"
        if "technical" in desc_lower or "deep-dive" in desc_lower:
            prefs["audience"] = "technical"

        notes = []
        if confidence < 0.3:
            notes.append("Low-confidence heuristic match; consider explicit report type selection.")

        return {
            "report_type_id": best_match,
            "user_prefs": prefs,
            "confidence": float(confidence),
            "reasoning": f"Heuristic keyword match found report type '{best_match}'.",
            "notes": notes,
        }

    def _llm_extract(
        self,
        description: str,
        provider: Any,
        model: str,
        available_report_types: list[str],
    ) -> dict[str, Any]:
        """LLM-driven extraction via provider."""
        # Build a focused extraction prompt
        type_list = "\n".join([f"  - {rid}" for rid in sorted(available_report_types)])
        prompt = f"""You are an expert at understanding data analysis requests and matching them to report types.

Available report types:
{type_list}

User request: "{description}"

Respond ONLY with a valid JSON object (no markdown, no explanation) with these exact keys:
{{
  "report_type_id": "chosen_report_type_id",
  "reasoning": "brief explanation of why this type matches the request",
  "confidence": 0.85,
  "tone": "neutral or assertive or conversational",
  "audience": "technical or executive or general",
  "focus": "primary area to emphasize in the report"
}}

Constraints:
- report_type_id MUST be one of the available types listed above
- confidence MUST be between 0.0 and 1.0
- All fields must be present
- Respond with ONLY valid JSON
"""

        try:
            from rv_reporter.providers.mock_provider import MockProvider

            # For now, mock extraction; in production, call provider.generate()
            if isinstance(provider, MockProvider) or str(type(provider).__name__) == "MockProvider":
                # Fallback to heuristic  
                return self._heuristic_extract(
                    description=description,
                    available_report_types=available_report_types,
                )

            # Attempt real LLM extraction (stubbed for now)
            # In a real scenario, you'd call provider methods appropriately
            # For MVP, revert to heuristic
            return self._heuristic_extract(
                description=description,
                available_report_types=available_report_types,
            )
        except Exception as e:
            # Fallback on any extraction error
            return {
                "report_type_id": "",
                "user_prefs": {},
                "confidence": 0.0,
                "reasoning": f"LLM extraction failed: {str(e)}",
                "notes": ["Extracted via heuristic fallback."],
            }
