from __future__ import annotations

from typing import Any

from rv_reporter.providers.base import ReportProvider
from rv_reporter.report_types.registry import ReportTypeDefinition


class MockProvider(ReportProvider):
    def generate_report_json(
        self,
        definition: ReportTypeDefinition,
        csv_profile: dict[str, Any],
        metrics: dict[str, Any],
        user_prefs: dict[str, Any],
    ) -> dict[str, Any]:
        focus = user_prefs.get("focus", "trends")
        summary = f"{definition.title}: generated from {csv_profile['row_count']} rows with focus on {focus}."
        recommendations = _recommendations_for(definition.metrics_profile, metrics)

        return {
            "report_type_id": definition.report_type_id,
            "report_title": definition.title,
            "summary": summary,
            "sections": [
                {
                    "title": "Data Profile",
                    "body": (
                        f"Rows: {csv_profile['row_count']}, columns: {csv_profile['column_count']}, "
                        f"numeric columns: {', '.join(csv_profile['numeric_columns']) or 'none'}."
                    ),
                },
                {
                    "title": "Metric Highlights",
                    "body": f"Computed metrics profile: {definition.metrics_profile}.",
                },
            ],
            "alerts": metrics.get("alerts", []),
            "recommendations": recommendations,
            "tables": [
                {
                    "name": "metrics_payload",
                    "rows": [metrics],
                }
            ],
            "charts": [],
            "metadata": {
                "schema_version": definition.version,
                "audience": user_prefs.get("audience", "leadership"),
                "tone": user_prefs.get("tone", "concise"),
            },
        }


def _recommendations_for(metrics_profile: str, metrics: dict[str, Any]) -> list[dict[str, str]]:
    if metrics_profile == "ops_kpi":
        return [
            {"priority": "high", "action": "Investigate services above error-rate threshold."},
            {"priority": "medium", "action": "Review latency p95 hotspots and scale where needed."},
        ]
    if metrics_profile == "finance_variance":
        return [
            {"priority": "high", "action": "Address line items with sustained adverse variance."},
            {"priority": "medium", "action": "Rebaseline budget assumptions for volatile categories."},
        ]
    if metrics_profile == "network_queue_congestion":
        return [
            {"priority": "high", "action": "Investigate and tune queues with highest mean drop ratio."},
            {"priority": "medium", "action": "Validate QoS policy mapping for affected interfaces."},
            {"priority": "medium", "action": "Correlate high-drop intervals with traffic bursts and capacity limits."},
        ]
    if metrics_profile == "twamp_session_health":
        return [
            {"priority": "high", "action": "Investigate intervals with peak discard rate and packet discard counts."},
            {"priority": "medium", "action": "Correlate yellow traffic share with discard spikes to tune policies."},
            {"priority": "medium", "action": "Track TWAMP delay/IPDV trend for stability degradation signals."},
        ]
    if metrics_profile == "pm_export_health":
        return [
            {"priority": "high", "action": "Prioritize interfaces with highest discard delta and verify queue/policy settings."},
            {"priority": "high", "action": "If CRC errors grow, run physical link checks (optic/cable/port health)."},
            {"priority": "medium", "action": "Review CPU/memory/disk headroom to reduce operational risk."},
        ]
    if metrics_profile == "jira_issue_portfolio":
        return [
            {"priority": "high", "action": "Prioritize projects with high backlog ratio and low closure ratio."},
            {"priority": "high", "action": "Escalate assignees with highest active issue counts and stale unresolved items."},
            {"priority": "high", "action": "Triage oldest active issues and assign closure owners with deadlines."},
            {"priority": "medium", "action": "Balance assignee workload where responsibility concentration is high."},
        ]
    if metrics_profile == "ms_biomarker_registry_health":
        return [
            {"priority": "high", "action": "Stabilize biomarker capture for columns with highest missingness before inference."},
            {"priority": "high", "action": "Prioritize follow-up completion for participants with baseline-only records."},
            {"priority": "medium", "action": "Standardize disease-course labels to reduce taxonomy drift (e.g., RRMS variants)."},
            {"priority": "medium", "action": "Validate date fields and enforce consistent date formats at ingestion."},
        ]
    return [{"priority": "medium", "action": "Review computed metrics with domain owner."}]
