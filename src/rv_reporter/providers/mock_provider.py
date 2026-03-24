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
        agent_plan: dict[str, Any] | None = None,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        focus = user_prefs.get("focus", "trends")
        row_count = csv_profile.get("row_count", 0)
        summary = (
            f"{definition.title}: processed {row_count} rows (focus: {focus}). "
            f"Metrics profile: {definition.metrics_profile}. "
            "This is a local preview — use OpenAI/Anthropic provider for AI-generated narrative."
        )
        recommendations = _recommendations_for(definition.metrics_profile, metrics)

        planned_sections = _sections_from_agent_plan(agent_plan, csv_profile, metrics, definition.metrics_profile)
        metric_sections = planned_sections or _sections_from_metrics(metrics, definition.metrics_profile)

        metadata = {
            "schema_version": definition.version,
            "audience": user_prefs.get("audience", "leadership"),
            "tone": user_prefs.get("tone", "concise"),
            "preview_mode": "local (stub — AI narrative not generated)",
        }
        if isinstance(agent_plan, dict):
            metadata["agent_plan_mode"] = "multi-agent"
            metadata["agent_planned_sections"] = [str(section.get("title", "")) for section in agent_plan.get("sections", []) if isinstance(section, dict)]

        # Forward charts supplied by the plugin, if any
        plugin_charts = metrics.get("charts", [])
        if not isinstance(plugin_charts, list):
            plugin_charts = []

        # Build tables: start with any structured tables from the plugin,
        # then append the flattened metrics payload as a fallback overview.
        tables: list[dict[str, Any]] = []
        _known_table_keys = (
            "per_interval", "per_session", "top_talkers", "top_conversations",
            "protocol_breakdown", "call_volume_per_resident",
            "response_times_per_caregiver", "resolution_time_per_resident",
            "peak_demand", "caregiver_performance", "compliance_checks",
            "resident_risks", "caregiver_risks", "staffing_optimization",
            "regulatory_mapping",
        )
        for key in _known_table_keys:
            items = metrics.get(key)
            if isinstance(items, list) and items:
                tables.append({"name": key, "rows": items})
        tables.append({"name": "metrics_payload", "rows": [_flatten_metrics(metrics)]})

        return {
            "report_type_id": definition.report_type_id,
            "report_title": definition.title,
            "summary": summary,
            "sections": metric_sections,
            "alerts": metrics.get("alerts", []),
            "recommendations": recommendations,
            "tables": tables,
            "charts": plugin_charts,
            "metadata": metadata,
        }


def _flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested metric dicts to a single-level dict for display in a table."""
    result = {}
    for k, v in metrics.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_metrics(v, key))
        elif isinstance(v, list):
            result[key] = f"[{len(v)} items]"
        else:
            result[key] = v
    return result


def _sections_from_agent_plan(
    agent_plan: dict[str, Any] | None,
    csv_profile: dict[str, Any],
    metrics: dict[str, Any],
    profile: str,
) -> list[dict[str, str]]:
    if not isinstance(agent_plan, dict):
        return []

    section_specs = agent_plan.get("sections", [])
    if not isinstance(section_specs, list):
        return []

    metric_sections = _sections_from_metrics(metrics, profile)
    metric_body = metric_sections[0]["body"] if metric_sections else "Computed metrics are available in the evidence table."
    alerts = metrics.get("alerts", []) if isinstance(metrics.get("alerts"), list) else []
    alert_body = "\n".join(
        f"[{item.get('severity', '?')}] {item.get('message', '')}"
        for item in alerts[:8]
        if isinstance(item, dict)
    ) or "No computed alerts were raised for this preview run."
    bodies = {
        "data_profile": (
            f"Rows: {csv_profile.get('row_count', 0)}, columns: {csv_profile.get('column_count', 0)}, "
            f"numeric columns: {', '.join(csv_profile.get('numeric_columns', [])) or 'none'}."
        ),
        "key_findings": metric_body,
        "alerts": alert_body,
        "evidence_table": "See the computed metrics table below for the structured evidence payload.",
        "recommendations": (
            f"This preview used the local metrics engine for profile '{profile}'. "
            "Switch to an AI provider for richer narrative interpretation over the same validated facts."
        ),
    }

    sections: list[dict[str, str]] = []
    for spec in section_specs:
        if not isinstance(spec, dict):
            continue
        section_id = str(spec.get("section_id", "")).strip()
        title = str(spec.get("title", section_id or "Analysis")).strip() or "Analysis"
        body = bodies.get(section_id) or metric_body
        sections.append({"title": title, "body": body})
    return sections


def _sections_from_metrics(metrics: dict[str, Any], profile: str) -> list[dict[str, str]]:
    """Build human-readable sections from computed metrics, surfacing key values."""
    sections = []

    # Build a flat summary of top-level numeric / short-string values
    highlights = []
    for k, v in metrics.items():
        if k == "alerts":
            continue
        if isinstance(v, (int, float)):
            highlights.append(f"{k}: {v:.4g}" if isinstance(v, float) else f"{k}: {v}")
        elif isinstance(v, str) and len(v) <= 200:
            highlights.append(f"{k}: {v}")
        elif isinstance(v, dict):
            # One level deep: e.g. rssi_stats: {mean: 1.2, min: ...}
            sub = ", ".join(
                f"{sk}={sv:.4g}" if isinstance(sv, float) else f"{sk}={sv}"
                for sk, sv in v.items()
                if isinstance(sv, (int, float, str)) and not isinstance(sv, bool)
            )
            if sub:
                highlights.append(f"{k}: {{{sub}}}")

    if highlights:
        sections.append({
            "title": "Metric Highlights",
            "body": "\n".join(highlights[:30]),
        })

    alerts = metrics.get("alerts", [])
    if alerts:
        alert_lines = [
            f"[{a.get('severity','?')}] {a.get('message', str(a))}"
            for a in alerts[:10]
            if isinstance(a, dict)
        ]
        if alert_lines:
            sections.append({
                "title": "Computed Alerts",
                "body": "\n".join(alert_lines),
            })

    sections.append({
        "title": "Next Steps",
        "body": (
            f"This preview used the local metrics engine (profile: {profile}). "
            "Use an AI provider (OpenAI/Anthropic) to generate a full narrative report with "
            "root-cause analysis, contextual recommendations, and executive summary."
        ),
    })

    return sections


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
    if metrics_profile == "wireshark_capture_health":
        return [
            {"priority": "high", "action": "Investigate top talkers and top conversations for concentration or suspicious traffic."},
            {"priority": "medium", "action": "Review elevated TCP reset patterns and correlate with policy/firewall logs."},
            {"priority": "medium", "action": "Inspect packet-rate spikes and protocol shifts across time buckets."},
        ]
    if metrics_profile == "twamp_session_threshold_sla_compliance":
        summary = metrics.get("summary", {})
        verdict = str(summary.get("overall_verdict", "good"))
        compliance_pct = float(summary.get("sla_compliance_pct", 100))
        idle = int(summary.get("idle_intervals", 0))
        recs = []
        if verdict == "critical":
            recs.append({"priority": "high", "action": "Critical SLA violations detected — investigate affected intervals for root cause (path congestion, peer outage, QoS misconfiguration)."})
        if compliance_pct < 90:
            recs.append({"priority": "high", "action": f"SLA compliance at {compliance_pct:.1f}% — escalate to network operations and review threshold definitions."})
        if idle > 0:
            recs.append({"priority": "medium", "action": f"{idle} idle interval(s) with zero traffic — verify TWAMP sender configuration and session provisioning."})
        recs.extend([
            {"priority": "medium", "action": "Correlate delay and jitter spikes with traffic load, topology changes, or maintenance windows."},
            {"priority": "medium", "action": "Review forward vs. backward path asymmetry in delay metrics for routing issues."},
            {"priority": "low", "action": "Validate TWAMP test parameters (interval, packet size, DSCP) match production SLA contract."},
        ])
        return recs
    if metrics_profile == "telecom_session_health_report":
        summary = metrics.get("summary", {})
        critical = int(summary.get("critical_sessions", 0))
        sla_pct = float(summary.get("sla_compliance_pct", 100))
        recs = []
        if critical > 0:
            recs.append({"priority": "high", "action": f"Investigate {critical} critical session(s) — check peer routing, QoS policy, and network path congestion."})
        if sla_pct < 90:
            recs.append({"priority": "high", "action": f"SLA compliance at {sla_pct:.1f}% — escalate to network operations and review SLA contract thresholds."})
        recs.extend([
            {"priority": "medium", "action": "Correlate degraded delay intervals with upstream traffic bursts or topology changes."},
            {"priority": "medium", "action": "Review IPDV/PDV trends for sessions approaching the PDV SLA limit."},
            {"priority": "low", "action": "Verify test sender configuration (interval, packet size, DSCP marking) matches production expectations."},
        ])
        return recs
    if metrics_profile == "ai_optimized_assisted_living_sla_compliance" or "assisted_living" in metrics_profile:
        summary = metrics.get("summary", {})
        risk = str(summary.get("overall_risk_level", "Low"))
        incomplete = int(summary.get("incomplete_incidents", 0))
        recs = []
        if risk in ("Moderate", "High"):
            recs.append({"priority": "high", "action": "Increase caregiver staffing during peak demand windows to improve response SLA."})
        if incomplete > 0:
            recs.append({"priority": "high", "action": f"{incomplete} incomplete incident(s) detected — enforce mandatory four-step event flow closure."})
        recs.extend([
            {"priority": "high", "action": "Set an internal response alert threshold at 5 minutes to catch SLA drift early."},
            {"priority": "medium", "action": "Review care plans for high-frequency residents to determine if repeated calls reflect higher-acuity needs."},
            {"priority": "medium", "action": "Use top-performing caregiver as workflow benchmark for acceptance speed and handling consistency."},
            {"priority": "medium", "action": "Maintain dual-confirmation closure policy to strengthen audit defensibility."},
            {"priority": "low", "action": "Consider automated escalation alerts when response time exceeds 80% of threshold."},
        ])
        return recs
    return [{"priority": "medium", "action": "Review computed metrics with domain owner."}]
