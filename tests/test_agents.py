from __future__ import annotations

import json
from pathlib import Path

from rv_reporter.agents.coordinator import MultiAgentReportCoordinator
from rv_reporter.orchestrator import run_pipeline
from rv_reporter.providers.mock_provider import MockProvider
from rv_reporter.providers.openai_provider import build_model_input_payload
from rv_reporter.report_types.registry import ReportTypeRegistry


def test_multi_agent_coordinator_builds_trace_and_sections() -> None:
    coordinator = MultiAgentReportCoordinator(registry=ReportTypeRegistry())
    prepared, plan = coordinator.prepare_generation(
        report_type_id="twamp_session_health",
        csv_path="samples/ETX2i_twamp.csv",
        user_prefs={"focus": "anomalies", "audience": "engineering"},
        provider_name="local",
        model="local-metrics",
    )

    assert prepared.definition.report_type_id == "twamp_session_health"
    assert plan.report_type_id == "twamp_session_health"
    assert len(plan.sections) >= 3
    assert [step.agent for step in plan.trace] == ["intent_router", "report_planner", "deterministic_executor"]


def test_build_model_input_payload_includes_agent_plan() -> None:
    coordinator = MultiAgentReportCoordinator(registry=ReportTypeRegistry())
    prepared, plan = coordinator.prepare_generation(
        report_type_id="twamp_session_health",
        csv_path="samples/ETX2i_twamp.csv",
        user_prefs={"focus": "anomalies"},
        provider_name="openai",
        model="gpt-4.1-mini",
    )

    payload = build_model_input_payload(
        prepared.definition,
        prepared.csv_profile,
        prepared.metrics,
        prepared.prefs,
        agent_plan=plan.as_dict(),
    )

    assert "agent_plan" in payload
    assert payload["agent_plan"]["report_type_id"] == "twamp_session_health"
    assert payload["agent_plan"]["trace"][0]["agent"] == "intent_router"


def test_run_pipeline_uses_existing_report_type_without_agent_workflow_metadata(tmp_path: Path) -> None:
    json_path, _ = run_pipeline(
        csv_path="samples/ETX2i_twamp.csv",
        report_type_id="twamp_session_health",
        output_dir=tmp_path,
        provider=MockProvider(),
        user_prefs={"focus": "anomalies"},
        generation_context={"backend": "local", "model": "local-metrics"},
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})

    assert payload.get("report_type_id") == "twamp_session_health"
    assert metadata.get("generation_backend") == "local"
    assert "agent_workflow" not in metadata
