from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rv_reporter.providers.base import ReportProvider
from rv_reporter.providers.mock_provider import MockProvider
from rv_reporter.rendering.html_renderer import render_html
from rv_reporter.rendering.pdf_renderer import render_pdf
from rv_reporter.report_types.registry import ReportTypeRegistry
from rv_reporter.services.ingest import load_csv_with_limit, validate_required_columns
from rv_reporter.services.metrics import compute_metrics
from rv_reporter.services.profiler import profile_dataframe
from rv_reporter.services.validator import validate_report_schema


def prepare_pipeline_inputs(
    csv_path: str | Path,
    report_type_id: str,
    user_prefs: dict[str, Any] | None = None,
    registry: ReportTypeRegistry | None = None,
    row_limit: int | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    report_registry = registry or ReportTypeRegistry()
    definition = report_registry.get(report_type_id)

    prefs = dict(definition.default_prefs)
    prefs.update(user_prefs or {})

    df = load_csv_with_limit(csv_path, row_limit=row_limit)
    validate_required_columns(df, definition.required_columns)
    csv_profile = profile_dataframe(df)
    metrics = compute_metrics(definition.metrics_profile, df, prefs)
    return definition, prefs, csv_profile, metrics


def run_pipeline(
    csv_path: str | Path,
    report_type_id: str,
    user_prefs: dict[str, Any] | None = None,
    output_dir: str | Path = "outputs",
    registry: ReportTypeRegistry | None = None,
    provider: ReportProvider | None = None,
    row_limit: int | None = None,
    generation_context: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    definition, prefs, csv_profile, metrics = prepare_pipeline_inputs(
        csv_path=csv_path,
        report_type_id=report_type_id,
        user_prefs=user_prefs,
        registry=registry,
        row_limit=row_limit,
    )

    report_provider = provider or MockProvider()
    report_json = report_provider.generate_report_json(definition, csv_profile, metrics, prefs)
    report_json = _normalize_report_output(report_json, definition.report_type_id, definition.title)
    _ensure_metrics_payload_for_charted_reports(report_json, definition.report_type_id, metrics)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc)
    run_id = generated_at.strftime("%y%m%d_%H%M_%f")
    _stamp_report_metadata(report_json, run_id=run_id, generated_at=generated_at)
    _stamp_generation_metadata(report_json, generation_context or {})
    validate_report_schema(report_json, definition.output_schema)
    json_content = json.dumps(report_json, indent=2)
    html_content = render_html(report_json)

    json_file = output_path / f"{report_type_id}.{run_id}.report.json"
    html_file = output_path / f"{report_type_id}.{run_id}.report.html"
    latest_json_file = output_path / f"{report_type_id}.report.json"
    latest_html_file = output_path / f"{report_type_id}.report.html"
    raw_file = output_path / f"{report_type_id}.{run_id}.openai.raw.json"
    latest_raw_file = output_path / f"{report_type_id}.openai.raw.json"
    pdf_file = output_path / f"{report_type_id}.{run_id}.report.pdf"
    latest_pdf_file = output_path / f"{report_type_id}.report.pdf"

    json_file.write_text(json_content, encoding="utf-8")
    html_file.write_text(html_content, encoding="utf-8")
    latest_json_file.write_text(json_content, encoding="utf-8")
    latest_html_file.write_text(html_content, encoding="utf-8")
    pdf_ok = render_pdf(html_file, pdf_file, fallback_report=report_json)
    if pdf_ok:
        render_pdf(latest_html_file, latest_pdf_file, fallback_report=report_json)
    raw_payload = getattr(report_provider, "last_raw_response", None)
    if isinstance(raw_payload, dict):
        raw_content = json.dumps(raw_payload, indent=2)
        raw_file.write_text(raw_content, encoding="utf-8")
        latest_raw_file.write_text(raw_content, encoding="utf-8")
    return json_file, html_file


def _normalize_report_output(report_json: dict[str, Any], report_type_id: str, report_title: str) -> dict[str, Any]:
    normalized = dict(report_json or {})
    normalized.setdefault("report_type_id", report_type_id)
    normalized.setdefault("report_title", report_title)
    normalized.setdefault("summary", "")

    for key in ("sections", "alerts", "recommendations", "tables", "charts"):
        value = normalized.get(key)
        if not isinstance(value, list):
            normalized[key] = []

    metadata_value = normalized.get("metadata")
    if not isinstance(metadata_value, dict):
        normalized["metadata"] = {}

    return normalized


def _stamp_report_metadata(report_json: dict[str, Any], run_id: str, generated_at: datetime) -> None:
    metadata = report_json.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        report_json["metadata"] = metadata
    metadata["report_id"] = f"{report_json.get('report_type_id', 'report')}.{run_id}"
    metadata["generated_at_utc"] = generated_at.isoformat()


def _stamp_generation_metadata(report_json: dict[str, Any], generation_context: dict[str, Any]) -> None:
    metadata = report_json.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        report_json["metadata"] = metadata
    if generation_context.get("backend"):
        metadata["generation_backend"] = str(generation_context["backend"])
    if generation_context.get("model"):
        metadata["generation_model"] = str(generation_context["model"])
    if generation_context.get("source_csv"):
        metadata["source_csv"] = str(generation_context["source_csv"])
    if generation_context.get("source_rows_used") is not None:
        metadata["source_rows_used"] = int(generation_context["source_rows_used"])


def _ensure_metrics_payload_for_charted_reports(
    report_json: dict[str, Any],
    report_type_id: str,
    metrics: dict[str, Any],
) -> None:
    if report_type_id not in {"network_queue_congestion", "twamp_session_health", "pm_export_health"}:
        return
    tables = report_json.get("tables")
    if not isinstance(tables, list):
        tables = []
        report_json["tables"] = tables

    payload_table = {"name": "metrics_payload", "rows": [metrics]}
    existing_idx = next(
        (idx for idx, table in enumerate(tables) if isinstance(table, dict) and table.get("name") == "metrics_payload"),
        None,
    )
    if existing_idx is None:
        tables.insert(0, payload_table)
    else:
        tables[existing_idx] = payload_table
