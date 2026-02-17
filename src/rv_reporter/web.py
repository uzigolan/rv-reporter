from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, flash, redirect, render_template, request, send_file, session, url_for
from werkzeug.utils import secure_filename
import yaml

from rv_reporter.orchestrator import run_pipeline
from rv_reporter.providers.mock_provider import MockProvider
from rv_reporter.providers.openai_provider import (
    OpenAIResponsesProvider,
    build_model_prompt_for_estimation,
)
from rv_reporter.rendering.html_renderer import render_html
from rv_reporter.rendering.pdf_renderer import render_pdf
from rv_reporter.report_types.registry import ReportTypeRegistry
from rv_reporter.services.cost_estimator import MODEL_PRICING_USD, estimate_openai_cost
from rv_reporter.orchestrator import prepare_pipeline_inputs

PROTECTED_REPORT_TYPES = {"network_queue_congestion", "twamp_session_health", "pm_export_health"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env_profile(profile: str = "sandbox") -> None:
    env_path = Path(f".env.{profile}")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def create_app(config_overrides: dict[str, Any] | None = None) -> Flask:
    profile = os.getenv("APP_ENV", "sandbox")
    load_env_profile(profile)
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-key-change-me")
    app.config["UPLOAD_FOLDER"] = str(_absolute_path("uploads"))
    app.config["OUTPUT_FOLDER"] = str(_absolute_path("outputs/web"))
    app.config["REPORT_TYPES_DIR"] = str(_absolute_path("configs/report_types"))
    app.config.update(config_overrides or {})
    app.config["UPLOAD_FOLDER"] = str(_absolute_path(app.config["UPLOAD_FOLDER"]))
    app.config["OUTPUT_FOLDER"] = str(_absolute_path(app.config["OUTPUT_FOLDER"]))
    app.config["REPORT_TYPES_DIR"] = str(_absolute_path(app.config["REPORT_TYPES_DIR"]))
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["OUTPUT_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["REPORT_TYPES_DIR"]).mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def index() -> str:
        registry = ReportTypeRegistry(config_dir=Path(app.config["REPORT_TYPES_DIR"]))
        report_types = registry.list_report_types()
        default_report_type = (
            "network_queue_congestion"
            if "network_queue_congestion" in report_types
            else (report_types[0] if report_types else "")
        )
        default_provider = str(session.get("last_provider", "local"))
        if default_provider not in {"local", "openai"}:
            default_provider = "local"
        default_csv_source = str(session.get("last_csv_source", "sample"))
        if default_csv_source not in {"sample", "upload"}:
            default_csv_source = "sample"
        return render_template(
            "index.html",
            report_types=report_types,
            sample_files=_sample_files(),
            defaults={
                "provider": default_provider,
                "csv_source": default_csv_source,
                "model": "gpt-4.1-mini",
                "row_limit": 1000,
                "report_type_id": default_report_type,
            },
            priced_models=sorted(MODEL_PRICING_USD.keys()),
        )

    @app.get("/about")
    def about() -> str:
        return render_template(
            "about.html",
            protected_report_types=sorted(PROTECTED_REPORT_TYPES),
            priced_models=sorted(MODEL_PRICING_USD.keys()),
        )

    @app.get("/report-types/new")
    def new_report_type() -> str:
        return render_template(
            "new_report_type.html",
            starter_yaml=_starter_report_type_yaml(),
            supported_metrics_profiles=sorted(_supported_metrics_profiles()),
        )

    @app.get("/report-types")
    def list_report_types_page() -> str:
        registry = ReportTypeRegistry(config_dir=Path(app.config["REPORT_TYPES_DIR"]))
        items = []
        for report_type_id in registry.list_report_types():
            path = Path(app.config["REPORT_TYPES_DIR"]) / f"{report_type_id}.yaml"
            items.append(
                {
                    "report_type_id": report_type_id,
                    "path": str(path),
                    "is_protected": report_type_id in PROTECTED_REPORT_TYPES,
                }
            )
        return render_template("report_types.html", report_types=items)

    @app.post("/report-types/new")
    def create_report_type() -> Any:
        yaml_text = request.form.get("report_type_yaml", "")
        try:
            parsed = yaml.safe_load(yaml_text)
            parsed = _normalize_report_type_payload(parsed, Path(app.config["REPORT_TYPES_DIR"]))
            _validate_report_type_yaml(parsed)
            report_type_id = str(parsed["report_type_id"])
            config_dir = Path(app.config["REPORT_TYPES_DIR"])
            path = config_dir / f"{report_type_id}.yaml"
            path.write_text(yaml.safe_dump(parsed, sort_keys=False), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            flash(f"Failed to save report type: {exc}", "danger")
            return render_template(
                "new_report_type.html",
                starter_yaml=yaml_text or _starter_report_type_yaml(),
                supported_metrics_profiles=sorted(_supported_metrics_profiles()),
            )

        flash(f"Saved report type '{report_type_id}'.", "success")
        return redirect(url_for("index"))

    @app.post("/report-types/delete")
    def delete_report_type() -> Any:
        report_type_id = request.form.get("report_type_id", "").strip()
        if not report_type_id:
            flash("Missing report_type_id.", "danger")
            return redirect(url_for("list_report_types_page"))
        if report_type_id in PROTECTED_REPORT_TYPES:
            flash(f"'{report_type_id}' is protected and cannot be removed from UI.", "danger")
            return redirect(url_for("list_report_types_page"))

        path = Path(app.config["REPORT_TYPES_DIR"]) / f"{report_type_id}.yaml"
        if not path.exists():
            flash(f"Report type not found: {report_type_id}", "danger")
            return redirect(url_for("list_report_types_page"))
        path.unlink()
        flash(f"Deleted report type '{report_type_id}'.", "success")
        return redirect(url_for("list_report_types_page"))

    @app.post("/generate")
    def generate() -> Any:
        report_type_id = request.form.get("report_type_id", "").strip()
        provider_name = request.form.get("provider", "local").strip().lower()
        if provider_name == "mock":
            provider_name = "local"
        model = request.form.get("model", "gpt-4.1-mini").strip()
        confirm_openai = request.form.get("confirm_openai", "0") == "1"
        expected_cost_token = request.form.get("expected_cost_token", "").strip()
        csv_source = request.form.get("csv_source", "sample")
        sample_csv = request.form.get("sample_csv", "").strip()
        uploaded_file = request.files.get("csv_upload")
        existing_csv_path = request.form.get("existing_csv_path", "").strip()
        output_token_budget = int(request.form.get("output_token_budget", "1200").strip() or "1200")
        row_limit_raw = request.form.get("row_limit", "").strip()
        row_limit = int(row_limit_raw) if row_limit_raw else None
        session["last_provider"] = provider_name
        session["last_csv_source"] = csv_source

        prefs = {
            "tone": request.form.get("tone", "concise").strip(),
            "audience": request.form.get("audience", "leadership").strip(),
            "focus": request.form.get("focus", "trends").strip(),
        }
        threshold_name = request.form.get("threshold_name", "").strip()
        threshold_value = request.form.get("threshold_value", "").strip()
        if threshold_name and threshold_value:
            try:
                prefs[threshold_name] = float(threshold_value)
            except ValueError:
                flash("Threshold value must be numeric.", "danger")
                return redirect(url_for("index"))

        try:
            csv_path = _resolve_csv_path(
                csv_source,
                sample_csv,
                uploaded_file,
                Path(app.config["UPLOAD_FOLDER"]),
                existing_csv_path=existing_csv_path,
            )
            definition, effective_prefs, csv_profile, metrics = prepare_pipeline_inputs(
                csv_path=csv_path,
                report_type_id=report_type_id,
                user_prefs=prefs,
                row_limit=row_limit,
            )

            if provider_name == "openai" and not confirm_openai:
                prompt_text = build_model_prompt_for_estimation(
                    definition=definition,
                    csv_profile=csv_profile,
                    metrics=metrics,
                    user_prefs=effective_prefs,
                )
                estimate = estimate_openai_cost(
                    model=model,
                    prompt_text=prompt_text,
                    estimated_output_tokens=output_token_budget,
                )
                cost_token = (
                    f"{estimate['model']}|{estimate['input_tokens_est']}|"
                    f"{estimate['output_tokens_est']}|{estimate['total_cost_usd_est']}"
                )
                return render_template(
                    "confirm_cost.html",
                    estimate=estimate,
                    report_type_id=report_type_id,
                    csv_source_label=Path(csv_path).name,
                    rows_used=csv_profile.get("row_count", 0),
                    provider=provider_name,
                    model=model,
                    csv_path=csv_path,
                    row_limit=row_limit_raw,
                    output_token_budget=output_token_budget,
                    prefs=effective_prefs,
                    expected_cost_token=cost_token,
                )

            if provider_name == "openai":
                provider = OpenAIResponsesProvider(model=model)
                prompt_text = build_model_prompt_for_estimation(
                    definition=definition,
                    csv_profile=csv_profile,
                    metrics=metrics,
                    user_prefs=effective_prefs,
                )
                fresh_estimate = estimate_openai_cost(
                    model=model,
                    prompt_text=prompt_text,
                    estimated_output_tokens=output_token_budget,
                )
                fresh_token = (
                    f"{fresh_estimate['model']}|{fresh_estimate['input_tokens_est']}|"
                    f"{fresh_estimate['output_tokens_est']}|{fresh_estimate['total_cost_usd_est']}"
                )
                if expected_cost_token and expected_cost_token != fresh_token:
                    flash("Cost estimate changed after input update. Please review and confirm again.", "danger")
                    return redirect(url_for("index"))
            else:
                provider = MockProvider()

            output_dir = Path(app.config["OUTPUT_FOLDER"]) / report_type_id
            report_json_path, report_html_path = run_pipeline(
                csv_path=csv_path,
                report_type_id=report_type_id,
                user_prefs=effective_prefs,
                output_dir=output_dir,
                provider=provider,
                row_limit=row_limit,
            )
        except Exception as exc:  # noqa: BLE001
            flash(str(exc), "danger")
            return redirect(url_for("index"))

        flash("Report generated successfully.", "success")
        raw_path = report_json_path.with_name(report_json_path.name.replace(".report.json", ".openai.raw.json"))
        pdf_path = report_json_path.with_name(report_json_path.name.replace(".report.json", ".report.pdf"))
        return render_template(
            "result.html",
            report_type_id=report_type_id,
            report_json_path=str(report_json_path),
            report_html_path=str(report_html_path),
            report_pdf_path=str(pdf_path) if pdf_path.exists() else "",
            report_raw_path=str(raw_path) if raw_path.exists() else "",
            report_json=_load_json(report_json_path),
        )

    @app.get("/reports")
    def reports() -> str:
        items = []
        root = Path(app.config["OUTPUT_FOLDER"])
        timestamped_pattern = re.compile(r"\.(\d{6}_\d{4}_\d{6})\.report\.json$")
        for path in sorted(root.rglob("*.report.json")):
            match = timestamped_pattern.search(path.name)
            # Skip compatibility/latest snapshot files (<report_type>.report.json) from history list.
            if not match and re.fullmatch(r"[a-z0-9_]+\.report\.json", path.name):
                continue
            html_path = path.with_suffix(".html")
            if path.name.endswith(".report.json"):
                html_path = path.with_name(path.name.replace(".report.json", ".report.html"))
            raw_path = path.with_name(path.name.replace(".report.json", ".openai.raw.json"))
            pdf_path = path.with_name(path.name.replace(".report.json", ".report.pdf"))
            if not html_path.exists():
                try:
                    payload = _load_json(path)
                    html_path.write_text(render_html(payload), encoding="utf-8")
                except Exception:  # noqa: BLE001
                    pass
            if not pdf_path.exists():
                try:
                    payload = _load_json(path)
                    render_pdf(html_path, pdf_path, fallback_report=payload)
                except Exception:  # noqa: BLE001
                    pass
            created_at = "unknown"
            created_epoch = path.stat().st_mtime
            if match:
                dt_utc = _run_id_to_utc(match.group(1))
                created_at = _display_local_time(dt_utc)
                created_epoch = dt_utc.timestamp()
            else:
                try:
                    payload = _load_json(path)
                    meta = payload.get("metadata", {}) if isinstance(payload, dict) else {}
                    generated = str(meta.get("generated_at_utc", "")).strip()
                    if generated:
                        dt_utc = _iso_to_utc(generated)
                        created_at = _display_local_time(dt_utc)
                        created_epoch = dt_utc.timestamp()
                except Exception:  # noqa: BLE001
                    created_at = "unknown"
            items.append(
                {
                    "json_path": str(path),
                    "html_path": str(html_path),
                    "raw_path": str(raw_path) if raw_path.exists() else "",
                    "pdf_path": str(pdf_path) if pdf_path.exists() else "",
                    "name": path.stem.replace(".report", ""),
                    "created_at": created_at,
                    "created_epoch": created_epoch,
                }
            )
        items.sort(key=lambda i: i["created_epoch"], reverse=True)
        return render_template("reports.html", reports=items, output_folder=str(root))

    @app.get("/reports/json")
    def view_json() -> Any:
        json_path = request.args.get("path", "")
        path = _validate_report_artifact_path(json_path, Path(app.config["OUTPUT_FOLDER"]))
        if not path.exists():
            flash(f"File not found: {path}", "danger")
            return redirect(url_for("reports"))
        return render_template("view_json.html", file_path=str(path), payload=_load_json(path))

    @app.get("/reports/html")
    def view_html() -> Any:
        html_path = request.args.get("path", "")
        path = _validate_report_artifact_path(html_path, Path(app.config["OUTPUT_FOLDER"]))
        if not path.exists():
            flash(f"File not found: {path}", "danger")
            return redirect(url_for("reports"))
        return send_file(path)

    @app.get("/reports/raw")
    def view_raw() -> Any:
        raw_path = request.args.get("path", "")
        path = _validate_report_artifact_path(raw_path, Path(app.config["OUTPUT_FOLDER"]))
        if not path.exists():
            flash(f"Raw OpenAI output file not found: {path}", "danger")
            return redirect(url_for("reports"))
        return render_template("view_json.html", file_path=str(path), payload=_load_json(path))

    @app.get("/reports/pdf")
    def view_pdf() -> Any:
        pdf_path = request.args.get("path", "")
        path = _validate_report_artifact_path(pdf_path, Path(app.config["OUTPUT_FOLDER"]))
        if not path.exists():
            flash(f"PDF file not found: {path}", "danger")
            return redirect(url_for("reports"))
        return send_file(path)

    return app


def _sample_files() -> list[str]:
    sample_dir = _absolute_path("samples")
    return sorted(str(path) for path in sample_dir.glob("*.csv"))


def _resolve_csv_path(
    csv_source: str,
    sample_csv: str,
    uploaded_file: Any,
    upload_dir: Path,
    existing_csv_path: str = "",
) -> str:
    if existing_csv_path:
        existing = _absolute_path(existing_csv_path)
        if not existing.exists():
            raise ValueError(f"Existing CSV path not found: {existing_csv_path}")
        return str(existing)

    if csv_source == "upload":
        if uploaded_file is None or not uploaded_file.filename:
            raise ValueError("Upload mode selected but no CSV file was provided.")
        filename = secure_filename(uploaded_file.filename)
        if not filename.lower().endswith(".csv"):
            raise ValueError("Only .csv uploads are supported.")
        destination = upload_dir / filename
        uploaded_file.save(destination)
        return str(destination)

    if not sample_csv:
        raise ValueError("Sample mode selected but no sample CSV was chosen.")
    sample_path = _absolute_path(sample_csv)
    if not sample_path.exists():
        raise ValueError(f"Sample CSV not found: {sample_csv}")
    return str(sample_path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _absolute_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _validate_report_artifact_path(path_value: str, output_root: Path) -> Path:
    candidate = _absolute_path(path_value)
    root = output_root.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Invalid report artifact path.") from exc
    return candidate


def _starter_report_type_yaml() -> str:
    payload = {
        "report_type_id": "my_custom_report",
        "version": "1.0.0",
        "title": "My Custom Report",
        "required_columns": ["timestamp", "service", "requests", "errors", "latency_ms"],
        "metrics_profile": "ops_kpi",
        "default_prefs": {"tone": "concise", "audience": "leadership", "focus": "trends"},
        "prompt_instructions": "Describe key patterns and actionable recommendations.",
        "output_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "report_type_id",
                "report_title",
                "summary",
                "sections",
                "alerts",
                "recommendations",
                "tables",
                "charts",
                "metadata",
            ],
            "properties": {
                "report_type_id": {"type": "string"},
                "report_title": {"type": "string"},
                "summary": {"type": "string"},
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["title", "body"],
                        "properties": {"title": {"type": "string"}, "body": {"type": "string"}},
                    },
                },
                "alerts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["severity", "message"],
                        "properties": {"severity": {"type": "string"}, "message": {"type": "string"}},
                    },
                },
                "recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["priority", "action"],
                        "properties": {"priority": {"type": "string"}, "action": {"type": "string"}},
                    },
                },
                "tables": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                "charts": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                "metadata": {"type": "object", "additionalProperties": True},
            },
        },
    }
    return yaml.safe_dump(payload, sort_keys=False)


def _supported_metrics_profiles() -> set[str]:
    return {"ops_kpi", "finance_variance", "network_queue_congestion", "twamp_session_health", "pm_export_health"}


def _validate_report_type_yaml(payload: dict[str, Any] | None) -> None:
    if not isinstance(payload, dict):
        raise ValueError("YAML must define an object at top level.")
    required = [
        "version",
        "title",
        "required_columns",
        "metrics_profile",
        "output_schema",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")
    report_type_id = str(payload.get("report_type_id", ""))
    if not re.fullmatch(r"[a-z0-9_]+", report_type_id):
        raise ValueError("report_type_id must match [a-z0-9_]+")
    metrics_profile = str(payload["metrics_profile"])
    if metrics_profile not in _supported_metrics_profiles():
        raise ValueError(
            f"Unsupported metrics_profile '{metrics_profile}'. "
            f"Supported: {sorted(_supported_metrics_profiles())}"
        )
    if not isinstance(payload.get("required_columns"), list):
        raise ValueError("required_columns must be a list.")
    if not isinstance(payload.get("output_schema"), dict):
        raise ValueError("output_schema must be an object.")


def _display_time_from_run_id(run_id: str) -> str:
    return _display_local_time(_run_id_to_utc(run_id))


def _display_time_from_iso(iso_text: str) -> str:
    return _display_local_time(_iso_to_utc(iso_text))


def _run_id_to_utc(run_id: str) -> datetime:
    return datetime.strptime(run_id, "%y%m%d_%H%M_%f").replace(tzinfo=timezone.utc)


def _iso_to_utc(iso_text: str) -> datetime:
    normalized = iso_text.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _display_local_time(dt_utc: datetime) -> str:
    return dt_utc.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _normalize_report_type_payload(payload: dict[str, Any], report_types_dir: Path) -> dict[str, Any]:
    normalized = dict(payload)
    report_type_id = str(normalized.get("report_type_id", "")).strip()
    title = str(normalized.get("title", "")).strip()
    if not report_type_id:
        if not title:
            raise ValueError("Provide either report_type_id or title.")
        base = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        if not base:
            base = "custom_report"
        report_type_id = base
    report_type_id = re.sub(r"[^a-z0-9_]", "_", report_type_id.lower())
    report_type_id = re.sub(r"_+", "_", report_type_id).strip("_")
    if not report_type_id:
        report_type_id = "custom_report"

    candidate = report_type_id
    i = 1
    while (report_types_dir / f"{candidate}.yaml").exists():
        candidate = f"{report_type_id}_{i}"
        i += 1
    normalized["report_type_id"] = candidate
    return normalized


def main() -> None:
    profile = os.getenv("APP_ENV", "sandbox")
    load_env_profile(profile)
    app = create_app()
    debug = os.getenv("FLASK_DEBUG", "1").lower() in {"1", "true", "yes"}
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    app.run(debug=debug, host=host, port=port)


if __name__ == "__main__":
    main()
