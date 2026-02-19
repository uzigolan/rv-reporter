from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any
from urllib import parse as urlparse, request as urlrequest
from urllib.error import HTTPError, URLError
from uuid import uuid4

from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, session, url_for
from markupsafe import Markup
import markdown
from werkzeug.utils import secure_filename
import yaml

from rv_reporter.orchestrator import run_pipeline
from rv_reporter.providers.anthropic_provider import AnthropicMessagesProvider
from rv_reporter.providers.mock_provider import MockProvider
from rv_reporter.providers.openai_chat_provider import OpenAIChatCompletionsProvider
from rv_reporter.providers.openai_provider import (
    OpenAIResponsesProvider,
    build_model_prompt_for_estimation,
)
from rv_reporter.rendering.html_renderer import render_html
from rv_reporter.rendering.pdf_renderer import render_pdf
from rv_reporter.report_types.registry import ReportTypeRegistry
from rv_reporter.services.cost_estimator import (
    MODEL_PRICING_USD,
    PRICING_SOURCE_URL,
    PRICING_VERIFIED_DATE,
    estimate_openai_cost,
    estimate_tokens,
)
from rv_reporter.orchestrator import prepare_pipeline_inputs
from rv_reporter.services.ingest import describe_tabular_source, list_excel_sheets

PROTECTED_REPORT_TYPES = {
    "network_queue_congestion",
    "twamp_session_health",
    "pm_export_health",
    "jira_issue_portfolio",
    "ms_biomarker_registry_health",
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOC_PAGES = {
    "install": ("INSTALL.md", "Install Guide"),
    "architecture": ("docs/architecture.md", "Architecture"),
    "ui-guide": ("docs/UI_GUIDE.md", "UI Guide"),
}
PROVIDER_CATALOG: dict[str, dict[str, Any]] = {
    "local": {
        "label": "local",
        "env_key": "",
        "default_base_url": "",
        "cost_estimate": False,
        "runtime": "local",
        "model_options": ["local-metrics"],
    },
    "openai": {
        "label": "openai",
        "env_key": "OPENAI_API_KEY",
        "default_base_url": "",
        "cost_estimate": True,
        "runtime": "responses",
        "model_options": [
            "gpt-5.2",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5.2-pro",
            "gpt-5",
            "gpt-4.1",
        ],
    },
    "xai": {
        "label": "xai (grok)",
        "env_key": "XAI_API_KEY",
        "default_base_url": "https://api.x.ai/v1",
        "cost_estimate": False,
        "runtime": "chat_compat",
        "model_options": ["grok-4", "grok-3", "grok-3-mini"],
    },
    "gemini": {
        "label": "google (gemini)",
        "env_key": "GEMINI_API_KEY",
        "default_base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "cost_estimate": False,
        "runtime": "chat_compat",
        "model_options": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
    },
    "claude": {
        "label": "anthropic (claude)",
        "env_key": "ANTHROPIC_API_KEY",
        "default_base_url": "https://api.anthropic.com/v1",
        "cost_estimate": False,
        "runtime": "anthropic",
        "model_options": ["claude-sonnet-4-5", "claude-opus-4-1", "claude-3-7-sonnet-latest"],
    },
    "openrouter": {
        "label": "openrouter",
        "env_key": "OPENROUTER_API_KEY",
        "default_base_url": "https://openrouter.ai/api/v1",
        "cost_estimate": False,
        "runtime": "chat_compat",
        "model_options": [
            "deepseek/deepseek-chat-v3.1",
            "deepseek/deepseek-v3.2",
            "qwen/qwen-max",
            "qwen/qwen3-max",
            "qwen/qwen-plus-2025-07-28",
            "mistralai/mistral-large-2512",
            "mistralai/mistral-medium-3.1",
            "cohere/command-a",
            "cohere/command-r-plus-08-2024",
            "meta-llama/llama-3.3-70b-instruct",
            "moonshotai/kimi-k2.5",
            "minimax/minimax-01",
        ],
    },
}

OPENROUTER_RECOMMENDED_MODELS: list[str] = [
    "deepseek/deepseek-chat-v3.1",
    "deepseek/deepseek-v3.2",
    "qwen/qwen-max",
    "qwen/qwen3-max",
    "qwen/qwen-plus-2025-07-28",
    "mistralai/mistral-large-2512",
    "mistralai/mistral-medium-3.1",
    "cohere/command-a",
    "cohere/command-r-plus-08-2024",
    "meta-llama/llama-3.3-70b-instruct",
    "moonshotai/kimi-k2.5",
    "minimax/minimax-01",
]

OPENAI_FRONTIER_MODEL_LABELS: dict[str, str] = {
    "gpt-5.2": "GPT-5.2",
    "gpt-5-mini": "GPT-5 mini",
    "gpt-5-nano": "GPT-5 nano",
    "gpt-5.2-pro": "GPT-5.2 pro",
    "gpt-5": "GPT-5",
    "gpt-4.1": "GPT-4.1",
}

PROVIDER_PRICING_INFO: dict[str, dict[str, str]] = {
    "openai": {"pricing_url": "https://platform.openai.com/docs/pricing"},
    "claude": {"pricing_url": "https://www.anthropic.com/pricing#api"},
    "gemini": {"pricing_url": "https://ai.google.dev/gemini-api/docs/pricing"},
    "xai": {"pricing_url": "https://docs.x.ai/docs/models"},
    "openrouter": {"pricing_url": "https://openrouter.ai/models"},
}

# Non-OpenAI pricing snapshots from official provider pricing docs (USD per 1M tokens).
# Used for cross-provider ratio display in UI only.
NON_OPENAI_MODEL_PRICING_USD: dict[str, tuple[float, float, str]] = {
    # Anthropic (API pricing page; standard tier values shown)
    "claude-sonnet-4": (3.0, 15.0, "https://www.anthropic.com/pricing#api"),
    "claude-sonnet-4-5": (3.0, 15.0, "https://www.anthropic.com/pricing#api"),
    "claude-sonnet-4-6": (3.0, 15.0, "https://www.anthropic.com/pricing#api"),
    "claude-opus-4": (15.0, 75.0, "https://www.anthropic.com/pricing#api"),
    "claude-opus-4-1": (15.0, 75.0, "https://www.anthropic.com/pricing#api"),
    "claude-opus-4-5": (5.0, 25.0, "https://www.anthropic.com/pricing#api"),
    "claude-opus-4-6": (5.0, 25.0, "https://www.anthropic.com/pricing#api"),
    "claude-haiku-4-5": (1.0, 5.0, "https://www.anthropic.com/pricing#api"),
    "claude-3-haiku": (0.25, 1.25, "https://www.anthropic.com/pricing#api"),
    "claude-3-7-sonnet-latest": (3.0, 15.0, "https://www.anthropic.com/pricing#api"),
    # Google Gemini (text/image/video prices; not audio-specific tiers)
    "gemini-2.5-pro": (1.25, 10.0, "https://ai.google.dev/gemini-api/docs/pricing"),
    "gemini-2.5-flash": (0.30, 2.50, "https://ai.google.dev/gemini-api/docs/pricing"),
    "gemini-2.0-flash": (0.10, 0.40, "https://ai.google.dev/gemini-api/docs/pricing"),
    # xAI (language model token pricing)
    "grok-4": (3.0, 15.0, "https://docs.x.ai/developers/models"),
    "grok-3": (3.0, 15.0, "https://docs.x.ai/developers/models"),
    "grok-3-mini": (0.30, 0.50, "https://docs.x.ai/developers/models"),
}
OPENROUTER_DYNAMIC_MODEL_PRICING_USD: dict[str, tuple[float, float, str]] = {}


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
        report_types = _visible_report_types_for_generation(registry.list_report_types())
        default_report_type = (
            "network_queue_congestion"
            if "network_queue_congestion" in report_types
            else (report_types[0] if report_types else "")
        )
        default_provider = str(session.get("last_provider", "local"))
        if default_provider not in PROVIDER_CATALOG:
            default_provider = "local"
        return render_template(
            "index.html",
            report_types=report_types,
            report_type_options=_report_type_options(report_types),
            recent_uploads=_recent_uploaded_sources(Path(app.config["UPLOAD_FOLDER"])),
            sheet_options=[],
            selected_sheet="",
            existing_csv_path="",
            defaults={
                "provider": default_provider,
                "model": "gpt-5-mini",
                "row_limit": 1000,
                "report_type_id": default_report_type,
                "api_key": "",
                "api_base_url": "",
            },
            priced_models=_openai_frontier_models(),
            model_options=_provider_default_options("openai"),
            provider_options=_provider_options(),
            provider_model_options=_provider_model_options(),
            provider_ids=list(PROVIDER_CATALOG.keys()),
        )

    @app.get("/api/excel-sheets")
    def excel_sheets() -> Any:
        path_value = request.args.get("path", "").strip()
        if not path_value:
            return jsonify({"sheets": []})
        path = _absolute_path(path_value)
        if not path.exists():
            return jsonify({"sheets": []})
        if path.suffix.lower() not in {".xlsx", ".xls"}:
            return jsonify({"sheets": []})
        sheets = list_excel_sheets(path)
        return jsonify({"sheets": sheets})

    @app.get("/api/source-metadata")
    def source_metadata() -> Any:
        path_value = request.args.get("path", "").strip()
        sheet_name = request.args.get("sheet_name", "").strip()
        if not path_value:
            return jsonify({"file_type": "unknown", "sheets": [], "selected_sheet": "", "columns": []})
        path = _absolute_path(path_value)
        if not path.exists():
            return jsonify({"file_type": "unknown", "sheets": [], "selected_sheet": "", "columns": []})
        return jsonify(describe_tabular_source(path, sheet_name=sheet_name or None))

    @app.get("/api/generation-status")
    def generation_status() -> Any:
        root = Path(app.config["OUTPUT_FOLDER"])
        return jsonify(_generation_status_snapshot(root))

    @app.get("/api/provider-models")
    def provider_models() -> Any:
        provider_name = request.args.get("provider", "").strip().lower()
        if provider_name not in PROVIDER_CATALOG:
            return jsonify({"options": [], "error": f"Unknown provider '{provider_name}'."}), 400

        if provider_name == "local":
            return jsonify({"options": [{"value": "local-metrics", "label": "local-metrics / 0x"}], "error": ""})

        api_key_override = request.args.get("api_key", "").strip()
        api_base_override = request.args.get("api_base_url", "").strip()

        spec = PROVIDER_CATALOG[provider_name]
        env_key = str(spec.get("env_key", "")).strip()
        api_key = api_key_override or (os.getenv(env_key, "").strip() if env_key else "")
        base_url = api_base_override or _provider_default_base_url(provider_name)

        if not api_key and provider_name != "openrouter":
            fallback = _provider_default_options(provider_name)
            return jsonify(
                {
                    "options": fallback,
                    "error": f"No token found ({env_key}). Showing built-in model list.",
                }
            )

        models, model_error = _fetch_provider_models(provider_id=provider_name, api_key=api_key, base_url=base_url)
        if not models:
            fallback = _provider_default_options(provider_name)
            return jsonify({"options": fallback, "error": model_error or "Failed to fetch models."})

        if provider_name == "openrouter":
            available = {m for m in models if m}
            curated = [m for m in OPENROUTER_RECOMMENDED_MODELS if m in available]
            if curated:
                models = curated

        sorted_models = [m for m in models if m] if provider_name == "openrouter" else _sort_models_by_ratio([m for m in models if m])
        options = [{"value": model, "label": _model_ratio_label(model)} for model in sorted_models]
        if not options:
            fallback = _provider_default_options(provider_name)
            return jsonify({"options": fallback, "error": "No models resolved from live list. Showing fallback."})
        return jsonify({"options": options, "error": ""})

    @app.post("/api/upload-excel-sheets")
    def upload_excel_sheets() -> Any:
        uploaded_file = request.files.get("file")
        if uploaded_file is None or not uploaded_file.filename:
            return jsonify({"sheets": [], "path": "", "error": "No file uploaded."}), 400
        filename = secure_filename(uploaded_file.filename)
        if not filename.lower().endswith((".xlsx", ".xls")):
            return jsonify({"sheets": [], "path": "", "error": "Supported: .xlsx, .xls"}), 400

        upload_dir = Path(app.config["UPLOAD_FOLDER"])
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        stored_name = f"{stem}_{uuid4().hex[:8]}{suffix}"
        destination = upload_dir / stored_name
        uploaded_file.save(destination)

        sheets = list_excel_sheets(destination)
        return jsonify({"sheets": sheets, "path": str(destination)})

    @app.post("/api/upload-source-metadata")
    def upload_source_metadata() -> Any:
        uploaded_file = request.files.get("file")
        if uploaded_file is None or not uploaded_file.filename:
            return jsonify({"path": "", "error": "No file uploaded."}), 400
        filename = secure_filename(uploaded_file.filename)
        if not filename.lower().endswith((".csv", ".xlsx", ".xls", ".pcap", ".pcapng")):
            return jsonify({"path": "", "error": "Supported: .csv, .xlsx, .xls, .pcap, .pcapng"}), 400

        upload_dir = Path(app.config["UPLOAD_FOLDER"])
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        stored_name = f"{stem}_{uuid4().hex[:8]}{suffix}"
        destination = upload_dir / stored_name
        uploaded_file.save(destination)

        metadata = describe_tabular_source(destination)
        return jsonify({"path": str(destination), **metadata})

    @app.get("/about")
    def about() -> str:
        return render_template(
            "about.html",
            protected_report_types=sorted(PROTECTED_REPORT_TYPES),
            priced_models=_openai_frontier_models(),
            provider_options=_provider_options(),
            doc_pages={doc_id: {"title": title} for doc_id, (_, title) in DOC_PAGES.items()},
        )

    @app.get("/prices")
    def prices() -> str:
        provider_rows: list[dict[str, Any]] = []
        for provider_id, spec in PROVIDER_CATALOG.items():
            if provider_id == "local":
                continue
            env_key = str(spec.get("env_key", "")).strip()
            token = os.getenv(env_key, "").strip() if env_key else ""
            base_url = str(spec.get("default_base_url", "")).strip() or {
                "openai": "https://api.openai.com/v1",
                "claude": "https://api.anthropic.com/v1",
                "xai": "https://api.x.ai/v1",
                "openrouter": "https://openrouter.ai/api/v1",
            }.get(provider_id, "")
            models, model_error = _fetch_provider_models(provider_id=provider_id, api_key=token, base_url=base_url)
            source_kind = "live"
            if not models:
                fallback = [str(m) for m in (spec.get("model_options", []) or []) if str(m).strip()]
                if fallback:
                    models = fallback
                    source_kind = "fallback"
            pricing_url = PROVIDER_PRICING_INFO.get(provider_id, {}).get("pricing_url", "")
            if provider_id == "openrouter":
                available = {m for m in models if m}
                curated = [m for m in OPENROUTER_RECOMMENDED_MODELS if m in available]
                sorted_models = curated if curated else [m for m in models if m]
            else:
                sorted_models = _sort_models_by_ratio([m for m in models if m])
            model_prices = []
            for model_name in sorted_models[:30]:
                reference = _model_reference_cost_and_ratio(model_name)
                model_prices.append(
                    {
                        "model": model_name,
                        "cost_text": reference["cost_text"],
                        "ratio_text": reference["ratio_text"],
                        "source_url": reference["source_url"],
                    }
                )
            provider_rows.append(
                {
                    "provider_id": provider_id,
                    "label": str(spec.get("label", provider_id)),
                    "env_key": env_key,
                    "has_token": bool(token),
                    "model_count": len(sorted_models),
                    "models_preview": ", ".join(sorted_models[:20]) if sorted_models else "-",
                    "models_error": model_error,
                    "model_source": source_kind,
                    "pricing_url": pricing_url,
                    "reference_cost": _reference_cost_text_for_provider(provider_id),
                    "model_prices": model_prices,
                }
            )
        return render_template("prices.html", providers=provider_rows)

    @app.get("/docs/<doc_id>")
    def view_doc(doc_id: str) -> Any:
        if doc_id not in DOC_PAGES:
            flash(f"Unknown documentation page: {doc_id}", "danger")
            return redirect(url_for("about"))
        rel_path, title = DOC_PAGES[doc_id]
        path = _absolute_path(rel_path)
        if not path.exists():
            flash(f"Documentation file not found: {rel_path}", "danger")
            return redirect(url_for("about"))

        text = path.read_text(encoding="utf-8")
        rendered = markdown.markdown(text, extensions=["fenced_code", "tables"])
        return render_template(
            "doc_view.html",
            doc_title=title,
            doc_source=rel_path,
            doc_html=Markup(rendered),
        )

    @app.get("/report-types/new")
    def new_report_type() -> str:
        registry = ReportTypeRegistry(config_dir=Path(app.config["REPORT_TYPES_DIR"]))
        return render_template(
            "new_report_type.html",
            starter_yaml=_starter_report_type_yaml(),
            supported_metrics_profiles=sorted(_supported_metrics_profiles()),
            report_types=registry.list_report_types(),
        )

    @app.get("/api/report-type-yaml")
    def report_type_yaml() -> Any:
        report_type_id = request.args.get("report_type_id", "").strip()
        if not report_type_id:
            return jsonify({"error": "Missing report_type_id"}), 400
        path = Path(app.config["REPORT_TYPES_DIR"]) / f"{report_type_id}.yaml"
        if not path.exists():
            return jsonify({"error": f"Unknown report_type_id: {report_type_id}"}), 404
        return jsonify({"report_type_id": report_type_id, "yaml": path.read_text(encoding="utf-8")})

    @app.get("/report-types/view")
    def view_report_type_yaml_page() -> Any:
        report_type_id = request.args.get("report_type_id", "").strip()
        if not report_type_id:
            flash("Missing report_type_id.", "danger")
            return redirect(url_for("list_report_types_page"))
        if not re.fullmatch(r"[a-z0-9_]+", report_type_id):
            flash("Invalid report_type_id.", "danger")
            return redirect(url_for("list_report_types_page"))
        path = Path(app.config["REPORT_TYPES_DIR"]) / f"{report_type_id}.yaml"
        if not path.exists():
            flash(f"Report type not found: {report_type_id}", "danger")
            return redirect(url_for("list_report_types_page"))
        yaml_text = path.read_text(encoding="utf-8")
        return render_template(
            "report_type_yaml_view.html",
            report_type_id=report_type_id,
            report_type_label=_friendly_report_type_label(report_type_id),
            yaml_text=yaml_text,
            yaml_path=str(path),
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
            registry = ReportTypeRegistry(config_dir=Path(app.config["REPORT_TYPES_DIR"]))
            return render_template(
                "new_report_type.html",
                starter_yaml=yaml_text or _starter_report_type_yaml(),
                supported_metrics_profiles=sorted(_supported_metrics_profiles()),
                report_types=registry.list_report_types(),
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
        attempt_id = request.form.get("attempt_id", "").strip() or f"evt_{uuid4().hex[:10]}"
        report_type_id = request.form.get("report_type_id", "").strip()
        visible_report_types = _visible_report_types_for_generation(
            ReportTypeRegistry(config_dir=Path(app.config["REPORT_TYPES_DIR"])).list_report_types()
        )
        if report_type_id not in visible_report_types:
            flash(f"Report type '{report_type_id}' is not enabled for generation.", "danger")
            return redirect(url_for("index"))
        provider_name = request.form.get("provider", "local").strip().lower()
        if provider_name == "mock":
            provider_name = "local"
        if provider_name not in PROVIDER_CATALOG:
            flash(f"Unknown provider '{provider_name}'.", "danger")
            return redirect(url_for("index"))
        model = request.form.get("model", "gpt-5-mini").strip()
        api_key = request.form.get("api_key", "").strip()
        api_base_url = request.form.get("api_base_url", "").strip()
        confirm_cost = request.form.get("confirm_cost", "0") == "1" or request.form.get("confirm_openai", "0") == "1"
        expected_cost_token = request.form.get("expected_cost_token", "").strip()
        sheet_name = request.form.get("sheet_name", "").strip()
        uploaded_file = request.files.get("csv_upload")
        existing_csv_path = request.form.get("existing_csv_path", "").strip()
        output_token_budget = int(request.form.get("output_token_budget", "1200").strip() or "1200")
        row_limit_raw = request.form.get("row_limit", "").strip()
        row_limit = int(row_limit_raw) if row_limit_raw else None
        generation_cost_usd_est: float | None = 0.0
        generation_input_tokens_est: int | None = None
        generation_output_tokens_est: int | None = None
        report_json_path: Path | None = None
        report_html_path: Path | None = None
        generation_started = False
        generation_succeeded = False
        session["last_provider"] = provider_name

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
                _append_report_event(
                    Path(app.config["OUTPUT_FOLDER"]),
                    {
                        "attempt_id": attempt_id,
                        "report_type_id": report_type_id,
                        "provider": provider_name,
                        "model": model,
                        "status": "failed",
                        "message": "Threshold value must be numeric.",
                    },
                )
                flash("Threshold value must be numeric.", "danger")
                return redirect(url_for("index"))

        try:
            if not confirm_cost:
                _append_report_event(
                    Path(app.config["OUTPUT_FOLDER"]),
                    {
                        "attempt_id": attempt_id,
                        "report_type_id": report_type_id,
                        "provider": provider_name,
                        "model": model,
                        "status": "requested",
                        "message": "Report generation request submitted.",
                    },
                )
            csv_path = _resolve_csv_path(
                uploaded_file,
                Path(app.config["UPLOAD_FOLDER"]),
                existing_csv_path=existing_csv_path,
            )
            if Path(csv_path).suffix.lower() in {".xlsx", ".xls"} and not sheet_name:
                sheets = list_excel_sheets(csv_path)
                if len(sheets) > 1:
                    flash("Excel file has multiple sheets. Choose a sheet and submit again.", "warning")
                    registry = ReportTypeRegistry(config_dir=Path(app.config["REPORT_TYPES_DIR"]))
                    report_types = _visible_report_types_for_generation(registry.list_report_types())
                    return render_template(
                        "index.html",
                        report_types=report_types,
                        report_type_options=_report_type_options(report_types),
                        recent_uploads=_recent_uploaded_sources(Path(app.config["UPLOAD_FOLDER"])),
                        sheet_options=sheets,
                        selected_sheet="",
                        existing_csv_path=csv_path,
                        defaults={
                            "provider": provider_name,
                            "model": model,
                            "row_limit": row_limit if row_limit is not None else 1000,
                            "report_type_id": report_type_id,
                            "api_key": api_key,
                            "api_base_url": api_base_url,
                        },
                        priced_models=_openai_frontier_models(),
                        model_options=_provider_default_options("openai"),
                        provider_options=_provider_options(),
                        provider_model_options=_provider_model_options(),
                        provider_ids=list(PROVIDER_CATALOG.keys()),
                    )
                if len(sheets) == 1:
                    sheet_name = sheets[0]

            definition, effective_prefs, csv_profile, metrics = prepare_pipeline_inputs(
                csv_path=csv_path,
                report_type_id=report_type_id,
                user_prefs=prefs,
                row_limit=row_limit,
                sheet_name=sheet_name or None,
            )

            prompt_text = ""
            if provider_name != "local":
                prompt_text = build_model_prompt_for_estimation(
                    definition=definition,
                    csv_profile=csv_profile,
                    metrics=metrics,
                    user_prefs=effective_prefs,
                )
                if not confirm_cost:
                    estimate = _estimate_provider_cost(
                        provider_name=provider_name,
                        model=model,
                        prompt_text=prompt_text,
                        estimated_output_tokens=output_token_budget,
                        output_root=Path(app.config["OUTPUT_FOLDER"]),
                        report_type_id=report_type_id,
                    )
                    _append_report_event(
                        Path(app.config["OUTPUT_FOLDER"]),
                        {
                            "attempt_id": attempt_id,
                            "report_type_id": report_type_id,
                            "provider": provider_name,
                            "model": model,
                            "status": "cost_estimated",
                            "message": f"Estimated cost ${float(estimate.get('total_cost_usd_est', 0.0)):.6f}",
                        },
                    )
                    cost_token = (
                        f"{provider_name}|{estimate['model']}|{estimate['input_tokens_est']}|"
                        f"{estimate['output_tokens_est']}|{estimate['total_cost_usd_est']}"
                    )
                    return render_template(
                        "confirm_cost.html",
                        estimate=estimate,
                        report_type_id=report_type_id,
                        report_type_label=_friendly_report_type_label(report_type_id),
                        csv_source_label=Path(csv_path).name + (f" (sheet: {sheet_name})" if sheet_name else ""),
                        rows_used=csv_profile.get("row_count", 0),
                        provider=provider_name,
                        model=model,
                        csv_path=csv_path,
                        sheet_name=sheet_name,
                        row_limit=row_limit_raw,
                        output_token_budget=output_token_budget,
                        prefs=effective_prefs,
                        expected_cost_token=cost_token,
                        attempt_id=attempt_id,
                        api_key=api_key,
                        api_base_url=api_base_url,
                    )

            runtime = str(PROVIDER_CATALOG.get(provider_name, {}).get("runtime", "local"))
            if provider_name != "local":
                resolved_api_key, resolved_base_url, default_headers = _resolve_provider_runtime_options(
                    provider_name=provider_name,
                    api_key=api_key,
                    api_base_url=api_base_url,
                )
                fresh_estimate = _estimate_provider_cost(
                    provider_name=provider_name,
                    model=model,
                    prompt_text=prompt_text,
                    estimated_output_tokens=output_token_budget,
                    output_root=Path(app.config["OUTPUT_FOLDER"]),
                    report_type_id=report_type_id,
                )
                fresh_token = (
                    f"{provider_name}|{fresh_estimate['model']}|{fresh_estimate['input_tokens_est']}|"
                    f"{fresh_estimate['output_tokens_est']}|{fresh_estimate['total_cost_usd_est']}"
                )
                if expected_cost_token and expected_cost_token != fresh_token:
                    _append_report_event(
                        Path(app.config["OUTPUT_FOLDER"]),
                        {
                            "attempt_id": attempt_id,
                            "report_type_id": report_type_id,
                            "provider": provider_name,
                            "model": model,
                            "status": "failed",
                            "message": "Cost estimate changed before execution.",
                        },
                    )
                    flash("Cost estimate changed after input update. Please review and confirm again.", "danger")
                    return redirect(url_for("index"))
                generation_cost_usd_est = float(fresh_estimate.get("total_cost_usd_est", 0.0))
                generation_input_tokens_est = int(fresh_estimate.get("input_tokens_est", 0) or 0)
                generation_output_tokens_est = int(fresh_estimate.get("output_tokens_est", 0) or 0)
            else:
                resolved_api_key, resolved_base_url, default_headers = None, None, None
                generation_cost_usd_est = 0.0

            if runtime == "local":
                provider = MockProvider()
            elif runtime == "responses":
                provider = OpenAIResponsesProvider(
                    model=model,
                    api_key=resolved_api_key,
                    base_url=resolved_base_url,
                    default_headers=default_headers,
                )
            elif runtime == "chat_compat":
                json_repair_model = "moonshotai/kimi-k2.5" if provider_name == "openrouter" else None
                provider = OpenAIChatCompletionsProvider(
                    model=model,
                    api_key=resolved_api_key,
                    base_url=resolved_base_url,
                    default_headers=default_headers,
                    json_repair_model=json_repair_model,
                )
            elif runtime == "anthropic":
                provider = AnthropicMessagesProvider(
                    model=model,
                    api_key=resolved_api_key or "",
                    base_url=resolved_base_url,
                )
            else:
                flash(f"Unsupported provider runtime '{runtime}' for provider '{provider_name}'.", "danger")
                return redirect(url_for("index"))

            output_dir = Path(app.config["OUTPUT_FOLDER"]) / report_type_id
            _append_report_event(
                Path(app.config["OUTPUT_FOLDER"]),
                {
                    "attempt_id": attempt_id,
                    "report_type_id": report_type_id,
                    "provider": provider_name,
                    "model": model,
                    "status": "started",
                    "message": "Report generation started.",
                },
            )
            generation_started = True
            generation_context = {
                "backend": provider_name,
                "model": model if provider_name != "local" else "local-metrics",
                "source_csv": Path(csv_path).name,
                "source_sheet": sheet_name,
                "source_rows_used": csv_profile.get("row_count"),
                "generation_cost_usd_est": generation_cost_usd_est,
                "generation_input_tokens_est": generation_input_tokens_est,
                "generation_output_tokens_est": generation_output_tokens_est,
            }
            report_json_path, report_html_path = run_pipeline(
                csv_path=csv_path,
                report_type_id=report_type_id,
                user_prefs=effective_prefs,
                output_dir=output_dir,
                provider=provider,
                row_limit=row_limit,
                sheet_name=sheet_name or None,
                generation_context=generation_context,
            )
            generation_succeeded = True
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            status = "timeout" if "timeout" in msg.lower() else "failed"
            _append_report_event(
                Path(app.config["OUTPUT_FOLDER"]),
                {
                    "attempt_id": attempt_id,
                    "report_type_id": report_type_id,
                    "provider": provider_name,
                    "model": model,
                    "status": status,
                    "message": msg,
                },
            )
            flash(str(exc), "danger")
            return redirect(url_for("index"))
        finally:
            if generation_started and generation_succeeded and report_json_path is not None:
                report_name = Path(report_json_path).stem.replace(".report", "")
                _append_report_event(
                    Path(app.config["OUTPUT_FOLDER"]),
                    {
                        "attempt_id": attempt_id,
                        "report_type_id": report_type_id,
                        "provider": provider_name,
                        "model": model,
                        "status": "finished",
                        "report_name": report_name,
                        "message": "Report generation completed.",
                    },
                )

        flash("Report generated successfully.", "success")
        raw_path = report_json_path.with_name(report_json_path.name.replace(".report.json", ".openai.raw.json"))
        pdf_path = report_json_path.with_name(report_json_path.name.replace(".report.json", ".report.pdf"))
        return render_template(
            "result.html",
            report_type_id=report_type_id,
            report_type_label=_friendly_report_type_label(report_type_id),
            report_json_path=str(report_json_path),
            report_html_path=str(report_html_path),
            report_pdf_path=str(pdf_path) if pdf_path.exists() else "",
            report_raw_path=str(raw_path) if raw_path.exists() else "",
            report_json=_load_json(report_json_path),
        )

    @app.get("/reports")
    def reports() -> str:
        root = Path(app.config["OUTPUT_FOLDER"])
        items = _collect_report_history(root)
        return render_template("reports.html", reports=items, output_folder=str(root))

    @app.get("/benchmark")
    def benchmark() -> str:
        root = Path(app.config["OUTPUT_FOLDER"])
        items = _collect_report_history(root)
        report_type_ids = sorted({str(i.get("report_type_id", "")).strip() for i in items if str(i.get("report_type_id", "")).strip()})
        report_type_filter = request.args.get("report_type_id", "").strip()
        selected_paths = [p.strip() for p in request.args.getlist("json_path") if p.strip()]
        if len(selected_paths) > 4:
            flash("Benchmark view supports up to 4 reports at once. Showing first 4.", "warning")
            selected_paths = selected_paths[:4]

        filtered_items = items
        if report_type_filter:
            filtered_items = [i for i in items if str(i.get("report_type_id", "")) == report_type_filter]

        by_path = {str(i.get("json_path", "")): i for i in filtered_items}
        selected_reports: list[dict[str, Any]] = []
        for json_path in selected_paths:
            item = by_path.get(json_path)
            if not item:
                continue
            payload: dict[str, Any] = {}
            try:
                path = _validate_report_artifact_path(json_path, root)
                payload = _load_json(path)
            except Exception:  # noqa: BLE001
                payload = {}
            metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
            sections = payload.get("sections", []) if isinstance(payload.get("sections"), list) else []
            alerts = payload.get("alerts", []) if isinstance(payload.get("alerts"), list) else []
            recommendations = payload.get("recommendations", []) if isinstance(payload.get("recommendations"), list) else []

            selected_reports.append(
                {
                    **item,
                    "summary_text": str(payload.get("summary", "") or ""),
                    "sections": sections,
                    "alerts": alerts,
                    "recommendations": recommendations,
                    "metadata": metadata,
                }
            )

        analytics: dict[str, Any] = {}
        comparison_rows: list[dict[str, Any]] = []
        similarity_rows: list[dict[str, Any]] = []
        comparison_brief: dict[str, Any] = {}
        baseline_assessment: dict[str, Any] = {}
        report_identity_by_name: dict[str, dict[str, str]] = {}
        if selected_reports:
            for report in selected_reports:
                cost = _safe_float(report.get("generation_cost_usd_est_raw"))
                duration = _safe_float(report.get("generation_duration_seconds_raw"))
                in_tok = _safe_float(report.get("generation_input_tokens_est_raw"))
                out_tok = _safe_float(report.get("generation_output_tokens_est_raw"))
                total_tok = in_tok + out_tok
                sections_count = len(report.get("sections", []) or [])
                alerts_count = len(report.get("alerts", []) or [])
                recs_count = len(report.get("recommendations", []) or [])
                summary_words = len(str(report.get("summary_text", "") or "").split())
                cost_per_1k_tokens = (cost / (total_tok / 1000.0)) if total_tok > 0 else 0.0
                duration_per_1k_tokens = (duration / (total_tok / 1000.0)) if total_tok > 0 else 0.0
                insight_count = sections_count + alerts_count + recs_count
                insights_per_1k_tokens = (insight_count / (total_tok / 1000.0)) if total_tok > 0 else 0.0
                comparison_rows.append(
                    {
                        "name": report.get("name", "-"),
                        "provider": report.get("backend", "-"),
                        "model": report.get("model", "-"),
                        "cost": cost,
                        "duration": duration,
                        "tokens": total_tok,
                        "sections": sections_count,
                        "alerts": alerts_count,
                        "recommendations": recs_count,
                        "summary_words": summary_words,
                        "cost_per_1k_tokens": cost_per_1k_tokens,
                        "duration_per_1k_tokens": duration_per_1k_tokens,
                        "insights_per_1k_tokens": insights_per_1k_tokens,
                    }
                )
            report_identity_by_name = {
                str(row["name"]): {"provider": str(row["provider"]), "model": str(row["model"])}
                for row in comparison_rows
            }

            costs = [r["cost"] for r in comparison_rows]
            durations = [r["duration"] for r in comparison_rows]
            tokens = [r["tokens"] for r in comparison_rows]

            cheapest = min(comparison_rows, key=lambda r: r["cost"])
            fastest = min(comparison_rows, key=lambda r: r["duration"])
            lowest_tokens = min(comparison_rows, key=lambda r: r["tokens"])
            best_cost_eff = min(comparison_rows, key=lambda r: r["cost_per_1k_tokens"])
            best_speed_eff = min(comparison_rows, key=lambda r: r["duration_per_1k_tokens"])
            best_signal_density = max(comparison_rows, key=lambda r: r["insights_per_1k_tokens"])
            analytics = {
                "selected_count": len(comparison_rows),
                "avg_cost": sum(costs) / len(costs) if costs else 0.0,
                "avg_duration": sum(durations) / len(durations) if durations else 0.0,
                "avg_tokens": sum(tokens) / len(tokens) if tokens else 0.0,
                "cheapest": cheapest,
                "fastest": fastest,
                "lowest_tokens": lowest_tokens,
                "best_cost_eff": best_cost_eff,
                "best_speed_eff": best_speed_eff,
                "best_signal_density": best_signal_density,
            }

            summaries = {
                row["name"]: _token_set_for_similarity(
                    str(next((r.get("summary_text", "") for r in selected_reports if r.get("name") == row["name"]), "") or "")
                )
                for row in comparison_rows
            }
            for left, right in combinations(comparison_rows, 2):
                left_name = str(left["name"])
                right_name = str(right["name"])
                sim = _jaccard_similarity(summaries.get(left_name, set()), summaries.get(right_name, set()))
                similarity_rows.append(
                    {
                        "left": left_name,
                        "right": right_name,
                        "similarity_pct": round(sim * 100, 1),
                    }
                )
            similarity_rows.sort(key=lambda r: r["similarity_pct"], reverse=True)

            label_names = ["A", "B", "C", "D"]
            labeled = []
            for idx, report in enumerate(selected_reports):
                label = label_names[idx] if idx < len(label_names) else f"R{idx+1}"
                text_blob = _benchmark_text_blob(report)
                labeled.append(
                    {
                        "label": label,
                        "name": str(report.get("name", "-")),
                        "summary_text": str(report.get("summary_text", "") or ""),
                        "text_blob": text_blob,
                        "sections": len(report.get("sections", []) or []),
                        "alerts": len(report.get("alerts", []) or []),
                        "recommendations": len(report.get("recommendations", []) or []),
                        "summary_words": len(str(report.get("summary_text", "") or "").split()),
                    }
                )

            if labeled:
                def _v(report: dict[str, Any], category: str) -> str:
                    if category == "overall_depth":
                        return _benchmark_overall_depth(report)
                    if category == "technical_rigor":
                        return _benchmark_technical_rigor(report)
                    if category == "exec_friendliness":
                        return _benchmark_exec_friendliness(report)
                    if category == "length_detail":
                        return _benchmark_length_detail(report)
                    if category == "rec_scope":
                        return _benchmark_recommendation_scope(report["text_blob"])
                    if category == "data_eng":
                        return _benchmark_data_engineering_guidance(report["text_blob"])
                    if category == "modeling":
                        return _benchmark_modeling_guidance(report["text_blob"])
                    if category == "longitudinal":
                        return _benchmark_longitudinal_readiness(report["text_blob"])
                    return "-"

                matrix_categories = [
                    ("Overall Depth", "overall_depth"),
                    ("Technical Rigor", "technical_rigor"),
                    ("Executive Friendliness", "exec_friendliness"),
                    ("Length / Detail", "length_detail"),
                    ("Recommendations – Scope", "rec_scope"),
                    ("Data Engineering Guidance", "data_eng"),
                    ("Modeling Guidance", "modeling"),
                    ("Longitudinal Readiness", "longitudinal"),
                ]

                matrix_rows: list[dict[str, Any]] = []
                for title, key in matrix_categories:
                    row = {"category": title, "values": []}
                    for r in labeled:
                        row["values"].append(_v(r, key))
                    matrix_rows.append(row)

                key_differences: list[str] = []
                if len(labeled) >= 2:
                    a = labeled[0]
                    b = labeled[1]
                    if a["summary_words"] < b["summary_words"]:
                        key_differences.append(f"{a['label']} is shorter and more executive-friendly; {b['label']} is more detailed.")
                    elif b["summary_words"] < a["summary_words"]:
                        key_differences.append(f"{b['label']} is shorter and more executive-friendly; {a['label']} is more detailed.")
                    if _benchmark_technical_rigor(a) != _benchmark_technical_rigor(b):
                        key_differences.append(
                            f"Technical rigor differs: {a['label']}={_benchmark_technical_rigor(a)}, {b['label']}={_benchmark_technical_rigor(b)}."
                        )
                if len(labeled) >= 3:
                    c = labeled[2]
                    b = labeled[1]
                    sim_bc = _jaccard_similarity(
                        _token_set_for_similarity(c["summary_text"]),
                        _token_set_for_similarity(b["summary_text"]),
                    )
                    if sim_bc >= 0.92:
                        key_differences.append(f"{c['label']} is effectively a near-duplicate of {b['label']} (summary similarity {sim_bc*100:.1f}%).")

                choose_rows = [
                    {
                        "use_case": "Executive update",
                        "best_label": labeled[0]["label"] if labeled else "-",
                        "best_name": labeled[0]["name"] if labeled else "-",
                    },
                    {
                        "use_case": "Data science / modeling prep",
                        "best_label": max(
                            labeled, key=lambda r: (r["sections"] + r["recommendations"], r["summary_words"])
                        )["label"],
                        "best_name": max(
                            labeled, key=lambda r: (r["sections"] + r["recommendations"], r["summary_words"])
                        )["name"],
                    },
                    {
                        "use_case": "Production data pipeline hardening",
                        "best_label": max(labeled, key=lambda r: _benchmark_ops_signal_score(r["text_blob"]))["label"],
                        "best_name": max(labeled, key=lambda r: _benchmark_ops_signal_score(r["text_blob"]))["name"],
                    },
                    {
                        "use_case": "Clinical / audit defensibility",
                        "best_label": max(labeled, key=lambda r: _benchmark_stats_signal_score(r["text_blob"]))["label"],
                        "best_name": max(labeled, key=lambda r: _benchmark_stats_signal_score(r["text_blob"]))["name"],
                    },
                ]

                comparison_brief = {
                    "title": "Side-by-Side Comparison Brief",
                    "labels": [r["label"] for r in labeled],
                    "label_to_name": {r["label"]: r["name"] for r in labeled},
                    "label_to_provider": {r["label"]: str(report_identity_by_name.get(r["name"], {}).get("provider", "-")) for r in labeled},
                    "label_to_model": {r["label"]: str(report_identity_by_name.get(r["name"], {}).get("model", "-")) for r in labeled},
                    "matrix_rows": matrix_rows,
                    "key_differences": key_differences,
                    "choose_rows": choose_rows,
                }

            selected_count = len(selected_reports)
            if selected_count in {2, 3}:
                source_files = [str(r.get("source_csv", "") or "").strip() for r in selected_reports]
                report_types = [
                    str(r.get("report_type_label", "") or _friendly_report_type_label(str(r.get("report_type_id", "") or ""))).strip()
                    for r in selected_reports
                ]
                row_limits = [_safe_float(r.get("source_rows_used_raw")) for r in selected_reports]
                model_ratios = [_model_cost_ratio_vs_baseline(str(r.get("model", "") or "")) for r in selected_reports]

                same_source_file = len({v for v in source_files if v}) == 1 and bool(source_files and source_files[0])
                same_report_type = len({v for v in report_types if v}) == 1 and bool(report_types and report_types[0])
                same_input_rows = len(set(int(v) for v in row_limits if v > 0)) == 1 and any(v > 0 for v in row_limits)

                known_ratios = [r for r in model_ratios if isinstance(r, (int, float))]
                same_model_strength = False
                almost_model_strength = False
                weak_model_strength = False
                if len(known_ratios) == selected_count:
                    ratio_span = max(known_ratios) - min(known_ratios)
                    same_model_strength = ratio_span <= 0.20
                    almost_model_strength = (not same_model_strength) and ratio_span < 4.0
                    weak_model_strength = (not same_model_strength) and (not almost_model_strength)

                positive_rows = [int(v) for v in row_limits if v > 0]
                almost_input_rows = False
                if (not same_input_rows) and len(positive_rows) == selected_count:
                    max_rows = max(positive_rows)
                    min_rows = min(positive_rows)
                    almost_input_rows = max_rows > 0 and ((max_rows - min_rows) / max_rows) <= 0.05

                c1_status, c1_score = _baseline_check_status(same_source_file, False)
                c2_status, c2_score = _baseline_check_status(same_report_type, False)
                c3_status, c3_score = _baseline_check_status(same_model_strength, almost_model_strength, weak_model_strength)
                c4_status, c4_score = _baseline_check_status(same_input_rows, almost_input_rows)

                checks = [
                    {"key": "same_source_file", "label": "Same source data file", "status": c1_status, "score": c1_score},
                    {"key": "same_report_type", "label": "Same report type", "status": c2_status, "score": c2_score},
                    {"key": "same_model_strength", "label": "Same model strength (1.x ratio)", "status": c3_status, "score": c3_score},
                    {"key": "same_input_rows", "label": "Same input rows used", "status": c4_status, "score": c4_score},
                ]
                total_score = c1_score + c2_score + c3_score + c4_score
                if total_score >= 11:
                    baseline_level = "strong"
                elif total_score >= 7:
                    baseline_level = "moderate"
                else:
                    baseline_level = "weak"

                baseline_assessment = {
                    "supported": True,
                    "selected_count": selected_count,
                    "checks": checks,
                    "baseline_level": baseline_level,
                    "score": total_score,
                    "score_max": 12,
                    "model_ratios": [round(float(r), 2) if isinstance(r, (int, float)) else None for r in model_ratios],
                    "sources": source_files,
                    "sources_display": _compact_values_display(source_files),
                    "report_types": report_types,
                    "report_types_display": _compact_values_display(report_types),
                    "rows": [int(v) if v > 0 else None for v in row_limits],
                    "rows_display": _compact_values_display([int(v) if v > 0 else None for v in row_limits], none_text="-"),
                }
            else:
                baseline_assessment = {
                    "supported": False,
                    "selected_count": selected_count,
                    "message": "Baseline criteria is evaluated only when exactly 2 or 3 reports are selected.",
                }

        return render_template(
            "benchmark.html",
            output_folder=str(root),
            reports=filtered_items,
            selected_reports=selected_reports,
            report_type_ids=report_type_ids,
            report_type_filter=report_type_filter,
            selected_paths=selected_paths,
            analytics=analytics,
            comparison_rows=comparison_rows,
            similarity_rows=similarity_rows,
            comparison_brief=comparison_brief,
            baseline_assessment=baseline_assessment,
            report_identity_by_name=report_identity_by_name,
        )

    @app.post("/reports/actual-cost")
    def update_report_actual_cost() -> Any:
        json_path = request.form.get("json_path", "").strip()
        actual_cost_raw = request.form.get("actual_cost_usd", "").strip()
        if not json_path:
            flash("Missing report path.", "danger")
            return redirect(url_for("reports"))
        try:
            path = _validate_report_artifact_path(json_path, Path(app.config["OUTPUT_FOLDER"]))
        except ValueError:
            flash("Invalid report path.", "danger")
            return redirect(url_for("reports"))
        if not path.exists():
            flash("Report file not found.", "danger")
            return redirect(url_for("reports"))

        try:
            payload = _load_json(path)
        except Exception:  # noqa: BLE001
            flash("Failed to read report JSON.", "danger")
            return redirect(url_for("reports"))

        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        if actual_cost_raw == "":
            metadata.pop("actual_cost_usd_user", None)
            flash("Actual cost cleared.", "success")
        else:
            try:
                actual_cost_value = float(actual_cost_raw)
            except ValueError:
                flash("Actual cost must be numeric.", "danger")
                return redirect(url_for("reports"))
            if actual_cost_value < 0:
                flash("Actual cost cannot be negative.", "danger")
                return redirect(url_for("reports"))
            metadata["actual_cost_usd_user"] = round(actual_cost_value, 6)
            flash("Actual cost saved.", "success")

        payload["metadata"] = metadata
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return redirect(url_for("reports"))

    @app.get("/events")
    def events() -> str:
        root = Path(app.config["OUTPUT_FOLDER"])
        rows = _load_report_events(root)
        return render_template(
            "events.html",
            events=rows,
            output_folder=str(root),
            events_signature=_event_rows_signature(rows),
        )

    @app.get("/api/events-status")
    def events_status() -> Any:
        root = Path(app.config["OUTPUT_FOLDER"])
        rows = _load_report_events(root)
        return jsonify(
            {
                "count": len(rows),
                "signature": _event_rows_signature(rows),
            }
        )

    @app.post("/events/delete")
    def delete_event() -> Any:
        idx_raw = request.form.get("line_index", "").strip()
        root = Path(app.config["OUTPUT_FOLDER"])
        path = _events_log_path(root)
        if not idx_raw:
            flash("Missing event index.", "danger")
            return redirect(url_for("events"))
        try:
            line_index = int(idx_raw)
        except ValueError:
            flash("Invalid event index.", "danger")
            return redirect(url_for("events"))
        if not path.exists():
            flash("Events log not found.", "danger")
            return redirect(url_for("events"))

        lines = path.read_text(encoding="utf-8").splitlines()
        if line_index < 0 or line_index >= len(lines):
            flash("Event not found.", "danger")
            return redirect(url_for("events"))
        del lines[line_index]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        flash("Event deleted.", "success")
        return redirect(url_for("events"))

    @app.post("/events/reset-in-progress")
    def reset_in_progress_events() -> Any:
        root = Path(app.config["OUTPUT_FOLDER"])
        snapshot = _generation_status_snapshot(root)
        active_attempt_ids = [str(a).strip() for a in snapshot.get("active_attempt_ids", []) if str(a).strip()]
        if not active_attempt_ids:
            flash("No in-progress generations to reset.", "warning")
            return redirect(url_for("events"))

        path = _events_log_path(root)
        latest_for_attempt: dict[str, dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except Exception:  # noqa: BLE001
                    continue
                if not isinstance(payload, dict):
                    continue
                attempt_id = str(payload.get("attempt_id", "")).strip()
                if attempt_id in active_attempt_ids:
                    latest_for_attempt[attempt_id] = payload

        reset_count = 0
        for attempt_id in active_attempt_ids:
            prev = latest_for_attempt.get(attempt_id, {})
            _append_report_event(
                root,
                {
                    "attempt_id": attempt_id,
                    "report_type_id": str(prev.get("report_type_id", "")),
                    "provider": str(prev.get("provider", "")),
                    "model": str(prev.get("model", "")),
                    "status": "cancelled",
                    "message": "Generation reset by user.",
                },
            )
            reset_count += 1

        flash(f"Reset {reset_count} in-progress generation(s).", "success")
        return redirect(url_for("events"))

    @app.get("/performance")
    def performance() -> str:
        root = Path(app.config["OUTPUT_FOLDER"])
        items = _collect_report_history(root)
        report_type_filter = request.args.get("report_type", "").strip().lower()
        provider_filter = request.args.get("provider", "").strip().lower()
        model_filter = request.args.get("model", "").strip().lower()
        perf_rows = []
        for row in items:
            backend = str(row.get("backend", "local"))
            cost_num = _safe_float(row.get("generation_cost_usd_est_raw"))
            in_tok = _safe_float(row.get("generation_input_tokens_est_raw"))
            out_tok = _safe_float(row.get("generation_output_tokens_est_raw"))
            if backend != "local" or cost_num > 0 or (in_tok + out_tok) > 0:
                perf_rows.append(row)

        provider_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"runs": 0, "cost": 0.0, "duration": 0.0, "tokens": 0.0})
        triplet_stats: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
            lambda: {"runs": 0, "cost": 0.0, "duration": 0.0, "tokens": 0.0}
        )
        model_tokens: dict[str, float] = defaultdict(float)
        model_costs: dict[str, float] = defaultdict(float)
        scatter_rows: list[dict[str, Any]] = []

        for row in perf_rows:
            provider = str(row.get("backend", "unknown"))
            model = str(row.get("model", "-"))
            report_type_id = str(row.get("report_type_id", ""))
            report_type_label = _friendly_report_type_label(report_type_id)
            duration_num = _safe_float(row.get("generation_duration_seconds_raw"))
            cost_num = _safe_float(row.get("generation_cost_usd_est_raw"))
            input_tokens = _safe_float(row.get("generation_input_tokens_est_raw"))
            output_tokens = _safe_float(row.get("generation_output_tokens_est_raw"))
            total_tokens = input_tokens + output_tokens
            src_rows = _safe_float(row.get("source_rows_used_raw"))

            ps = provider_stats[provider]
            ps["runs"] += 1
            ps["cost"] += cost_num
            ps["duration"] += duration_num
            ps["tokens"] += total_tokens

            ts = triplet_stats[(report_type_label, provider, model)]
            ts["runs"] += 1
            ts["cost"] += cost_num
            ts["duration"] += duration_num
            ts["tokens"] += total_tokens

            if model and model != "-":
                model_tokens[model] += total_tokens
                model_costs[model] += cost_num

            if duration_num > 0 and total_tokens > 0:
                scatter_rows.append(
                    {
                        "x": duration_num,
                        "y": total_tokens,
                        "provider": provider,
                        "model": model,
                        "rows": src_rows,
                        "cost": cost_num,
                    }
                )

        provider_rows = []
        for provider, stats in sorted(provider_stats.items(), key=lambda kv: kv[0]):
            runs = int(stats["runs"])
            provider_rows.append(
                {
                    "provider": provider,
                    "runs": runs,
                    "total_cost": round(float(stats["cost"]), 6),
                    "avg_duration": round((float(stats["duration"]) / runs), 2) if runs else 0.0,
                    "total_tokens": int(round(float(stats["tokens"]))),
                }
            )

        model_tokens_rows = [
            {"model": model, "tokens": int(round(tokens))}
            for model, tokens in sorted(model_tokens.items(), key=lambda kv: kv[1], reverse=True)[:12]
        ]
        model_cost_rows = [
            {"model": model, "cost": round(float(cost), 6)}
            for model, cost in sorted(model_costs.items(), key=lambda kv: kv[1], reverse=True)[:12]
        ]
        report_provider_model_rows = []
        for (report_type_label, provider, model), stats in sorted(
            triplet_stats.items(),
            key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]),
        ):
            runs = int(stats["runs"])
            total_cost = round(float(stats["cost"]), 6)
            avg_duration = round((float(stats["duration"]) / runs), 2) if runs else 0.0
            total_tokens = int(round(float(stats["tokens"])))
            report_provider_model_rows.append(
                {
                    "report_type": report_type_label,
                    "provider": provider,
                    "model": model,
                    "runs": runs,
                    "total_cost": total_cost,
                    "avg_duration": avg_duration,
                    "total_tokens": total_tokens,
                }
            )
        filter_report_type_options = sorted({str(r.get("report_type", "")).strip() for r in report_provider_model_rows if str(r.get("report_type", "")).strip()})
        filter_provider_options = sorted({str(r.get("provider", "")).strip() for r in report_provider_model_rows if str(r.get("provider", "")).strip()})
        filter_model_options = sorted({str(r.get("model", "")).strip() for r in report_provider_model_rows if str(r.get("model", "")).strip()})

        filtered_report_provider_model_rows = report_provider_model_rows
        if report_type_filter:
            filtered_report_provider_model_rows = [
                r for r in filtered_report_provider_model_rows if str(r.get("report_type", "")).strip().lower() == report_type_filter
            ]
        if provider_filter:
            filtered_report_provider_model_rows = [
                r for r in filtered_report_provider_model_rows if str(r.get("provider", "")).strip().lower() == provider_filter
            ]
        if model_filter:
            filtered_report_provider_model_rows = [
                r for r in filtered_report_provider_model_rows if str(r.get("model", "")).strip().lower() == model_filter
            ]

        used_triplet_rows = [
            r
            for r in filtered_report_provider_model_rows
            if float(r.get("total_cost", 0.0)) > 0
            or float(r.get("avg_duration", 0.0)) > 0
            or int(r.get("total_tokens", 0)) > 0
        ]
        top_triplet_cost_rows = sorted(
            used_triplet_rows,
            key=lambda r: float(r.get("total_cost", 0.0)),
            reverse=True,
        )[:12]
        top_triplet_duration_rows = sorted(
            used_triplet_rows,
            key=lambda r: float(r.get("avg_duration", 0.0)),
            reverse=True,
        )[:12]

        return render_template(
            "performance.html",
            output_folder=str(root),
            provider_rows=provider_rows,
            model_tokens_rows=model_tokens_rows,
            model_cost_rows=model_cost_rows,
            scatter_rows=scatter_rows,
            report_provider_model_rows=filtered_report_provider_model_rows,
            top_triplet_cost_rows=top_triplet_cost_rows,
            top_triplet_duration_rows=top_triplet_duration_rows,
            report_type_filter=report_type_filter,
            provider_filter=provider_filter,
            model_filter=model_filter,
            filter_report_type_options=filter_report_type_options,
            filter_provider_options=filter_provider_options,
            filter_model_options=filter_model_options,
            total_runs=len(perf_rows),
            total_cost=_format_cost_usd(sum(_safe_float(r.get("total_cost")) for r in provider_rows)),
            total_tokens=f"{int(sum(r['total_tokens'] for r in provider_rows)):,}",
        )

    @app.post("/reports/delete")
    def delete_report() -> Any:
        json_path = request.form.get("json_path", "").strip()
        if not json_path:
            flash("Missing report path.", "danger")
            return redirect(url_for("reports"))
        try:
            path = _validate_report_artifact_path(json_path, Path(app.config["OUTPUT_FOLDER"]))
        except ValueError:
            flash("Invalid report path.", "danger")
            return redirect(url_for("reports"))

        if not path.exists():
            flash("Report file not found.", "danger")
            return redirect(url_for("reports"))

        html_path = path.with_name(path.name.replace(".report.json", ".report.html"))
        pdf_path = path.with_name(path.name.replace(".report.json", ".report.pdf"))
        raw_path = path.with_name(path.name.replace(".report.json", ".openai.raw.json"))

        deleted = 0
        for artifact in (path, html_path, pdf_path, raw_path):
            if artifact.exists():
                artifact.unlink()
                deleted += 1

        flash(f"Deleted report artifacts ({deleted} files).", "success")
        return redirect(url_for("reports"))

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
        download = request.args.get("download", "0") == "1"
        path = _validate_report_artifact_path(html_path, Path(app.config["OUTPUT_FOLDER"]))
        if not path.exists():
            flash(f"File not found: {path}", "danger")
            return redirect(url_for("reports"))
        return send_file(path, as_attachment=download, download_name=path.name)

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
        download = request.args.get("download", "0") == "1"
        path = _validate_report_artifact_path(pdf_path, Path(app.config["OUTPUT_FOLDER"]))
        if not path.exists():
            flash(f"PDF file not found: {path}", "danger")
            return redirect(url_for("reports"))
        return send_file(path, as_attachment=download, download_name=path.name)

    return app


def _recent_uploaded_sources(upload_dir: Path, limit: int = 8) -> list[str]:
    if not upload_dir.exists():
        return []
    files = (
        list(upload_dir.glob("*.csv"))
        + list(upload_dir.glob("*.xlsx"))
        + list(upload_dir.glob("*.xls"))
        + list(upload_dir.glob("*.pcap"))
        + list(upload_dir.glob("*.pcapng"))
    )
    files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return [str(p) for p in files[:limit]]


def _provider_options() -> list[dict[str, str]]:
    return [{"id": pid, "label": str(spec.get("label", pid))} for pid, spec in PROVIDER_CATALOG.items()]


def _report_type_options(report_type_ids: list[str]) -> list[dict[str, str]]:
    return [{"id": rid, "label": _friendly_report_type_label(rid)} for rid in report_type_ids]


def _visible_report_types_for_generation(all_report_types: list[str]) -> list[str]:
    configured = (os.getenv("GENERATION_VISIBLE_REPORT_TYPES", "") or "").strip()
    if not configured:
        return all_report_types

    requested = [v.strip() for v in configured.split(",") if v.strip()]
    if not requested:
        return all_report_types

    requested_set = set(requested)
    filtered = [rid for rid in all_report_types if rid in requested_set]
    return filtered or all_report_types


def _openai_frontier_models() -> list[str]:
    models = PROVIDER_CATALOG.get("openai", {}).get("model_options", []) or []
    return [str(m) for m in models if str(m).strip()]


def _provider_model_options() -> dict[str, list[dict[str, str]]]:
    options: dict[str, list[dict[str, str]]] = {}
    for provider_id, spec in PROVIDER_CATALOG.items():
        models = spec.get("model_options", []) or []
        model_names = _sort_models_by_ratio([str(m) for m in models])
        options[provider_id] = [{"value": model_name, "label": _model_ratio_label(model_name)} for model_name in model_names]
    return options


def _provider_default_options(provider_id: str) -> list[dict[str, str]]:
    spec = PROVIDER_CATALOG.get(provider_id, {})
    models = spec.get("model_options", []) or []
    priced = _sort_models_by_ratio([str(m) for m in models if _is_priced_model(str(m))])
    if priced:
        return [{"value": m, "label": _model_ratio_label(m)} for m in priced]
    fallback_models = _sort_models_by_ratio([str(m) for m in models if str(m).strip()])
    return [{"value": m, "label": _model_ratio_label(m)} for m in fallback_models]


def _provider_default_base_url(provider_name: str) -> str:
    default = str(PROVIDER_CATALOG.get(provider_name, {}).get("default_base_url", "")).strip()
    if default:
        return default
    return {
        "openai": "https://api.openai.com/v1",
        "claude": "https://api.anthropic.com/v1",
        "xai": "https://api.x.ai/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
        "openrouter": "https://openrouter.ai/api/v1",
    }.get(provider_name, "")


def _resolve_provider_runtime_options(
    provider_name: str,
    api_key: str,
    api_base_url: str,
) -> tuple[str | None, str | None, dict[str, str] | None]:
    spec = PROVIDER_CATALOG.get(provider_name)
    if spec is None:
        raise ValueError(f"Unknown provider '{provider_name}'.")

    if provider_name == "local":
        return None, None, None

    env_key_name = str(spec.get("env_key", "")).strip()
    resolved_api_key = api_key.strip() or (os.getenv(env_key_name, "").strip() if env_key_name else "")
    if not resolved_api_key:
        hint = f"Set {env_key_name} in env or provide API Token in UI." if env_key_name else "Provide API Token in UI."
        raise ValueError(f"Missing API token for provider '{provider_name}'. {hint}")

    default_base_url = str(spec.get("default_base_url", "")).strip()
    resolved_base_url = api_base_url.strip() or default_base_url or None
    return resolved_api_key, resolved_base_url, None


def _collect_report_history(output_root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    timestamped_pattern = re.compile(r"\.(\d{6}_\d{4}_\d{6})\.report\.json$")
    for path in sorted(output_root.rglob("*.report.json")):
        match = timestamped_pattern.search(path.name)
        if not match and re.fullmatch(r"[a-z0-9_]+\.report\.json", path.name):
            continue

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

        metadata: dict[str, Any] = {}
        payload: dict[str, Any] = {}
        try:
            payload = _load_json(path)
            meta_candidate = payload.get("metadata", {})
            metadata = meta_candidate if isinstance(meta_candidate, dict) else {}
        except Exception:  # noqa: BLE001
            metadata = {}

        created_at = "unknown"
        created_epoch = path.stat().st_mtime
        if match:
            dt_utc = _run_id_to_utc(match.group(1))
            created_at = _display_local_time(dt_utc)
            created_epoch = dt_utc.timestamp()
        else:
            try:
                generated = str(metadata.get("generated_at_utc", "")).strip()
                if generated:
                    dt_utc = _iso_to_utc(generated)
                    created_at = _display_local_time(dt_utc)
                    created_epoch = dt_utc.timestamp()
            except Exception:  # noqa: BLE001
                created_at = "unknown"

        duration_raw = metadata.get("generation_duration_seconds")
        cost_raw = metadata.get("generation_cost_usd_est")
        cost_actual_raw = metadata.get("generation_cost_usd_actual")
        actual_cost_raw = metadata.get("actual_cost_usd_user")
        input_tokens_raw = metadata.get("generation_input_tokens_est")
        output_tokens_raw = metadata.get("generation_output_tokens_est")
        input_tokens_actual_raw = metadata.get("generation_input_tokens_actual")
        output_tokens_actual_raw = metadata.get("generation_output_tokens_actual")
        rows_raw = metadata.get("source_rows_used")
        backend_raw = str(metadata.get("generation_backend", "")).strip().lower() or "local"
        model_raw = str(metadata.get("generation_model", "")).strip() or ("local-metrics" if backend_raw == "local" else "-")
        model_display = _friendly_model_label(model_raw, backend_raw)
        effective_input_tokens = input_tokens_actual_raw if input_tokens_actual_raw is not None else input_tokens_raw
        effective_output_tokens = output_tokens_actual_raw if output_tokens_actual_raw is not None else output_tokens_raw
        if cost_actual_raw is not None and str(cost_actual_raw).strip() != "":
            cost_raw = cost_actual_raw
        if cost_raw is None or cost_raw == "":
            if backend_raw == "local":
                cost_raw = 0.0
            else:
                inferred_cost = _infer_cost_from_tokens_and_model(
                    input_tokens=effective_input_tokens,
                    output_tokens=effective_output_tokens,
                    model=model_raw,
                )
                if inferred_cost is not None:
                    cost_raw = inferred_cost

        raw_name = path.stem.replace(".report", "")
        report_type_id_raw = str(payload.get("report_type_id", "")).strip() if isinstance(payload, dict) else ""
        display_name = _friendly_report_name(raw_name, report_type_id_raw)
        report_type_label = _friendly_report_type_label(report_type_id_raw)

        items.append(
            {
                "json_path": str(path),
                "html_path": str(html_path),
                "raw_path": str(raw_path) if raw_path.exists() else "",
                "pdf_path": str(pdf_path) if pdf_path.exists() else "",
                "name": display_name,
                "name_raw": raw_name,
                "report_type_id": report_type_id_raw,
                "report_type_label": report_type_label,
                "created_at": created_at,
                "created_epoch": created_epoch,
                "backend": backend_raw,
                "model": model_display,
                "source_csv": str(metadata.get("source_csv", "-")),
                "source_sheet": str(metadata.get("source_sheet", "")),
                "source_rows_used": rows_raw if rows_raw is not None else "-",
                "source_rows_used_raw": rows_raw if rows_raw is not None else 0,
                "generation_duration_seconds": _format_duration_seconds(duration_raw),
                "generation_duration_seconds_raw": duration_raw if duration_raw is not None else 0,
                "generation_cost_usd_est": _format_cost_usd(cost_raw),
                "generation_cost_usd_est_raw": cost_raw if cost_raw is not None else 0.0,
                "generation_cost_usd_actual_raw": cost_actual_raw if cost_actual_raw is not None else None,
                "actual_cost_usd": _format_cost_usd(actual_cost_raw),
                "actual_cost_usd_raw": actual_cost_raw if actual_cost_raw is not None else None,
                "generation_input_tokens_est_raw": effective_input_tokens if effective_input_tokens is not None else 0,
                "generation_output_tokens_est_raw": effective_output_tokens if effective_output_tokens is not None else 0,
                "generation_input_tokens_actual_raw": input_tokens_actual_raw if input_tokens_actual_raw is not None else None,
                "generation_output_tokens_actual_raw": output_tokens_actual_raw if output_tokens_actual_raw is not None else None,
            }
        )
    items.sort(key=lambda i: i["created_epoch"], reverse=True)
    return items


def _events_log_path(output_root: Path) -> Path:
    return output_root / "report_events.jsonl"


def _friendly_report_name(raw_name: str, report_type_id: str) -> str:
    short = _friendly_report_type_label(report_type_id)
    if not short:
        return raw_name
    suffix = ""
    if raw_name.startswith(report_type_id + "."):
        suffix = raw_name[len(report_type_id) + 1 :]
    elif "." in raw_name:
        suffix = raw_name.split(".", 1)[1]
    return f"{short}.{suffix}" if suffix else short


def _friendly_report_type_label(report_type_id: str) -> str:
    label_map = {
        "twamp_session_health": "twamp",
        "ms_biomarker_registry_health": "biomarkers",
        "network_queue_congestion": "network",
        "pm_export_health": "performance",
        "jira_issue_portfolio": "jira",
        "wireshark_capture_health": "wireshark",
    }
    return label_map.get(report_type_id, report_type_id)


def _friendly_model_label(model: str, provider: str = "") -> str:
    value = str(model or "").strip()
    if not value:
        return "-"
    is_claude = str(provider).strip().lower() == "claude" or value.lower().startswith("claude-")
    if is_claude:
        # Normalize dated Anthropic model ids like claude-sonnet-4-5-20250929.
        value = re.sub(r"-\d{8}$", "", value)
    return value


def _append_report_event(output_root: Path, event: dict[str, Any]) -> None:
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        path = _events_log_path(output_root)
        payload = dict(event)
        payload["ts_utc"] = datetime.now(timezone.utc).isoformat()
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception as exc:  # noqa: BLE001
        # Event logging must never break report generation.
        print(f"[rv_reporter] failed to append report event: {exc}", file=sys.stderr)
        return


def _load_report_events(output_root: Path) -> list[dict[str, Any]]:
    path = _events_log_path(output_root)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(payload, dict):
            continue
        ts = str(payload.get("ts_utc", "")).strip()
        display = "unknown"
        epoch = 0.0
        if ts:
            try:
                dt_utc = _iso_to_utc(ts)
                display = _display_local_time(dt_utc)
                epoch = dt_utc.timestamp()
            except Exception:  # noqa: BLE001
                display = ts
        report_type_id = str(payload.get("report_type_id", ""))
        raw_report_name = str(payload.get("report_name", ""))
        rows.append(
            {
                "line_index": line_index,
                "created_at": display,
                "created_epoch": epoch,
                "attempt_id": str(payload.get("attempt_id", "")),
                "report_type_id": report_type_id,
                "report_type_label": _friendly_report_type_label(report_type_id),
                "provider": str(payload.get("provider", "")),
                "model": _friendly_model_label(str(payload.get("model", "")), str(payload.get("provider", ""))),
                "status": str(payload.get("status", "")),
                "report_name": _friendly_report_name(raw_report_name, report_type_id),
                "message": str(payload.get("message", "")),
            }
        )
    rows.sort(key=lambda r: r["created_epoch"], reverse=True)
    return rows


def _generation_status_snapshot(output_root: Path) -> dict[str, Any]:
    path = _events_log_path(output_root)
    if not path.exists():
        return {
            "active_count": 0,
            "active_attempt_ids": [],
            "completed_attempts": {},
            "server_ts_ms": int(datetime.now(timezone.utc).timestamp() * 1000),
        }

    active_statuses = {"requested", "cost_estimated", "started"}
    terminal_statuses = {"finished", "failed", "timeout", "cancelled"}
    active_max_age_seconds = 15 * 60
    now_epoch = datetime.now(timezone.utc).timestamp()
    latest_by_attempt: dict[str, dict[str, Any]] = {}
    completed_attempts: dict[str, dict[str, Any]] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(payload, dict):
            continue
        attempt_id = str(payload.get("attempt_id", "")).strip()
        if not attempt_id:
            continue
        status = str(payload.get("status", "")).strip().lower()
        ts_raw = str(payload.get("ts_utc", "")).strip()
        ts_epoch = 0.0
        if ts_raw:
            try:
                ts_epoch = _iso_to_utc(ts_raw).timestamp()
            except Exception:  # noqa: BLE001
                ts_epoch = 0.0
        prev = latest_by_attempt.get(attempt_id)
        prev_epoch = float(prev.get("ts_epoch", 0.0)) if prev else 0.0
        if prev is None or ts_epoch >= prev_epoch:
            latest_by_attempt[attempt_id] = {
                "status": status,
                "ts_epoch": ts_epoch,
                "message": str(payload.get("message", "")).strip(),
                "report_name": str(payload.get("report_name", "")).strip(),
            }

    for attempt_id, info in latest_by_attempt.items():
        status = str(info.get("status", ""))
        if status in terminal_statuses:
            completed_attempts[attempt_id] = {
                "status": status,
                "message": str(info.get("message", "")),
                "report_name": str(info.get("report_name", "")),
            }

    active_attempt_ids = sorted(
        [
            attempt_id
            for attempt_id, info in latest_by_attempt.items()
            if str(info.get("status", "")) in active_statuses
            and float(info.get("ts_epoch", 0.0) or 0.0) > (now_epoch - active_max_age_seconds)
        ]
    )
    return {
        "active_count": len(active_attempt_ids),
        "active_attempt_ids": active_attempt_ids,
        "completed_attempts": completed_attempts,
        "server_ts_ms": int(datetime.now(timezone.utc).timestamp() * 1000),
    }


def _event_rows_signature(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "0:none"
    first = rows[0]
    return (
        f"{len(rows)}:"
        f"{first.get('line_index', '')}:"
        f"{first.get('status', '')}:"
        f"{first.get('attempt_id', '')}:"
        f"{first.get('created_epoch', '')}"
    )


def _resolve_csv_path(
    uploaded_file: Any,
    upload_dir: Path,
    existing_csv_path: str = "",
) -> str:
    if existing_csv_path:
        existing = _absolute_path(existing_csv_path)
        if not existing.exists():
            raise ValueError(f"Existing CSV path not found: {existing_csv_path}")
        return str(existing)

    if uploaded_file is None or not uploaded_file.filename:
        raise ValueError("Upload a file or choose one from recent uploads.")
    filename = secure_filename(uploaded_file.filename)
    if not filename.lower().endswith((".csv", ".xlsx", ".xls", ".pcap", ".pcapng")):
        raise ValueError("Supported uploads: .csv, .xlsx, .xls, .pcap, .pcapng")
    destination = upload_dir / filename
    uploaded_file.save(destination)
    return str(destination)


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
    return {
        "ops_kpi",
        "finance_variance",
        "network_queue_congestion",
        "twamp_session_health",
        "pm_export_health",
        "jira_issue_portfolio",
        "ms_biomarker_registry_health",
        "wireshark_capture_health",
    }


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
    local_dt = dt_utc.astimezone()
    now_local = datetime.now().astimezone()
    if local_dt.year == now_local.year:
        return f"{local_dt.year % 100}/{local_dt.month} {local_dt.strftime('%H:%M')}"
    return local_dt.strftime("%y-%m-%d %H:%M")


def _format_duration_seconds(value: Any) -> str:
    if value is None or value == "":
        return "-"
    try:
        total_seconds = int(round(float(value)))
    except (TypeError, ValueError):
        return "-"
    if total_seconds < 0:
        return "-"
    if total_seconds < 60:
        return str(total_seconds)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"


def _format_cost_usd(value: Any) -> str:
    if value is None or value == "":
        return "-"


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _token_set_for_similarity(text: str) -> set[str]:
    if not text:
        return set()
    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "but",
        "not",
        "into",
        "over",
        "under",
        "very",
        "more",
        "less",
        "than",
        "only",
        "also",
        "can",
        "will",
        "may",
        "report",
        "data",
    }
    tokens = set()
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if len(word) < 3 or word in stop_words:
            continue
        tokens.add(word)
    return tokens


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _benchmark_text_blob(report: dict[str, Any]) -> str:
    parts = [str(report.get("summary_text", "") or "")]
    sections = report.get("sections", []) or []
    for section in sections:
        if isinstance(section, dict):
            parts.append(str(section.get("title", "") or ""))
            parts.append(str(section.get("body", "") or ""))
    alerts = report.get("alerts", []) or []
    for alert in alerts:
        if isinstance(alert, dict):
            parts.append(str(alert.get("message", "") or ""))
    recommendations = report.get("recommendations", []) or []
    for rec in recommendations:
        if isinstance(rec, dict):
            parts.append(str(rec.get("action", "") or ""))
    return "\n".join(parts).lower()


def _benchmark_contains_any(text: str, words: set[str]) -> bool:
    return any(word in text for word in words)


def _benchmark_count_hits(text: str, words: set[str]) -> int:
    return sum(1 for word in words if word in text)


def _benchmark_overall_depth(report: dict[str, Any]) -> str:
    score = int(report.get("sections", 0)) + int(report.get("alerts", 0)) + int(report.get("recommendations", 0))
    words = int(report.get("summary_words", 0))
    if score >= 14 or words >= 180:
        return "Detailed analytical + operational"
    if score >= 8 or words >= 110:
        return "Balanced analytical summary"
    return "High-level executive summary"


def _benchmark_technical_rigor(report: dict[str, Any]) -> str:
    text = str(report.get("text_blob", "") or "")
    stats_terms = {"correlation", "spearman", "pearson", "confidence", "paired", "distribution", "percentile"}
    hits = _benchmark_count_hits(text, stats_terms)
    if hits >= 4:
        return "High"
    if hits >= 2:
        return "Moderate"
    return "Basic"


def _benchmark_exec_friendliness(report: dict[str, Any]) -> str:
    words = int(report.get("summary_words", 0))
    if words <= 90:
        return "Very readable"
    if words <= 160:
        return "Readable"
    return "Technical-heavy"


def _benchmark_length_detail(report: dict[str, Any]) -> str:
    words = int(report.get("summary_words", 0))
    if words <= 90:
        return "Shorter"
    if words <= 170:
        return "Medium detail"
    return "Comprehensive"


def _benchmark_recommendation_scope(text: str) -> str:
    ops_terms = {"pipeline", "validate", "normalize", "reshape", "monitor", "threshold", "quality"}
    if _benchmark_count_hits(text, ops_terms) >= 3:
        return "Operational / implementation-ready"
    return "Strategic / general"


def _benchmark_data_engineering_guidance(text: str) -> str:
    terms = {"id", "pid", "sid", "date", "normalize", "reshape", "validation", "uniqueness", "missing"}
    if _benchmark_count_hits(text, terms) >= 4:
        return "Strong"
    if _benchmark_count_hits(text, terms) >= 2:
        return "Moderate"
    return "Limited"


def _benchmark_modeling_guidance(text: str) -> str:
    terms = {"regression", "model", "multicollinearity", "feature", "factor", "imputation", "mixed-effects"}
    if _benchmark_count_hits(text, terms) >= 3:
        return "Strong"
    if _benchmark_count_hits(text, terms) >= 1:
        return "Moderate"
    return "Limited"


def _benchmark_longitudinal_readiness(text: str) -> str:
    terms = {"follow-up", "longitudinal", "time series", "delta", "interval", "slope"}
    if _benchmark_count_hits(text, terms) >= 3:
        return "Explicitly prepared"
    if _benchmark_count_hits(text, terms) >= 1:
        return "Suggested"
    return "Not explicit"


def _benchmark_ops_signal_score(text: str) -> int:
    terms = {"pipeline", "validate", "normalize", "reshape", "monitor", "threshold", "quality", "operational"}
    return _benchmark_count_hits(text, terms)


def _benchmark_stats_signal_score(text: str) -> int:
    terms = {"correlation", "spearman", "pearson", "distribution", "confidence", "paired", "evidence", "defensible"}
    return _benchmark_count_hits(text, terms)


def _baseline_check_status(full_match: bool, almost_match: bool = False, weak_match: bool = False) -> tuple[str, int]:
    if full_match:
        return "yes", 3
    if almost_match:
        return "almost", 2
    if weak_match:
        return "weak", 1
    return "no", 0


def _compact_values_display(values: list[Any], none_text: str = "-") -> str:
    cleaned = [none_text if v is None else str(v).strip() for v in values]
    cleaned = [v if v else none_text for v in cleaned]
    unique_ordered = list(dict.fromkeys(cleaned))
    if not unique_ordered:
        return none_text
    if len(unique_ordered) == 1:
        return unique_ordered[0]
    return " | ".join(unique_ordered)


def _infer_cost_from_tokens_and_model(input_tokens: Any, output_tokens: Any, model: str) -> float | None:
    pricing = _model_pricing_tuple(model)
    if pricing is None:
        return None
    in_tok = _safe_float(input_tokens)
    out_tok = _safe_float(output_tokens)
    input_per_1m, output_per_1m, _ = pricing
    return round((in_tok / 1_000_000) * input_per_1m + (out_tok / 1_000_000) * output_per_1m, 6)
    try:
        return f"${float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def _model_options_with_cost_ratio() -> list[dict[str, str]]:
    return [{"value": model, "label": _model_ratio_label(model)} for model in sorted(MODEL_PRICING_USD.keys())]


def _model_ratio_label(model: str) -> str:
    ratio = _model_cost_ratio_vs_baseline(model)
    base_label = model
    if ratio is None:
        return f"{base_label} / n/a"
    return f"{base_label} / {_ratio_text(ratio)}"


def _model_cost_ratio_vs_baseline(model: str) -> float | None:
    baseline_model = "gpt-5-mini"
    baseline = MODEL_PRICING_USD.get(baseline_model)
    current = _model_pricing_tuple(model)
    if baseline is None or current is None:
        return None
    baseline_cost = _calc_reference_cost(baseline.input_per_1m, baseline.output_per_1m)
    if baseline_cost <= 0:
        return None
    input_per_1m, output_per_1m, _ = current
    model_cost = _calc_reference_cost(input_per_1m, output_per_1m)
    return model_cost / baseline_cost


def _model_pricing_tuple(model: str) -> tuple[float, float, str] | None:
    canonical = _canonical_pricing_model_key(model)
    pricing = MODEL_PRICING_USD.get(canonical)
    if pricing is not None:
        return pricing.input_per_1m, pricing.output_per_1m, PRICING_SOURCE_URL
    dynamic = OPENROUTER_DYNAMIC_MODEL_PRICING_USD.get(canonical)
    if dynamic is not None:
        return dynamic
    return NON_OPENAI_MODEL_PRICING_USD.get(canonical)


def _is_priced_model(model: str) -> bool:
    return _model_pricing_tuple(model) is not None


def _sort_models_by_ratio(models: list[str]) -> list[str]:
    def _sort_key(name: str) -> tuple[float, str]:
        ratio = _model_cost_ratio_vs_baseline(name)
        if ratio is None:
            return (float("inf"), name)
        return (ratio, name)

    return sorted(models, key=_sort_key)


def _canonical_pricing_model_key(model: str) -> str:
    key = model.strip().lower()
    if not key:
        return model

    # OpenRouter/aggregator prefixed model ids (e.g. anthropic/claude-sonnet-4.6).
    tail = key.split("/", 1)[1] if "/" in key else key

    # Claude family normalization for live API model ids that may include dated suffixes.
    if tail.startswith("claude"):
        claude = tail.replace(".", "-")
        claude = re.sub(r"-\d{8}$", "", claude)
        if claude.startswith("claude-sonnet-4-6"):
            return "claude-sonnet-4-6"
        if claude.startswith("claude-sonnet-4-5"):
            return "claude-sonnet-4-5"
        if claude.startswith("claude-sonnet-4"):
            return "claude-sonnet-4"
        if claude.startswith("claude-opus-4-6"):
            return "claude-opus-4-6"
        if claude.startswith("claude-opus-4-5"):
            return "claude-opus-4-5"
        if claude.startswith("claude-opus-4-1"):
            return "claude-opus-4-1"
        if claude.startswith("claude-opus-4"):
            return "claude-opus-4"
        if claude.startswith("claude-haiku-4-5"):
            return "claude-haiku-4-5"
        if claude.startswith("claude-3-7-sonnet"):
            return "claude-3-7-sonnet-latest"
        if claude.startswith("claude-3-haiku") or claude.startswith("claude-3-5-haiku"):
            return "claude-3-haiku"
        return claude

    if key.startswith("claude-sonnet-4-5"):
        return "claude-sonnet-4-5"
    if key.startswith("claude-opus-4-1"):
        return "claude-opus-4-1"
    if key.startswith("claude-3-7-sonnet"):
        return "claude-3-7-sonnet-latest"
    if "claude" in key and "sonnet" in key and "4-5" in key:
        return "claude-sonnet-4-5"
    if "claude" in key and "opus" in key and "4-1" in key:
        return "claude-opus-4-1"

    return key


def _model_reference_cost_and_ratio(model: str) -> dict[str, str]:
    pricing = _model_pricing_tuple(model)
    if pricing is None:
        return {
            "cost_text": "n/a",
            "ratio_text": "n/a",
            "source_url": "",
        }
    input_per_1m, output_per_1m, source_url = pricing
    reference_cost = _calc_reference_cost(input_per_1m, output_per_1m)
    ratio = _model_cost_ratio_vs_baseline(model)
    return {
        "cost_text": f"${reference_cost:.6f}",
        "ratio_text": _ratio_text(ratio) if ratio is not None else "n/a",
        "source_url": source_url,
    }


def _calc_reference_cost(input_per_1m: float, output_per_1m: float) -> float:
    return (1000 / 1_000_000) * input_per_1m + (100 / 1_000_000) * output_per_1m


def _ratio_text(ratio: float) -> str:
    if ratio >= 10:
        return f"{ratio:.0f}x"
    if ratio >= 1:
        text = f"{ratio:.1f}".rstrip("0").rstrip(".")
        return f"{text}x"
    text = f"{ratio:.2f}".rstrip("0").rstrip(".")
    return f"{text}x"


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

    if (report_types_dir / f"{report_type_id}.yaml").exists():
        raise ValueError(
            f"report_type_id '{report_type_id}' already exists. "
            "Please change report_type_id before saving."
        )
    normalized["report_type_id"] = report_type_id
    return normalized


def _fetch_provider_models(provider_id: str, api_key: str, base_url: str) -> tuple[list[str], str | None]:
    if not api_key and provider_id != "openrouter":
        return [], "Missing API token in environment."
    try:
        if provider_id in {"openai", "xai"}:
            endpoint = f"{base_url.rstrip('/')}/models"
            payload = _http_get_json(endpoint, headers={"authorization": f"Bearer {api_key}"})
            rows = payload.get("data", []) if isinstance(payload, dict) else []
            models = [str(item.get("id", "")).strip() for item in rows if isinstance(item, dict)]
            if provider_id == "openai":
                frontier = list(OPENAI_FRONTIER_MODEL_LABELS.keys())
                available = set(m for m in models if m)
                filtered = [m for m in frontier if m in available]
                if filtered:
                    return filtered, None
                return frontier, None
            return sorted({m for m in models if m}), None

        if provider_id == "openrouter":
            endpoint = f"{base_url.rstrip('/')}/models"
            headers: dict[str, str] = {}
            if api_key:
                headers["authorization"] = f"Bearer {api_key}"
            payload = _http_get_json(endpoint, headers=headers)
            rows = payload.get("data", []) if isinstance(payload, dict) else []
            models = [str(item.get("id", "")).strip() for item in rows if isinstance(item, dict)]
            _cache_openrouter_model_pricing(rows)
            return sorted({m for m in models if m}), None

        if provider_id == "claude":
            endpoint = f"{base_url.rstrip('/')}/models"
            payload = _http_get_json(
                endpoint,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
            rows = payload.get("data", []) if isinstance(payload, dict) else []
            models = [str(item.get("id", "")).strip() for item in rows if isinstance(item, dict)]
            return sorted({m for m in models if m}), None

        if provider_id == "gemini":
            endpoint = "https://generativelanguage.googleapis.com/v1beta/models"
            query = urlparse.urlencode({"key": api_key})
            payload = _http_get_json(f"{endpoint}?{query}")
            rows = payload.get("models", []) if isinstance(payload, dict) else []
            names = []
            for item in rows:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if name.startswith("models/"):
                    name = name.split("/", 1)[1]
                if name:
                    names.append(name)
            return sorted(set(names)), None

        return [], "Provider model listing not implemented."
    except HTTPError as exc:
        return [], f"HTTP {exc.code}: {exc.reason}"
    except URLError as exc:
        return [], f"Connection failed: {exc.reason}"
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)


def _http_get_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    req = urlrequest.Request(url=url, method="GET", headers=headers or {})
    with urlrequest.urlopen(req, timeout=12) as resp:  # noqa: S310
        raw = resp.read().decode("utf-8")
    payload = json.loads(raw)
    if isinstance(payload, dict):
        return payload
    return {}


def _cache_openrouter_model_pricing(rows: Any) -> None:
    if not isinstance(rows, list):
        return
    updated: dict[str, tuple[float, float, str]] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id", "")).strip()
        if not model_id:
            continue
        pricing = item.get("pricing", {})
        if not isinstance(pricing, dict):
            continue
        prompt_per_token = _safe_float(pricing.get("prompt"))
        completion_per_token = _safe_float(pricing.get("completion"))
        if prompt_per_token <= 0 and completion_per_token <= 0:
            continue
        input_per_1m = prompt_per_token * 1_000_000
        output_per_1m = completion_per_token * 1_000_000
        canonical = _canonical_pricing_model_key(model_id)
        updated[canonical] = (input_per_1m, output_per_1m, "https://openrouter.ai/models")
    if updated:
        OPENROUTER_DYNAMIC_MODEL_PRICING_USD.update(updated)


def _reference_cost_text_for_provider(provider_id: str) -> str:
    if provider_id != "openai":
        return "See provider pricing page"
    baseline_model = "gpt-5-mini"
    pricing = MODEL_PRICING_USD.get(baseline_model)
    if pricing is None:
        return "Not configured"
    return f"${_calc_reference_cost(pricing.input_per_1m, pricing.output_per_1m):.6f} (1000 in / 100 out)"


def _estimate_provider_cost(
    *,
    provider_name: str,
    model: str,
    prompt_text: str,
    estimated_output_tokens: int,
    output_root: Path,
    report_type_id: str,
) -> dict[str, float | int | str]:
    effective_output_tokens = _scaled_output_tokens(provider_name, estimated_output_tokens)
    scale_factor = round(effective_output_tokens / max(1, int(estimated_output_tokens)), 2)
    if provider_name == "openai":
        if model not in MODEL_PRICING_USD:
            raise ValueError(f"No price map configured for OpenAI model '{model}'.")
        estimate = estimate_openai_cost(
            model=model,
            prompt_text=prompt_text,
            estimated_output_tokens=effective_output_tokens,
        )
        estimate["output_tokens_user_budget"] = int(max(1, int(estimated_output_tokens)))
        estimate["output_tokens_scale_factor"] = scale_factor
        return _apply_cost_calibration_if_available(
            estimate=estimate,
            output_root=output_root,
            provider_name=provider_name,
            model=model,
            report_type_id=report_type_id,
        )

    pricing = _model_pricing_tuple(model)
    if pricing is None:
        raise ValueError(f"No price map configured for provider '{provider_name}' model '{model}'.")

    input_per_1m, output_per_1m, source_url = pricing
    input_tokens = estimate_tokens(prompt_text)
    output_tokens = max(1, int(effective_output_tokens))
    input_cost = (input_tokens / 1_000_000) * input_per_1m
    output_cost = (output_tokens / 1_000_000) * output_per_1m
    total_cost = input_cost + output_cost
    estimate = {
        "model": model,
        "input_tokens_est": input_tokens,
        "output_tokens_est": output_tokens,
        "output_tokens_user_budget": int(max(1, int(estimated_output_tokens))),
        "output_tokens_scale_factor": scale_factor,
        "input_cost_usd_est": round(input_cost, 6),
        "output_cost_usd_est": round(output_cost, 6),
        "total_cost_usd_est": round(total_cost, 6),
        "pricing_source_url": source_url,
        "pricing_verified_date": PRICING_VERIFIED_DATE,
    }
    return _apply_cost_calibration_if_available(
        estimate=estimate,
        output_root=output_root,
        provider_name=provider_name,
        model=model,
        report_type_id=report_type_id,
    )


def _apply_cost_calibration_if_available(
    *,
    estimate: dict[str, float | int | str],
    output_root: Path,
    provider_name: str,
    model: str,
    report_type_id: str,
) -> dict[str, float | int | str]:
    calibrated = dict(estimate)
    raw_total = _safe_float(calibrated.get("total_cost_usd_est"))
    calibrated["raw_total_cost_usd_est"] = round(raw_total, 6)
    calibrated["calibration_applied"] = False
    calibrated["calibration_factor"] = 1.0
    calibrated["calibration_sample_size"] = 0
    calibrated["calibration_scope"] = "none"
    factor = _provider_cost_factor(provider_name)
    if factor <= 0:
        factor = 1.0
    calibrated_total = round(raw_total * factor, 6)
    calibrated["total_cost_usd_est"] = calibrated_total
    calibrated["calibration_applied"] = abs(factor - 1.0) > 1e-9
    calibrated["calibration_factor"] = round(factor, 4)
    calibrated["calibration_sample_size"] = 0
    calibrated["calibration_scope"] = "provider_factor"
    return calibrated


def _provider_cost_factor(provider_name: str) -> float:
    provider = str(provider_name or "").strip().lower()
    # Claude is treated as baseline (no extra tuning factor).
    if provider == "claude":
        return 1.0
    env_key = f"RV_COST_FACTOR_{provider.upper()}" if provider else ""
    raw = os.getenv(env_key, "").strip() if env_key else ""
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return 1.0


def _resolve_cost_calibration(
    *,
    output_root: Path,
    provider_name: str,
    model: str,
    report_type_id: str,
) -> tuple[float, int, str] | None:
    ratios = _collect_actual_to_estimated_ratios(output_root)
    if not ratios:
        return None

    key_order = [
        ("provider_model_report", (provider_name, model, report_type_id)),
        ("provider_model", (provider_name, model, "")),
        ("provider_report", (provider_name, "", report_type_id)),
        ("provider", (provider_name, "", "")),
        ("global", ("", "", "")),
    ]
    min_samples_by_scope = {
        "provider_model_report": 2,
        "provider_model": 2,
        "provider_report": 2,
        "provider": 2,
        "global": 3,
    }

    for scope, key in key_order:
        values = ratios.get(key, [])
        if len(values) < min_samples_by_scope.get(scope, 2):
            continue
        factor = _median(values)
        if factor <= 0:
            continue
        return factor, len(values), scope
    return None


def _collect_actual_to_estimated_ratios(output_root: Path) -> dict[tuple[str, str, str], list[float]]:
    rows: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for path in output_root.rglob("*.report.json"):
        try:
            payload = _load_json(path)
        except Exception:  # noqa: BLE001
            continue
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        actual_cost = _safe_float(metadata.get("actual_cost_usd_user"))
        estimated_cost = _safe_float(metadata.get("generation_cost_usd_est"))
        if actual_cost <= 0 or estimated_cost <= 0:
            continue
        ratio = actual_cost / estimated_cost
        if not (0.05 <= ratio <= 20):
            continue

        provider = str(metadata.get("generation_backend", "")).strip().lower()
        model = str(metadata.get("generation_model", "")).strip()
        report_type_id = str(payload.get("report_type_id", "")).strip()
        rows[(provider, model, report_type_id)].append(ratio)
        rows[(provider, model, "")].append(ratio)
        rows[(provider, "", report_type_id)].append(ratio)
        rows[(provider, "", "")].append(ratio)
        rows[("", "", "")].append(ratio)
    return rows


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _scaled_output_tokens(provider_name: str, requested_tokens: int) -> int:
    base = max(1, int(requested_tokens))
    multiplier = {
        "claude": 1.0,
        "gemini": 1.5,
        "xai": 1.4,
        "openai": 1.0,
    }.get(provider_name, 1.0)
    return max(1, int(round(base * multiplier)))


def main() -> None:
    profile = os.getenv("APP_ENV", "sandbox")
    load_env_profile(profile)
    app = create_app()
    debug = os.getenv("FLASK_DEBUG", "1").lower() in {"1", "true", "yes"}
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    app.run(debug=debug, host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
