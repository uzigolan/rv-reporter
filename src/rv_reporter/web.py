from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
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
import pandas as pd

from rv_reporter.orchestrator import prepare_pipeline_inputs, run_pipeline
from rv_reporter.providers.anthropic_provider import AnthropicMessagesProvider
from rv_reporter.providers.mock_provider import MockProvider
from rv_reporter.providers.openai_chat_provider import OpenAIChatCompletionsProvider
from rv_reporter.providers.openai_provider import (
    OpenAIResponsesProvider,
    build_model_prompt_for_estimation,
)
from rv_reporter.rendering.html_renderer import render_html
from rv_reporter.rendering.pdf_renderer import render_pdf
from rv_reporter.report_types.scaffold import DOMAINS, FAMILIES, MODES, scaffold_report_type
from rv_reporter.report_types.plugins import list_supported_metrics_profiles, invalidate_plugin_cache
from rv_reporter.report_types.registry import ReportTypeRegistry
from rv_reporter.services.cost_estimator import (
    MODEL_PRICING_USD,
    PRICING_SOURCE_URL,
    PRICING_VERIFIED_DATE,
    estimate_openai_cost,
    estimate_tokens,
)
from rv_reporter.services.ingest import describe_tabular_source, list_excel_sheets, load_csv_with_limit, preflight_tabular_source
from rv_reporter.services.profiler import profile_dataframe

PROTECTED_REPORT_TYPES = {
    "network_queue_congestion",
    "twamp_session_health",
    "pm_export_health",
    "jira_issue_portfolio",
    "ms_biomarker_registry_health",
}
REPORT_TYPE_LABEL_MAP: dict[str, str] = {
    "twamp_session_health": "twamp",
    "ms_biomarker_registry_health": "biomarkers",
    "network_queue_congestion": "network",
    "pm_export_health": "performance",
    "jira_issue_portfolio": "jira",
    "wireshark_capture_health": "wireshark",
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
UI_BUILD_MARKER = "generate-ui-2026-02-22-multi-source-v2"

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
    app.config["PLUGIN_ROOT"] = str(_absolute_path("report_type_plugins"))
    app.config["REPORT_TYPE_AGENT_MODEL"] = os.getenv("REPORT_TYPE_AGENT_MODEL", "gpt-4o")
    app.config.update(config_overrides or {})
    app.config["UPLOAD_FOLDER"] = str(_absolute_path(app.config["UPLOAD_FOLDER"]))
    app.config["OUTPUT_FOLDER"] = str(_absolute_path(app.config["OUTPUT_FOLDER"]))
    app.config["REPORT_TYPES_DIR"] = str(_absolute_path(app.config["REPORT_TYPES_DIR"]))
    app.config["PLUGIN_ROOT"] = str(_absolute_path(app.config["PLUGIN_ROOT"]))
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["OUTPUT_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["REPORT_TYPES_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["PLUGIN_ROOT"]).mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def index() -> str:
        registry = ReportTypeRegistry(config_dir=Path(app.config["REPORT_TYPES_DIR"]))
        report_types = _visible_report_types_for_generation(
            registry.list_report_types(),
            Path(app.config["PLUGIN_ROOT"]),
        )
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
            ui_build_marker=UI_BUILD_MARKER,
            report_types=report_types,
            report_type_options=_report_type_options(report_types),
            recent_uploads=_recent_uploaded_sources(Path(app.config["UPLOAD_FOLDER"])),
            sheet_options=[],
            selected_sheet="",
            existing_csv_path="",
            defaults={
                "provider": default_provider,
                "model": "gpt-5-mini",
                "row_limit": None,
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

    @app.get("/logic")
    def logic() -> str:
        return render_template("logic.html")

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
            report_types=registry.list_report_types(),
            prompt_text="",
            selected_clone_type="",
            report_type_agent_model=app.config["REPORT_TYPE_AGENT_MODEL"],
            recent_uploads=_recent_uploaded_sources(Path(app.config["UPLOAD_FOLDER"])),
            existing_csv_path="",
            selected_sheet="",
            sheet_options=[],
            source_summary=None,
            source_sample_percent=100,
            families=sorted(FAMILIES),
            domains=sorted(DOMAINS),
            modes=sorted(MODES),
        )

    @app.post("/api/recommend-classification")
    def recommend_classification_api() -> Any:
        """Recommend domain/family/mode from source column names."""
        data = request.get_json(silent=True) or {}
        columns = [str(c).strip().lower() for c in data.get("columns", []) if str(c).strip()]
        if not columns:
            return jsonify({"domain": "", "family": "", "mode": "", "confidence": "none"})

        joined = " ".join(columns)

        # ── Domain scoring ───────────────────────────────────────────
        domain_signals: dict[str, list[str]] = {
            "networking": ["latency", "packet", "throughput", "bandwidth", "queue", "interface", "flow", "byte", "octets", "port"],
            "telecom": ["twamp", "session", "bearer", "handover", "rssi", "snr", "cell", "lte", "pdv", "ipdv", "delay", "jitter"],
            "security": ["attack", "threat", "anomaly", "intrusion", "vulnerability", "auth", "firewall", "alert", "severity"],
            "finance": ["revenue", "cost", "budget", "variance", "profit", "forecast", "spend", "amount", "invoice"],
            "operations": ["uptime", "sla", "incident", "kpi", "utilization", "capacity", "maintenance", "device", "property", "sensor"],
            "observability": ["metric", "trace", "error_rate", "service", "endpoint", "percentile", "span"],
            "healthcare": ["patient", "biomarker", "clinical", "diagnosis", "lab", "cohort", "specimen", "assisted", "living", "vital", "heart", "oxygen", "temperature", "blood", "pulse"],
            "sales": ["pipeline", "conversion", "opportunity", "deal", "quota", "lead", "funnel"],
            "product": ["user", "event", "feature", "engagement", "retention", "churn", "signup"],
            "customer_support": ["ticket", "case", "resolution", "sla", "escalation", "agent", "satisfaction"],
            "supply_chain": ["warehouse", "shipment", "inventory", "supplier", "sku", "order", "fulfillment"],
            "manufacturing": ["defect", "yield", "batch", "production", "downtime", "oee", "scrap"],
            "energy": ["consumption", "kwh", "power", "meter", "generation", "grid", "solar", "wind"],
            "project_management": ["issue", "sprint", "story", "epic", "assignee", "priority", "status"],
            "research": ["sample", "experiment", "measurement", "variable", "observation", "trial"],
            "education": ["student", "grade", "course", "score", "enrollment", "attendance"],
            "government": ["census", "population", "district", "regulation", "compliance", "agency"],
        }
        # Also match against the filename if present
        filename_hint = data.get("filename", "").lower()
        domain_scores: dict[str, int] = {}
        for dom, keywords in domain_signals.items():
            col_hits = sum(1 for kw in keywords if kw in joined)
            fname_hits = sum(1 for kw in keywords if kw in filename_hint) if filename_hint else 0
            # Filename matches are a stronger signal (worth 2 each)
            total = col_hits + fname_hits * 2
            if total:
                domain_scores[dom] = total

        best_domain = ""
        if domain_scores:
            best_domain = max(domain_scores, key=lambda d: domain_scores[d])
        elif has_timestamp:
            best_domain = "operations"

        # ── Family inference ─────────────────────────────────────────
        has_timestamp = any(kw in joined for kw in ["time", "date", "timestamp", "start", "elapsed", "interval", "epoch"])
        has_event = any(kw in joined for kw in ["event", "log", "message", "severity", "type", "action"])
        has_text = any(kw in joined for kw in ["log", "message", "text", "body", "description", "comment"])
        has_entity = any(kw in joined for kw in ["id", "name", "status", "type", "category"])

        if has_timestamp and not has_event:
            best_family = "time_series"
        elif has_event and has_text:
            best_family = "log_text"
        elif has_event:
            best_family = "event"
        elif has_entity and not has_timestamp:
            best_family = "entity_snapshot"
        else:
            best_family = "tabular_statistical"

        # Refine with domain-family map
        from rv_reporter.report_types.scaffold import FAMILIES as _ALL_FAMILIES  # noqa: F811
        domain_family_map = {
            "networking": ["time_series", "event", "hybrid"],
            "telecom": ["time_series", "event"],
            "observability": ["time_series", "hybrid"],
            "security": ["event", "log_text", "hybrid"],
            "operations": ["time_series", "tabular_statistical", "hybrid"],
            "manufacturing": ["time_series", "tabular_statistical"],
            "supply_chain": ["time_series", "relational"],
            "energy": ["time_series", "tabular_statistical"],
            "finance": ["tabular_statistical", "relational"],
            "sales": ["tabular_statistical", "relational"],
            "product": ["tabular_statistical", "event"],
            "customer_support": ["tabular_statistical", "event"],
            "healthcare": ["tabular_statistical", "entity_snapshot"],
            "research": ["tabular_statistical", "time_series"],
            "education": ["tabular_statistical", "entity_snapshot"],
            "government": ["tabular_statistical", "relational"],
            "project_management": ["entity_snapshot", "relational"],
        }
        valid_families = domain_family_map.get(best_domain, list(_ALL_FAMILIES))
        if best_family not in valid_families and valid_families:
            best_family = valid_families[0]

        # ── Mode inference ───────────────────────────────────────────
        has_threshold = any(kw in joined for kw in ["threshold", "sla", "limit", "max", "min", "critical", "violation"])
        has_anomaly = any(kw in joined for kw in ["anomaly", "outlier", "deviation", "abnormal"])
        has_loss = any(kw in joined for kw in ["loss", "error", "fail", "drop", "reject", "fault"])
        has_score = any(kw in joined for kw in ["score", "health", "rating", "index", "grade"])

        if has_threshold:
            best_mode = "threshold_sla"
        elif has_anomaly:
            best_mode = "anomaly_detection"
        elif has_loss:
            best_mode = "issue_detection"
        elif has_score:
            best_mode = "health_score"
        elif has_timestamp:
            best_mode = "trend_analysis"
        else:
            best_mode = "overview_summary"

        total_hits = sum(domain_scores.values()) if domain_scores else 0
        if total_hits >= 4:
            confidence = "high"
        elif total_hits >= 2:
            confidence = "medium"
        elif total_hits >= 1 or best_domain:
            confidence = "low"
        else:
            confidence = "none"

        return jsonify({
            "domain": best_domain,
            "family": best_family,
            "mode": best_mode,
            "confidence": confidence,
        })

    @app.post("/api/validate-prompt")
    def validate_prompt_api() -> Any:
        data = request.get_json(silent=True) or {}
        prompt_text = str(data.get("prompt_text", "")).strip()
        hint_domain = str(data.get("hint_domain", "")).strip()

        issues: list[str] = []
        warnings: list[str] = []
        suggestions: list[str] = []

        word_count = len(prompt_text.split()) if prompt_text else 0
        lower = prompt_text.lower()

        # ── Issues (blocking quality problems, -30 each) ─────────────
        if word_count < 10:
            issues.append("Prompt is too short. Describe the report purpose, key columns, and what to detect.")

        # ── Warnings (notable gaps, -15 each) ────────────────────────
        if word_count >= 10:
            # Must mention specific column names (exact identifiers, not just "csv" or "columns")
            col_pattern = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+){1,}\b")
            has_specific_columns = bool(col_pattern.search(lower))
            if not has_specific_columns:
                warnings.append(
                    "No specific column names detected (e.g. 'delay_ms', 'packet_loss_pct'). "
                    "Explicit column names improve schema accuracy significantly."
                )

            # Must define at least one threshold or numeric criterion
            has_threshold = bool(re.search(r"\b(\d+[\.,]?\d*\s*(?:ms|pct|%|dbm|db|s|k|m|g|x)?)\b|"
                                           r"[<>≤≥]=?\s*\d|"
                                           r"\b(threshold|sla|limit|ceiling|floor|baseline|acceptable)\b", lower))
            if not has_threshold:
                warnings.append(
                    "No thresholds or criteria defined. Specify what values constitute 'good', 'degraded', or 'critical' "
                    "(e.g. 'RSSI < -110 dBm is poor', 'latency > 50ms is critical')."
                )

            # Unanswered context questions: if question lines are present but no numeric answers follow
            question_lines = [l.strip() for l in prompt_text.splitlines() if l.strip().endswith("?")]
            if question_lines:
                # Check whether any line after a question contains a number or answer-like text
                answered = bool(re.search(r"\b\d+\b|\b(per.session|per.device|aggregate|absolute|relative|trend)\b", lower))
                if not answered:
                    warnings.append(
                        f"Prompt contains {len(question_lines)} unanswered question(s). "
                        "Fill in at least some answers (e.g. thresholds, aggregation level, definition of 'degraded') "
                        "so the AI generates targeted metrics instead of generic ones."
                    )

            # Must mention derived / computed metrics
            has_derived = bool(re.search(
                r"\b(average|avg|mean|median|p95|p99|percentile|ratio|rate|count|sum|trend|score|index|distribution)\b", lower
            ))
            if not has_derived:
                warnings.append(
                    "No derived metrics mentioned. Specify what to compute "
                    "(e.g. 'per-session average delay', 'packet loss rate', 'top-10 degraded sessions')."
                )

        # ── Suggestions (quality improvements, -10 each) ─────────────
        if word_count >= 10:
            # Analysis granularity
            has_granularity = bool(re.search(
                r"\b(per.session|per.device|per.flow|per.ue|per.interface|aggregate|overall|by.type|by.region|by.bearer)\b", lower
            ))
            if not has_granularity:
                suggestions.append(
                    "Specify the analysis granularity: should metrics be per-session, per-device, aggregate, or by type/category?"
                )

            # Intent verb
            intent_verbs = ["analyze", "detect", "summarize", "identify", "track", "monitor",
                            "report", "trend", "compare", "rank", "forecast", "highlight", "flag", "surface"]
            if not any(v in lower for v in intent_verbs):
                suggestions.append("Add a clear action verb (e.g. 'detect', 'identify', 'rank', 'flag').")

            # Domain keyword match
            domain_keywords: dict[str, list[str]] = {
                "networking": ["latency", "packet", "throughput", "bandwidth", "queue", "interface", "flow"],
                "security": ["attack", "threat", "anomaly", "intrusion", "vulnerability", "auth", "firewall"],
                "finance": ["revenue", "cost", "budget", "variance", "profit", "forecast", "spend"],
                "operations": ["uptime", "sla", "incident", "kpi", "utilization", "capacity", "maintenance"],
                "observability": ["metric", "trace", "latency", "error_rate", "service", "endpoint", "percentile"],
                "telecom": ["session", "bearer", "ue", "handover", "rssi", "snr", "cell", "lte"],
                "healthcare": ["patient", "biomarker", "clinical", "diagnosis", "lab", "cohort"],
                "sales": ["pipeline", "conversion", "opportunity", "deal", "quota", "forecast"],
            }
            if hint_domain and hint_domain in domain_keywords:
                found = [kw for kw in domain_keywords[hint_domain] if kw in lower]
                if not found:
                    suggestions.append(
                        f"Domain is '{hint_domain}' but no typical terms found "
                        f"(e.g. {', '.join(domain_keywords[hint_domain][:4])}). Add domain context."
                    )

            # Mention of alert/output expectations
            has_output_hint = bool(re.search(r"\b(alert|warn|flag|section|chart|table|summary|recommendation)\b", lower))
            if not has_output_hint:
                suggestions.append(
                    "Mention expected output sections (e.g. 'produce a summary table', 'flag sessions as alerts', 'include trend charts')."
                )

        # ── Score ─────────────────────────────────────────────────────
        score = 100
        score -= len(issues) * 30
        score -= len(warnings) * 15
        score -= len(suggestions) * 10
        score = max(0, min(100, score))

        return jsonify({
            "valid": len(issues) == 0,
            "score": score,
            "issues": issues,
            "warnings": warnings,
            "suggestions": suggestions,
        })

    @app.post("/api/improve-prompt")
    def improve_prompt_api() -> Any:
        data = request.get_json(silent=True) or {}
        prompt_text = str(data.get("prompt_text", "")).strip()
        hint_domain = str(data.get("hint_domain", "")).strip() or None
        hint_family = str(data.get("hint_family", "")).strip() or None
        hint_mode = str(data.get("hint_mode", "")).strip() or None
        source_columns = [
            str(item).strip()
            for item in data.get("source_columns", [])
            if str(item).strip()
        ]
        if not prompt_text:
            return jsonify({"error": "Prompt text is required."}), 400
        try:
            result = _improve_report_type_prompt(
                prompt_text=prompt_text,
                model=str(app.config["REPORT_TYPE_AGENT_MODEL"]),
                hint_domain=hint_domain,
                hint_family=hint_family,
                hint_mode=hint_mode,
                source_columns=source_columns,
            )
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)

    @app.get("/api/report-type-yaml")
    def report_type_yaml() -> Any:
        report_type_id = request.args.get("report_type_id", "").strip()
        if not report_type_id:
            return jsonify({"error": "Missing report_type_id"}), 400
        path = Path(app.config["REPORT_TYPES_DIR"]) / f"{report_type_id}.yaml"
        if not path.exists():
            return jsonify({"error": f"Unknown report_type_id: {report_type_id}"}), 404
        return jsonify({"report_type_id": report_type_id, "yaml": path.read_text(encoding="utf-8")})

    def _report_type_review_context(report_type_id: str) -> dict[str, str]:
        contexts = session.get("report_type_review_contexts", {})
        if not isinstance(contexts, dict):
            return {}
        context = contexts.get(report_type_id, {})
        return context if isinstance(context, dict) else {}

    def _store_report_type_review_context(
        report_type_id: str,
        *,
        source_path: str = "",
        sheet_name: str = "",
        draft_workflow: list[dict[str, Any]] | None = None,
    ) -> None:
        payload: dict[str, str] = {}
        if source_path:
            payload["existing_csv_path"] = source_path
        if sheet_name:
            payload["sheet_name"] = sheet_name
        if draft_workflow:
            payload["draft_workflow"] = json.dumps(draft_workflow)
        if not payload:
            return

        contexts = session.get("report_type_review_contexts", {})
        if not isinstance(contexts, dict):
            contexts = {}
        existing = contexts.get(report_type_id, {})
        if not isinstance(existing, dict):
            existing = {}
        existing.update(payload)
        contexts[report_type_id] = existing
        session["report_type_review_contexts"] = contexts
        session.modified = True

    def _persist_sample_source_in_manifest(
        report_type_id: str,
        *,
        plugin_root: Path,
        source_path: str,
        sheet_name: str = "",
    ) -> None:
        """Write sample_source (and sheet) into manifest extensions so it survives session loss."""
        if not source_path:
            return
        manifest_path = plugin_root / report_type_id / "manifest.yaml"
        if not manifest_path.exists():
            return
        manifest = _load_plugin_manifest(report_type_id, plugin_root)
        if not manifest:
            return
        extensions = manifest.get("extensions") or {}
        extensions["sample_source"] = source_path
        if sheet_name:
            extensions["sample_sheet"] = sheet_name
        elif "sample_sheet" in extensions:
            del extensions["sample_sheet"]
        manifest["extensions"] = extensions
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

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

        plugin_root = Path(app.config["PLUGIN_ROOT"])
        manifest_path = plugin_root / report_type_id / "manifest.yaml"
        plugin_path = plugin_root / report_type_id / "plugin.py"
        smoke_test_path = plugin_root / report_type_id / "tests" / "test_smoke.py"
        manifest = _load_plugin_manifest(report_type_id, plugin_root)
        review_context = _report_type_review_context(report_type_id)
        sample_source_path = str(review_context.get("existing_csv_path", "")).strip()
        sample_sheet_name = str(review_context.get("sheet_name", "")).strip()
        if not sample_source_path:
            _ext = (manifest or {}).get("extensions") or {}
            sample_source_path = str(_ext.get("sample_source", "")).strip()
            sample_sheet_name = sample_sheet_name or str(_ext.get("sample_sheet", "")).strip()
        raw_draft_workflow = str(review_context.get("draft_workflow", "")).strip()
        draft_workflow: list[dict[str, Any]] = []
        if raw_draft_workflow:
            try:
                parsed_workflow = json.loads(raw_draft_workflow)
                if isinstance(parsed_workflow, list):
                    draft_workflow = [item for item in parsed_workflow if isinstance(item, dict)]
            except json.JSONDecodeError:
                draft_workflow = []
        has_sample_source = bool(sample_source_path and Path(sample_source_path).exists())
        manifest_status = str(manifest.get("status", "unknown")).strip() or "unknown"
        yaml_payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        required_columns = [
            str(item).strip()
            for item in (yaml_payload.get("required_columns", []) if isinstance(yaml_payload, dict) else [])
            if str(item).strip()
        ]
        source_summary = _safe_describe_source(sample_source_path, sheet_name=sample_sheet_name or None) if has_sample_source else None
        source_columns = [
            str(item).strip()
            for item in ((source_summary or {}).get("columns", []) if isinstance(source_summary, dict) else [])
            if str(item).strip()
        ]
        missing_source_columns = [col for col in required_columns if col not in set(source_columns)]
        source_matches_required_columns = has_sample_source and not missing_source_columns
        source_preflight = _safe_preflight_source(
            sample_source_path,
            sheet_name=sample_sheet_name or None,
            required_columns=required_columns,
        ) if has_sample_source else None
        source_preflight_issues = [
            str(item).strip() for item in ((source_preflight or {}).get("issues", []) if isinstance(source_preflight, dict) else []) if str(item).strip()
        ]
        source_preflight_warnings = [
            str(item).strip() for item in ((source_preflight or {}).get("warnings", []) if isinstance(source_preflight, dict) else []) if str(item).strip()
        ]

        # Retrieve preview artifact paths stored by the generate route
        preview_html = str(review_context.get("preview_html_path", "")).strip()
        preview_json = str(review_context.get("preview_json_path", "")).strip()
        # Only show if the files actually exist
        if preview_html and not Path(preview_html).exists():
            preview_html = ""
        if preview_json and not Path(preview_json).exists():
            preview_json = ""

        # Fallback: scan output directory for latest report artifacts
        if not preview_html or not preview_json:
            output_dir = Path(app.config["OUTPUT_FOLDER"]) / report_type_id
            if output_dir.is_dir():
                latest_html = output_dir / f"{report_type_id}.report.html"
                latest_json = output_dir / f"{report_type_id}.report.json"
                if not preview_html and latest_html.exists():
                    preview_html = str(latest_html)
                if not preview_json and latest_json.exists():
                    preview_json = str(latest_json)

        return render_template(
            "report_type_yaml_view.html",
            report_type_id=report_type_id,
            report_type_label=_friendly_report_type_label(report_type_id),
            yaml_text=path.read_text(encoding="utf-8"),
            yaml_path=str(path),
            manifest_text=manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else "",
            manifest_path=str(manifest_path),
            plugin_text=plugin_path.read_text(encoding="utf-8") if plugin_path.exists() else "",
            plugin_path=str(plugin_path),
            smoke_test_text=smoke_test_path.read_text(encoding="utf-8") if smoke_test_path.exists() else "",
            smoke_test_path=str(smoke_test_path),
            manifest_status=manifest_status,
            sample_source_path=sample_source_path,
            sample_sheet_name=sample_sheet_name,
            has_sample_source=has_sample_source,
            can_generate_sample=(
                has_sample_source
                and manifest_status.lower() in {"draft", "active", "planned"}
                and source_matches_required_columns
                and not source_preflight_issues
            ),
            required_columns=required_columns,
            source_columns=source_columns,
            missing_source_columns=missing_source_columns,
            source_matches_required_columns=source_matches_required_columns,
            source_preflight=source_preflight,
            source_preflight_issues=source_preflight_issues,
            source_preflight_warnings=source_preflight_warnings,
            draft_workflow=draft_workflow,
            preview_html_path=preview_html,
            preview_json_path=preview_json,
        )

    @app.post("/report-types/save-file")
    def save_report_type_file() -> Any:
        report_type_id = request.form.get("report_type_id", "").strip()
        file_key = request.form.get("file_key", "").strip()   # yaml | manifest | plugin | smoke_test
        content = request.form.get("content", "")

        if not report_type_id or not re.fullmatch(r"[a-z0-9_]+", report_type_id):
            return jsonify({"ok": False, "error": "Invalid report_type_id."}), 400

        allowed_keys = {"yaml", "manifest", "plugin", "smoke_test"}
        if file_key not in allowed_keys:
            return jsonify({"ok": False, "error": f"Unknown file_key '{file_key}'."}), 400

        config_dir = Path(app.config["REPORT_TYPES_DIR"])
        plugin_root = Path(app.config["PLUGIN_ROOT"])

        file_map: dict[str, Path] = {
            "yaml":       config_dir / f"{report_type_id}.yaml",
            "manifest":   plugin_root / report_type_id / "manifest.yaml",
            "plugin":     plugin_root / report_type_id / "plugin.py",
            "smoke_test": plugin_root / report_type_id / "tests" / "test_smoke.py",
        }
        target = file_map[file_key]

        if not target.exists():
            return jsonify({"ok": False, "error": f"File not found: {target}"}), 404

        # Block edits to protected report types
        if report_type_id in PROTECTED_REPORT_TYPES:
            return jsonify({"ok": False, "error": f"'{report_type_id}' is protected and cannot be edited via UI."}), 403

        target.write_text(content, encoding="utf-8")
        return jsonify({"ok": True})

    @app.post("/report-types/mark-planned")
    def mark_report_type_planned() -> Any:
        report_type_id = request.form.get("report_type_id", "").strip()
        if not report_type_id or not re.fullmatch(r"[a-z0-9_]+", report_type_id):
            flash("Invalid report_type_id.", "danger")
            return redirect(url_for("list_report_types_page"))
        if report_type_id in PROTECTED_REPORT_TYPES:
            flash(f"'{report_type_id}' is protected.", "danger")
            return redirect(url_for("view_report_type_yaml_page", report_type_id=report_type_id))

        plugin_root = Path(app.config["PLUGIN_ROOT"])
        manifest_path = plugin_root / report_type_id / "manifest.yaml"
        manifest = _load_plugin_manifest(report_type_id, plugin_root)
        if not manifest or not manifest_path.exists():
            flash(f"Plugin manifest not found for '{report_type_id}'.", "danger")
            return redirect(url_for("view_report_type_yaml_page", report_type_id=report_type_id))

        manifest["status"] = "planned"
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        flash(f"'{report_type_id}' marked as planned. Generate a sample report to preview, then publish when ready.", "success")
        return redirect(url_for("view_report_type_yaml_page", report_type_id=report_type_id))

    @app.post("/report-types/preview-and-mark-planned")
    def preview_and_mark_planned() -> Any:
        """Mark report type as planned, then redirect to generate preview sample."""
        report_type_id = request.form.get("report_type_id", "").strip()
        if not report_type_id or not re.fullmatch(r"[a-z0-9_]+", report_type_id):
            flash("Invalid report_type_id.", "danger")
            return redirect(url_for("list_report_types_page"))
        if report_type_id in PROTECTED_REPORT_TYPES:
            flash(f"'{report_type_id}' is protected.", "danger")
            return redirect(url_for("view_report_type_yaml_page", report_type_id=report_type_id))

        plugin_root = Path(app.config["PLUGIN_ROOT"])
        manifest_path = plugin_root / report_type_id / "manifest.yaml"
        manifest = _load_plugin_manifest(report_type_id, plugin_root)
        if not manifest or not manifest_path.exists():
            flash(f"Plugin manifest not found for '{report_type_id}'.", "danger")
            return redirect(url_for("view_report_type_yaml_page", report_type_id=report_type_id))

        # Mark as planned if not already
        current_status = str(manifest.get("status", "")).strip().lower()
        if current_status == "draft":
            manifest["status"] = "planned"
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

        # 307 preserves the POST method and body so /generate receives form fields correctly
        return redirect(url_for("generate"), 307)

    @app.post("/report-types/publish")
    def publish_report_type() -> Any:
        report_type_id = request.form.get("report_type_id", "").strip()
        if not report_type_id or not re.fullmatch(r"[a-z0-9_]+", report_type_id):
            flash("Invalid report_type_id.", "danger")
            return redirect(url_for("list_report_types_page"))

        plugin_root = Path(app.config["PLUGIN_ROOT"])
        manifest_path = plugin_root / report_type_id / "manifest.yaml"
        manifest = _load_plugin_manifest(report_type_id, plugin_root)
        if not manifest or not manifest_path.exists():
            flash(f"Plugin manifest not found for '{report_type_id}'.", "danger")
            return redirect(url_for("view_report_type_yaml_page", report_type_id=report_type_id))

        current_status = str(manifest.get("status", "")).strip().lower()
        if current_status != "active":
            _write_canonical_smoke_test(report_type_id, plugin_root)
            # Run smoke test before publishing
            test_success, test_message = _run_smoke_test(report_type_id, plugin_root)
            if not test_success:
                flash(f"Cannot publish: {test_message}", "danger")
                return redirect(url_for("view_report_type_yaml_page", report_type_id=report_type_id))
            
            now_utc = datetime.now(timezone.utc).isoformat()
            manifest["status"] = "active"
            manifest.setdefault("published_at", now_utc)
            manifest["updated_at"] = now_utc
            manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
            flash(f"Published report type '{report_type_id}'. It is now available for generation.", "success")
        else:
            flash(f"Report type '{report_type_id}' is already published.", "info")
        return redirect(url_for("view_report_type_yaml_page", report_type_id=report_type_id))

    @app.post("/report-types/rename")
    def rename_report_type() -> Any:
        current_id = request.form.get("current_report_type_id", "").strip()
        raw_new_id = request.form.get("new_report_type_id", "").strip()
        new_id = _normalize_report_type_id(raw_new_id)
        
        # Validate both IDs
        if not current_id or not re.fullmatch(r"[a-z0-9_]+", current_id):
            flash("Invalid current report_type_id.", "danger")
            return redirect(url_for("list_report_types_page"))
        
        if not new_id or not re.fullmatch(r"[a-z0-9_]+", new_id):
            flash("Invalid new report_type_id. Use letters, digits, spaces, hyphens, or underscores.", "danger")
            return redirect(url_for("view_report_type_yaml_page", report_type_id=current_id))
        
        if current_id == new_id:
            flash("New ID is the same as current ID.", "info")
            return redirect(url_for("view_report_type_yaml_page", report_type_id=current_id))
        
        config_dir = Path(app.config["REPORT_TYPES_DIR"])
        plugin_root = Path(app.config["PLUGIN_ROOT"])
        
        # Check source files exist
        curr_yaml = config_dir / f"{current_id}.yaml"
        curr_plugin_dir = plugin_root / current_id
        
        if not curr_yaml.exists() or not curr_plugin_dir.exists():
            flash(f"Report type '{current_id}' not found.", "danger")
            return redirect(url_for("list_report_types_page"))
        
        # Check target doesn't already exist
        new_yaml = config_dir / f"{new_id}.yaml"
        new_plugin_dir = plugin_root / new_id
        
        if new_yaml.exists() or new_plugin_dir.exists():
            flash(f"Report type '{new_id}' already exists.", "danger")
            return redirect(url_for("view_report_type_yaml_page", report_type_id=current_id))
        
        try:
            # Rename YAML config file
            curr_yaml.rename(new_yaml)
            
            # Rename plugin directory
            shutil.move(str(curr_plugin_dir), str(new_plugin_dir))
            _rewrite_renamed_report_type_files(
                old_id=current_id,
                new_id=new_id,
                yaml_path=new_yaml,
                plugin_dir=new_plugin_dir,
            )
            
            # Move session context to new ID
            review_contexts = session.get("report_type_review_contexts", {})
            if not isinstance(review_contexts, dict):
                review_contexts = {}
            if current_id in review_contexts:
                review_contexts[new_id] = review_contexts.pop(current_id)
                session["report_type_review_contexts"] = review_contexts
                session.modified = True
            
            if raw_new_id != new_id:
                flash(f"Renamed report type from '{current_id}' to '{new_id}' (normalized from '{raw_new_id}').", "success")
            else:
                flash(f"Renamed report type from '{current_id}' to '{new_id}'.", "success")
            return redirect(url_for("view_report_type_yaml_page", report_type_id=new_id))
        
        except Exception as e:  # noqa: BLE001
            flash(f"Error renaming report type: {str(e)}", "danger")
            return redirect(url_for("view_report_type_yaml_page", report_type_id=current_id))

    @app.get("/report-types")
    def list_report_types_page() -> str:
        registry = ReportTypeRegistry(config_dir=Path(app.config["REPORT_TYPES_DIR"]))
        items = []
        for report_type_id in registry.list_report_types():
            path = Path(app.config["REPORT_TYPES_DIR"]) / f"{report_type_id}.yaml"
            manifest = _load_plugin_manifest(report_type_id, Path(app.config["PLUGIN_ROOT"]))
            published_raw = str(manifest.get("published_at", "")).strip()
            updated_raw = str(manifest.get("updated_at", "")).strip()
            published_at = _display_local_time(_iso_to_utc(published_raw)) if published_raw else "-"
            updated_at = _display_local_time(_iso_to_utc(updated_raw)) if updated_raw else "-"
            items.append(
                {
                    "report_type_id": report_type_id,
                    "path": str(path),
                    "is_protected": report_type_id in PROTECTED_REPORT_TYPES,
                    "published_at": published_at,
                    "updated_at": updated_at,
                }
            )
        return render_template("report_types.html", report_types=items)

    @app.post("/report-types/new")
    def create_report_type() -> Any:
        prompt_text = request.form.get("prompt_text", "").strip()
        clone_from = request.form.get("clone_source_type", "").strip()
        sheet_name = request.form.get("sheet_name", "").strip()
        hint_domain = request.form.get("hint_domain", "").strip()
        hint_family = request.form.get("hint_family", "").strip()
        hint_mode = request.form.get("hint_mode", "").strip()
        source_sample_percent_raw = request.form.get("source_sample_percent", "100").strip()
        uploaded_files = [f for f in request.files.getlist("csv_upload") if f and f.filename]
        existing_csv_path = request.form.get("existing_csv_path", "").strip()
        registry = ReportTypeRegistry(config_dir=Path(app.config["REPORT_TYPES_DIR"]))
        source_path_for_review = ""
        draft_workflow: list[dict[str, Any]] = []
        try:
            if not prompt_text:
                raise ValueError("Describe the report type you want the AI to draft.")
            available_types = registry.list_report_types()
            if clone_from and clone_from not in available_types:
                raise ValueError(f"Unknown clone source '{clone_from}'.")
            source_sample_percent = int(source_sample_percent_raw or "100")
            if source_sample_percent not in {10, 20, 30, 40, 50, 60, 70, 80, 90, 100}:
                raise ValueError("Source sampling percentage must be one of: 10, 20, 30, ..., 100.")

            draft_workflow.append(
                {
                    "agent": "report_type_request",
                    "status": "completed",
                    "summary": "Captured the natural-language request for a new report type.",
                    "details": {
                        "prompt_text": prompt_text,
                        "clone_from": clone_from,
                        "hint_domain": hint_domain,
                        "hint_family": hint_family,
                        "hint_mode": hint_mode,
                        "source_sample_percent": source_sample_percent,
                    },
                }
            )

            clone_yaml = ""
            if clone_from:
                clone_yaml = (Path(app.config["REPORT_TYPES_DIR"]) / f"{clone_from}.yaml").read_text(encoding="utf-8")
                draft_workflow.append(
                    {
                        "agent": "report_type_clone_context",
                        "status": "completed",
                        "summary": f"Loaded '{clone_from}' as a draft starting point.",
                        "details": {"clone_from": clone_from},
                    }
                )

            source_profile = None
            resolved_existing_csv_path = existing_csv_path
            sheet_options: list[str] = []
            if uploaded_files or existing_csv_path:
                csv_paths = _resolve_csv_paths(
                    uploaded_files,
                    Path(app.config["UPLOAD_FOLDER"]),
                    existing_csv_path=existing_csv_path,
                )
                if len(csv_paths) != 1:
                    raise ValueError("Select exactly one CSV or Excel file for AI draft classification.")
                source_path = csv_paths[0]
                source_path_for_review = source_path
                resolved_existing_csv_path = source_path
                if Path(source_path).suffix.lower() in {".xlsx", ".xls"} and not sheet_name:
                    sheet_options = list_excel_sheets(source_path)
                    if len(sheet_options) > 1:
                        flash("Excel file has multiple sheets. Choose a sheet and submit again.", "warning")
                        return render_template(
                            "new_report_type.html",
                            report_types=registry.list_report_types(),
                            prompt_text=prompt_text,
                            selected_clone_type=clone_from,
                            report_type_agent_model=app.config["REPORT_TYPE_AGENT_MODEL"],
                            recent_uploads=_recent_uploaded_sources(Path(app.config["UPLOAD_FOLDER"])),
                            existing_csv_path=resolved_existing_csv_path,
                            selected_sheet="",
                            sheet_options=sheet_options,
                            source_summary=describe_tabular_source(source_path),
                            source_sample_percent=source_sample_percent,
                        )
                    if len(sheet_options) == 1:
                        sheet_name = sheet_options[0]
                source_profile = _build_report_type_source_profile(
                    source_path,
                    sheet_name=sheet_name or None,
                    sample_percent=source_sample_percent,
                )
                draft_workflow.append(
                    {
                        "agent": "report_type_source_profiler",
                        "status": "completed",
                        "summary": "Profiled the sample source to constrain required columns and plugin design.",
                        "details": {
                            "source_path": source_path,
                            "sheet_name": sheet_name,
                            "columns": ((source_profile or {}).get("source_metadata") or {}).get("columns", []),
                            "sampling": (source_profile or {}).get("sampling", {}),
                        },
                    }
                )

            draft = _generate_report_type_agent_draft(
                prompt_text=prompt_text,
                clone_from=clone_from or None,
                clone_yaml=clone_yaml,
                report_types=available_types,
                model=str(app.config["REPORT_TYPE_AGENT_MODEL"]),
                source_profile=source_profile,
                hint_domain=hint_domain or None,
                hint_family=hint_family or None,
                hint_mode=hint_mode or None,
            )
            draft_workflow.append(
                {
                    "agent": "report_type_drafter",
                    "status": "completed",
                    "summary": "Generated the draft YAML, plugin, and smoke test for the new report type.",
                    "details": {
                        "report_type_id": draft.get("report_type_id", ""),
                        "family": draft.get("family", ""),
                        "domain": draft.get("domain", ""),
                        "mode": draft.get("mode", ""),
                        "required_columns": draft.get("required_columns", []),
                    },
                }
            )
            report_type_id = _materialize_report_type_agent_draft(
                draft=draft,
                config_dir=Path(app.config["REPORT_TYPES_DIR"]),
                plugin_root=Path(app.config["PLUGIN_ROOT"]),
                clone_from=clone_from or None,
                source_profile=source_profile,
            )
            draft_workflow.append(
                {
                    "agent": "report_type_materializer",
                    "status": "completed",
                    "summary": "Wrote the draft report type files into the workspace for review and publishing.",
                    "details": {"report_type_id": report_type_id},
                }
            )
        except Exception as exc:  # noqa: BLE001
            flash(f"Failed to generate AI draft: {exc}", "danger")
            return render_template(
                "new_report_type.html",
                report_types=registry.list_report_types(),
                prompt_text=prompt_text,
                selected_clone_type=clone_from,
                report_type_agent_model=app.config["REPORT_TYPE_AGENT_MODEL"],
                recent_uploads=_recent_uploaded_sources(Path(app.config["UPLOAD_FOLDER"])),
                existing_csv_path=existing_csv_path,
                selected_sheet=sheet_name,
                sheet_options=list_excel_sheets(existing_csv_path) if existing_csv_path and Path(existing_csv_path).exists() and Path(existing_csv_path).suffix.lower() in {".xlsx", ".xls"} else [],
                source_summary=_safe_describe_source(existing_csv_path, sheet_name=sheet_name or None),
                source_sample_percent=int(source_sample_percent_raw or "100"),
                families=sorted(FAMILIES),
                domains=sorted(DOMAINS),
                modes=sorted(MODES),
            )

        _store_report_type_review_context(
            report_type_id,
            source_path=source_path_for_review,
            sheet_name=sheet_name,
            draft_workflow=draft_workflow,
        )
        _persist_sample_source_in_manifest(
            report_type_id,
            plugin_root=Path(app.config["PLUGIN_ROOT"]),
            source_path=source_path_for_review,
            sheet_name=sheet_name,
        )
        flash(
            f"Generated AI draft for report type '{report_type_id}'. Review it, publish it, and generate a sample report when ready.",
            "success",
        )
        return redirect(url_for("view_report_type_yaml_page", report_type_id=report_type_id))

    @app.post("/report-types/delete")
    def delete_report_type() -> Any:
        is_xhr = request.headers.get("X-Requested-With") == "XMLHttpRequest"
        report_type_id = request.form.get("report_type_id", "").strip()
        if not report_type_id:
            if is_xhr:
                return {"ok": False, "error": "Missing report_type_id."}, 400
            flash("Missing report_type_id.", "danger")
            return redirect(url_for("list_report_types_page"))
        if report_type_id in PROTECTED_REPORT_TYPES:
            if is_xhr:
                return {"ok": False, "error": f"'{report_type_id}' is protected and cannot be removed from UI."}, 403
            flash(f"'{report_type_id}' is protected and cannot be removed from UI.", "danger")
            return redirect(url_for("list_report_types_page"))

        path = Path(app.config["REPORT_TYPES_DIR"]) / f"{report_type_id}.yaml"
        if not path.exists():
            if is_xhr:
                return {"ok": False, "error": f"Report type not found: {report_type_id}"}, 404
            flash(f"Report type not found: {report_type_id}", "danger")
            return redirect(url_for("list_report_types_page"))
        path.unlink()
        plugin_dir = Path(app.config["PLUGIN_ROOT"]) / report_type_id
        if plugin_dir.exists() and plugin_dir.is_dir():
            import shutil
            shutil.rmtree(plugin_dir, ignore_errors=True)
        if is_xhr:
            return {"ok": True}, 200
        flash(f"Deleted report type '{report_type_id}'.", "success")
        return redirect(url_for("list_report_types_page"))

    @app.post("/generate")
    def generate() -> Any:
        attempt_id = request.form.get("attempt_id", "").strip() or f"evt_{uuid4().hex[:10]}"
        report_type_id = request.form.get("report_type_id", "").strip()
        return_to_report_type_view = request.form.get("return_to_report_type_view", "").strip() == "1"

        def failure_redirect() -> Any:
            if return_to_report_type_view and report_type_id:
                return redirect(url_for("view_report_type_yaml_page", report_type_id=report_type_id))
            return redirect(url_for("index"))

        registry = ReportTypeRegistry(config_dir=Path(app.config["REPORT_TYPES_DIR"]))
        visible_report_types = _visible_report_types_for_generation(
            registry.list_report_types(),
            Path(app.config["PLUGIN_ROOT"]),
        )
        if report_type_id not in visible_report_types:
            flash(f"Report type '{report_type_id}' is not enabled for generation.", "danger")
            return failure_redirect()
        definition = registry.get(report_type_id)
        provider_name = request.form.get("provider", "local").strip().lower()
        if provider_name == "mock":
            provider_name = "local"
        if provider_name not in PROVIDER_CATALOG:
            flash(f"Unknown provider '{provider_name}'.", "danger")
            return failure_redirect()
        model = request.form.get("model", "gpt-5-mini").strip()
        api_key = request.form.get("api_key", "").strip()
        api_base_url = request.form.get("api_base_url", "").strip()
        confirm_cost = request.form.get("confirm_cost", "0") == "1" or request.form.get("confirm_openai", "0") == "1"
        expected_cost_token = request.form.get("expected_cost_token", "").strip()
        sheet_name = request.form.get("sheet_name", "").strip()
        uploaded_files = [f for f in request.files.getlist("csv_upload") if f and f.filename]
        existing_csv_path = request.form.get("existing_csv_path", "").strip()
        source_labels_text = request.form.get("source_labels", "").strip()
        source_labels = _parse_source_labels_text(source_labels_text)
        output_token_budget_raw = request.form.get("output_token_budget", "").strip()
        output_token_budget = int(output_token_budget_raw) if output_token_budget_raw else None
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

        prefs = _report_type_runtime_default_prefs(
            report_type_id=report_type_id,
            definition=definition,
            plugin_root=Path(app.config["PLUGIN_ROOT"]),
        )
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
            csv_paths = _resolve_csv_paths(
                uploaded_files,
                Path(app.config["UPLOAD_FOLDER"]),
                existing_csv_path=existing_csv_path,
            )
            csv_path, source_files_display, was_combined_source = _prepare_pipeline_source(
                csv_paths=csv_paths,
                report_type_id=report_type_id,
                row_limit=row_limit,
                sheet_name=sheet_name or None,
                upload_dir=Path(app.config["UPLOAD_FOLDER"]),
                source_labels=source_labels,
            )
            pipeline_row_limit = None if was_combined_source else row_limit
            if return_to_report_type_view and report_type_id and not was_combined_source:
                _store_report_type_review_context(
                    report_type_id,
                    source_path=csv_path,
                    sheet_name=sheet_name or "",
                )
                _persist_sample_source_in_manifest(
                    report_type_id,
                    plugin_root=Path(app.config["PLUGIN_ROOT"]),
                    source_path=csv_path,
                    sheet_name=sheet_name or "",
                )
            if Path(csv_path).suffix.lower() in {".xlsx", ".xls"} and not sheet_name:
                sheets = list_excel_sheets(csv_path)
                if len(sheets) > 1:
                    flash("Excel file has multiple sheets. Choose a sheet and submit again.", "warning")
                    report_types = _visible_report_types_for_generation(
                        registry.list_report_types(),
                        Path(app.config["PLUGIN_ROOT"]),
                    )
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
                            "row_limit": row_limit,
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

            source_preflight = preflight_tabular_source(
                csv_path,
                sheet_name=sheet_name or None,
                required_columns=getattr(definition, "required_columns", []),
                row_limit=pipeline_row_limit,
            )
            if not bool(source_preflight.get("ok")):
                issues = [
                    str(item).strip()
                    for item in source_preflight.get("issues", [])
                    if str(item).strip()
                ]
                raise ValueError("Source preflight failed: " + ("; ".join(issues) if issues else "Unable to safely parse the selected source."))

            definition, effective_prefs, csv_profile, metrics = prepare_pipeline_inputs(
                csv_path=csv_path,
                report_type_id=report_type_id,
                user_prefs=prefs,
                registry=registry,
                row_limit=pipeline_row_limit,
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
                            "message": (
                                f"Estimated cost ${float(estimate.get('total_cost_usd_est', 0.0)):.6f}"
                                if estimate.get("total_cost_usd_est") is not None
                                else "Estimated cost is unbounded because output budget is unlimited."
                            ),
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
                        csv_source_label=_format_source_label(source_files_display, sheet_name=sheet_name),
                        rows_used=csv_profile.get("row_count", 0),
                        provider=provider_name,
                        model=model,
                        csv_path=csv_path,
                        sheet_name=sheet_name,
                        row_limit=row_limit_raw,
                        output_token_budget=output_token_budget_raw,
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
                    return failure_redirect()
                generation_cost_usd_est = (
                    float(fresh_estimate.get("total_cost_usd_est", 0.0))
                    if fresh_estimate.get("total_cost_usd_est") is not None
                    else None
                )
                generation_input_tokens_est = int(fresh_estimate.get("input_tokens_est", 0) or 0)
                generation_output_tokens_est = (
                    int(fresh_estimate.get("output_tokens_est", 0) or 0)
                    if fresh_estimate.get("output_tokens_est") is not None
                    else None
                )
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
                return failure_redirect()

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
                "source_csv": " | ".join(source_files_display),
                "source_csv_list": source_files_display,
                "source_labels": source_labels,
                "source_sheet": sheet_name,
                "source_rows_used": csv_profile.get("row_count"),
                "tone": str(effective_prefs.get("tone", "")).strip(),
                "audience": str(effective_prefs.get("audience", "")).strip(),
                "focus": str(effective_prefs.get("focus", "")).strip(),
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
                row_limit=pipeline_row_limit,
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
            return failure_redirect()
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
        if return_to_report_type_view and report_type_id:
            # Store preview artifact paths in session for the YAML view page
            contexts = session.get("report_type_review_contexts", {})
            if not isinstance(contexts, dict):
                contexts = {}
            ctx = contexts.get(report_type_id, {})
            if not isinstance(ctx, dict):
                ctx = {}
            ctx["preview_html_path"] = str(report_html_path) if report_html_path else ""
            ctx["preview_json_path"] = str(report_json_path) if report_json_path else ""
            contexts[report_type_id] = ctx
            session["report_type_review_contexts"] = contexts
            session.modified = True
            return redirect(url_for("view_report_type_yaml_page", report_type_id=report_type_id))
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
        report_type_ids = {
            str(i.get("report_type_id", "")).strip() for i in items if str(i.get("report_type_id", "")).strip()
        }
        report_type_options = [
            {"value": rid, "label": _friendly_report_type_label(rid)}
            for rid in sorted({_canonical_report_type_id(v) for v in report_type_ids})
        ]
        report_type_filter = _resolve_report_type_filter_value(
            request.args.get("report_type_id", "").strip(),
            report_type_ids,
        )
        selected_paths = [p.strip() for p in request.args.getlist("json_path") if p.strip()]
        if len(selected_paths) > 4:
            flash("Benchmark view supports up to 4 reports at once. Showing first 4.", "warning")
            selected_paths = selected_paths[:4]

        filtered_items = items
        if report_type_filter:
            filtered_items = [
                i
                for i in items
                if _canonical_report_type_id(str(i.get("report_type_id", "")).strip()) == report_type_filter
            ]

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
        ptp_benchmark: dict[str, Any] = {}
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
                tones = [str((r.get("metadata", {}) or {}).get("tone", "")).strip().lower() for r in selected_reports]
                audiences = [str((r.get("metadata", {}) or {}).get("audience", "")).strip().lower() for r in selected_reports]
                focuses = [str((r.get("metadata", {}) or {}).get("focus", "")).strip().lower() for r in selected_reports]
                row_limits = [_safe_float(r.get("source_rows_used_raw")) for r in selected_reports]
                model_ratios = [_model_cost_ratio_vs_baseline(str(r.get("model", "") or "")) for r in selected_reports]

                same_source_file = len({v for v in source_files if v}) == 1 and bool(source_files and source_files[0])
                same_report_type = len({v for v in report_types if v}) == 1 and bool(report_types and report_types[0])
                same_tone = len({v for v in tones if v}) == 1 and bool(tones and tones[0])
                same_audience = len({v for v in audiences if v}) == 1 and bool(audiences and audiences[0])
                same_focus = len({v for v in focuses if v}) == 1 and bool(focuses and focuses[0])
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
                c3_status, c3_score = _baseline_check_status(same_tone, False)
                c4_status, c4_score = _baseline_check_status(same_audience, False)
                c5_status, c5_score = _baseline_check_status(same_focus, False)
                c6_status, c6_score = _baseline_check_status(same_model_strength, almost_model_strength, weak_model_strength)
                c7_status, c7_score = _baseline_check_status(same_input_rows, almost_input_rows)

                checks = [
                    {"key": "same_source_file", "label": "Same source data file", "status": c1_status, "score": c1_score},
                    {"key": "same_report_type", "label": "Same report type", "status": c2_status, "score": c2_score},
                    {"key": "same_tone", "label": "Same tone", "status": c3_status, "score": c3_score},
                    {"key": "same_audience", "label": "Same audience", "status": c4_status, "score": c4_score},
                    {"key": "same_focus", "label": "Same focus", "status": c5_status, "score": c5_score},
                    {"key": "same_model_strength", "label": "Same model strength (1.x ratio)", "status": c6_status, "score": c6_score},
                    {"key": "same_input_rows", "label": "Same input rows used", "status": c7_status, "score": c7_score},
                ]
                total_score = c1_score + c2_score + c3_score + c4_score + c5_score + c6_score + c7_score
                if total_score >= 18:
                    baseline_level = "strong"
                elif total_score >= 12:
                    baseline_level = "moderate"
                else:
                    baseline_level = "weak"

                baseline_assessment = {
                    "supported": True,
                    "selected_count": selected_count,
                    "checks": checks,
                    "baseline_level": baseline_level,
                    "score": total_score,
                    "score_max": 21,
                    "model_ratios": [round(float(r), 2) if isinstance(r, (int, float)) else None for r in model_ratios],
                    "sources": source_files,
                    "sources_display": _compact_values_display(source_files),
                    "report_types": report_types,
                    "report_types_display": _compact_values_display(report_types),
                    "tones": tones,
                    "tones_display": _compact_values_display(tones),
                    "audiences": audiences,
                    "audiences_display": _compact_values_display(audiences),
                    "focuses": focuses,
                    "focuses_display": _compact_values_display(focuses),
                    "rows": [int(v) if v > 0 else None for v in row_limits],
                    "rows_display": _compact_values_display([int(v) if v > 0 else None for v in row_limits], none_text="-"),
                }
            else:
                baseline_assessment = {
                    "supported": False,
                    "selected_count": selected_count,
                    "message": "Baseline criteria is evaluated only when exactly 2 or 3 reports are selected.",
                }
            ptp_benchmark = _build_wireshark_ptp_benchmark(selected_reports)

        return render_template(
            "benchmark.html",
            output_folder=str(root),
            reports=filtered_items,
            selected_reports=selected_reports,
            report_type_options=report_type_options,
            report_type_filter=report_type_filter,
            selected_paths=selected_paths,
            analytics=analytics,
            comparison_rows=comparison_rows,
            similarity_rows=similarity_rows,
            comparison_brief=comparison_brief,
            baseline_assessment=baseline_assessment,
            ptp_benchmark=ptp_benchmark,
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


def _load_plugin_manifest(report_type_id: str, plugin_root: Path) -> dict[str, Any]:
    manifest_path = plugin_root / report_type_id / "manifest.yaml"
    if not manifest_path.exists():
        return {}
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return {}
    return manifest if isinstance(manifest, dict) else {}


def _normalize_report_type_id(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _rewrite_renamed_report_type_files(*, old_id: str, new_id: str, yaml_path: Path, plugin_dir: Path) -> None:
    if yaml_path.exists():
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        if isinstance(payload, dict):
            payload["report_type_id"] = new_id
            if str(payload.get("metrics_profile", "")).strip() == old_id:
                payload["metrics_profile"] = new_id
            yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    manifest_path = plugin_dir / "manifest.yaml"
    if manifest_path.exists():
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        if isinstance(manifest, dict):
            manifest["plugin_id"] = new_id
            if str(manifest.get("metrics_profile", "")).strip() == old_id:
                manifest["metrics_profile"] = new_id
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    for relative_path in (Path("plugin.py"), Path("tests") / "test_smoke.py"):
        path = plugin_dir / relative_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace(old_id, new_id), encoding="utf-8")


def _canonical_smoke_test_template(*, report_type_id: str) -> str:
    return f'''from __future__ import annotations

import pandas as pd

from rv_reporter.report_types.plugins import ReportPluginManager


def test_{report_type_id}_smoke_build() -> None:
    manager = ReportPluginManager(plugin_dir="report_type_plugins")
    result = manager.compute_metrics(
        metrics_profile="{report_type_id}",
        df=pd.DataFrame({{"sample": [1, 2, 3]}}),
        prefs={{}},
        report_type_id="{report_type_id}",
    )
    assert isinstance(result, dict)
'''


def _write_canonical_smoke_test(report_type_id: str, plugin_root: Path) -> None:
    smoke_test_path = plugin_root / report_type_id / "tests" / "test_smoke.py"
    # Preserve agent-generated smoke tests that already use the correct imports;
    # overwrite only if the file is missing or has broken / stale imports.
    if smoke_test_path.exists():
        existing = smoke_test_path.read_text(encoding="utf-8")
        if "from rv_reporter.report_types.plugins import ReportPluginManager" in existing:
            return  # valid agent-generated test — keep it
    smoke_test_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_test_path.write_text(_canonical_smoke_test_template(report_type_id=report_type_id), encoding="utf-8")


def _run_smoke_test(report_type_id: str, plugin_root: Path) -> tuple[bool, str]:
    """Run smoke test for a plugin. Returns (success, output_message)."""
    smoke_test_path = plugin_root / report_type_id / "tests" / "test_smoke.py"
    
    if not smoke_test_path.exists():
        return True, "No smoke test found (skipped)."
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(smoke_test_path), "-xvs"],
            cwd=str(plugin_root.parent),
            capture_output=True,
            timeout=30,
            text=True,
        )
        if result.returncode == 0:
            return True, "Smoke test passed."
        else:
            return False, f"Smoke test failed:\n{result.stdout}\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "Smoke test exceeded 30 second timeout."
    except Exception as e:  # noqa: BLE001
        return False, f"Error running smoke test: {str(e)}"


def _visible_report_types_for_generation(all_report_types: list[str], plugin_root: Path | None = None) -> list[str]:
    hidden_raw = (os.getenv("GENERATION_HIDDEN_REPORT_TYPES", "") or "").strip()
    hidden_set = {v.strip() for v in hidden_raw.split(",") if v.strip()}

    # draft = no local preview yet; planned/active = allowed for generation
    draft_only_types: set[str] = set()
    if plugin_root is not None:
        for report_type_id in all_report_types:
            manifest = _load_plugin_manifest(report_type_id, plugin_root)
            if manifest and str(manifest.get("status", "")).strip().lower() == "draft":
                draft_only_types.add(report_type_id)

    return [rid for rid in all_report_types if rid not in hidden_set and rid not in draft_only_types]


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
    
def _provider_pricing_url(provider_name: str) -> str:
    return str(PROVIDER_PRICING_INFO.get(provider_name, {}).get("pricing_url", PRICING_SOURCE_URL))


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
    return REPORT_TYPE_LABEL_MAP.get(report_type_id, report_type_id)


def _canonical_report_type_id(report_type_id: str) -> str:
    value = str(report_type_id or "").strip()
    if not value:
        return ""
    if value in REPORT_TYPE_LABEL_MAP:
        return value
    alias_to_canonical = {alias: canonical for canonical, alias in REPORT_TYPE_LABEL_MAP.items()}
    return alias_to_canonical.get(value, value)


def _resolve_report_type_filter_value(raw_value: str, available_ids: set[str]) -> str:
    value = str(raw_value or "").strip()
    if not value:
        return ""
    if value in available_ids:
        return value
    return _canonical_report_type_id(value)


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
    terminal_statuses = {"finished", "failed", "timeout", "cancelled", "canceled"}
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


def _resolve_csv_paths(
    uploaded_files: list[Any],
    upload_dir: Path,
    existing_csv_path: str = "",
) -> list[str]:
    # Always prefer explicit uploaded files over a potentially stale hidden recent-path value.
    if uploaded_files:
        paths: list[str] = []
        for uploaded_file in uploaded_files:
            filename = secure_filename(uploaded_file.filename)
            if not filename.lower().endswith((".csv", ".xlsx", ".xls", ".pcap", ".pcapng")):
                raise ValueError("Supported uploads: .csv, .xlsx, .xls, .pcap, .pcapng")
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            stored_name = f"{stem}_{uuid4().hex[:8]}{suffix}"
            destination = upload_dir / stored_name
            uploaded_file.save(destination)
            paths.append(str(destination))
        return paths

    if existing_csv_path:
        existing = _absolute_path(existing_csv_path)
        if not existing.exists():
            raise ValueError(f"Existing CSV path not found: {existing_csv_path}")
        return [str(existing)]

    if not uploaded_files:
        raise ValueError("Upload a file or choose one from recent uploads.")
    return []


def _prepare_pipeline_source(
    csv_paths: list[str],
    report_type_id: str,
    row_limit: int | None,
    sheet_name: str | None,
    upload_dir: Path,
    source_labels: dict[str, str] | None = None,
) -> tuple[str, list[str], bool]:
    if not csv_paths:
        raise ValueError("No input sources resolved.")
    source_names = [Path(p).name for p in csv_paths]
    if len(csv_paths) == 1:
        single_name = Path(csv_paths[0]).name
        # Keep row-limit behavior stable between estimate->confirm for already combined sources.
        is_precombined = bool(re.fullmatch(r"combined_[0-9a-f]{10}\.csv", single_name))
        if report_type_id == "wireshark_capture_health" and source_labels and len(source_labels) > 1:
            raise ValueError(
                "Multiple source labels were provided, but only one source file was selected. "
                "Select all intended files together before generating."
            )
        return csv_paths[0], source_names, is_precombined

    suffixes = {Path(p).suffix.lower() for p in csv_paths}
    if len(suffixes) > 1:
        raise ValueError("Multiple files must share the same type (.csv/.pcap/.pcapng).")
    if any(s in {".xlsx", ".xls"} for s in suffixes):
        raise ValueError("Multiple Excel files are not supported yet. Use a single Excel source.")
    if report_type_id != "wireshark_capture_health":
        raise ValueError("Multiple files are currently supported for wireshark report type only.")

    frames: list[pd.DataFrame] = []
    label_map = source_labels or {}
    for path in csv_paths:
        # For multi-source compare flows, row_limit is applied per file so each source is represented.
        frame = load_csv_with_limit(path, row_limit=row_limit, sheet_name=sheet_name)
        if frame.empty:
            continue
        frame = frame.copy()
        source_name = Path(path).name
        frame["source_file"] = source_name
        frame["source_label"] = _resolve_source_label(source_name, label_map)
        frames.append(frame)
    if not frames:
        raise ValueError("Resolved sources are empty after parsing.")
    combined = pd.concat(frames, ignore_index=True)
    destination = upload_dir / f"combined_{uuid4().hex[:10]}.csv"
    combined.to_csv(destination, index=False)
    return str(destination), source_names, True


def _format_source_label(source_files: list[str], sheet_name: str = "") -> str:
    if not source_files:
        return "-"
    if len(source_files) == 1:
        return source_files[0] + (f" (sheet: {sheet_name})" if sheet_name else "")
    shown = ", ".join(source_files[:3])
    suffix = f" (+{len(source_files)-3} more)" if len(source_files) > 3 else ""
    return f"{len(source_files)} files: {shown}{suffix}"


def _parse_source_labels_text(text: str) -> dict[str, str]:
    if not text.strip():
        return {}
    labels: dict[str, str] = {}
    for line in text.splitlines():
        row = line.strip()
        if not row or "=" not in row:
            continue
        key, value = row.split("=", 1)
        k = key.strip()
        v = value.strip()
        if not k or not v:
            continue
        labels[k] = v
    return labels


def _resolve_source_label(source_name: str, labels: dict[str, str]) -> str:
    stem = Path(source_name).stem
    stem_no_upload_suffix = re.sub(r"_[0-9a-fA-F]{8}$", "", stem)
    candidates = [
        source_name,
        stem,
        f"{stem_no_upload_suffix}{Path(source_name).suffix}",
        stem_no_upload_suffix,
    ]
    for key in candidates:
        value = labels.get(key, "").strip()
        if value:
            return value
    return source_name


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
        "default_prefs": _classification_default_prefs(
            family="tabular_statistical",
            domain="generic",
            mode="statistical_summary",
        ),
        "prompt_instructions": "Describe key patterns and actionable recommendations.",
        "output_schema": _default_report_output_schema(),
    }
    return yaml.safe_dump(payload, sort_keys=False)


def _classification_default_prefs(*, family: str, domain: str, mode: str) -> dict[str, str]:
    engineering_domains = {"networking", "observability", "operations", "security", "telecom"}
    business_domains = {
        "finance",
        "product",
        "sales",
        "project_management",
        "government",
        "education",
        "research",
    }
    anomaly_modes = {
        "issue_detection",
        "anomaly_detection",
        "threshold_sla",
        "burst_detection",
        "flow_bottleneck",
        "root_cause_triage",
    }
    cost_or_efficiency_domains = {"finance", "supply_chain", "manufacturing", "energy", "sales", "product"}

    if domain in engineering_domains or family in {"time_series", "event", "log_text", "hybrid"}:
        tone = "technical"
    elif domain in business_domains:
        tone = "executive"
    else:
        tone = "concise"

    if domain in engineering_domains:
        audience = "engineering"
    elif domain == "customer_support":
        audience = "customer"
    else:
        audience = "leadership"

    if mode in anomaly_modes:
        focus = "anomalies"
    elif domain in cost_or_efficiency_domains and mode in {
        "overview_summary",
        "statistical_summary",
        "variance_analysis",
        "ranking_prioritization",
    }:
        focus = "cost"
    else:
        focus = "trends"

    return {"tone": tone, "audience": audience, "focus": focus}


def _report_type_runtime_default_prefs(
    *,
    report_type_id: str,
    definition: Any,
    plugin_root: Path,
) -> dict[str, Any]:
    existing = dict(getattr(definition, "default_prefs", {}) or {})
    manifest_path = plugin_root / report_type_id / "manifest.yaml"
    if not manifest_path.exists():
        return {}
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return {}

    family = str(manifest.get("family", "")).strip()
    domain = str(manifest.get("domain", "")).strip()
    mode = str(manifest.get("mode", "")).strip()
    if not family or not domain or not mode:
        return {}

    inferred = _classification_default_prefs(family=family, domain=domain, mode=mode)
    return {key: value for key, value in inferred.items() if not str(existing.get(key, "")).strip()}


def _default_report_output_schema() -> dict[str, Any]:
    return {
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
    }


def _supported_metrics_profiles() -> set[str]:
    return list_supported_metrics_profiles()


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


def _extract_metrics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    tables = payload.get("tables", [])
    if not isinstance(tables, list):
        return {}
    for table in tables:
        if not isinstance(table, dict):
            continue
        if table.get("name") != "metrics_payload":
            continue
        rows = table.get("rows", [])
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            return rows[0]
    return {}


def _build_wireshark_ptp_benchmark(selected_reports: list[dict[str, Any]]) -> dict[str, Any]:
    wireshark_rows: list[dict[str, Any]] = []
    for report in selected_reports:
        if str(report.get("report_type_id", "")).strip() != "wireshark_capture_health":
            continue
        try:
            payload = _load_json(Path(str(report.get("json_path", ""))))
        except Exception:  # noqa: BLE001
            continue
        metrics_payload = _extract_metrics_payload(payload)
        ptp_summary = metrics_payload.get("ptp_summary", {}) if isinstance(metrics_payload, dict) else {}
        port_health = metrics_payload.get("ptp_port_health", []) if isinstance(metrics_payload, dict) else []
        top_port = port_health[0] if isinstance(port_health, list) and port_health else {}

        sync_packets = int(_safe_float(ptp_summary.get("sync_packets")))
        announce_packets = int(_safe_float(ptp_summary.get("announce_packets")))
        follow_up_packets = int(_safe_float(ptp_summary.get("follow_up_packets")))
        two_step_pct = _safe_float(ptp_summary.get("two_step_pct")) if ptp_summary.get("two_step_pct") is not None else None
        correction_median = _safe_float(ptp_summary.get("correction_ns_median")) if ptp_summary.get("correction_ns_median") is not None else None
        correction_p95 = _safe_float(ptp_summary.get("correction_ns_p95")) if ptp_summary.get("correction_ns_p95") is not None else None
        ts_delta = _safe_float(ptp_summary.get("timestamp_delta_ns_median")) if ptp_summary.get("timestamp_delta_ns_median") is not None else None
        sync_interval_median = _safe_float(top_port.get("sync_interval_ms_median")) if top_port.get("sync_interval_ms_median") is not None else None
        sync_interval_p95 = _safe_float(top_port.get("sync_interval_ms_p95")) if top_port.get("sync_interval_ms_p95") is not None else None

        score = 0
        reasons: list[str] = []
        if sync_packets > 0 and announce_packets > 0:
            score += 1
            reasons.append("Sync/Announce traffic present.")
        else:
            reasons.append("Missing Sync or Announce traffic.")

        if correction_median is not None and correction_p95 is not None and correction_median > 0:
            corr_ratio = correction_p95 / correction_median
            if corr_ratio <= 1.05:
                score += 3
                reasons.append(f"Correction-field spread is tight (p95/median={corr_ratio:.3f}).")
            elif corr_ratio <= 1.20:
                score += 2
                reasons.append(f"Correction-field spread is moderate (p95/median={corr_ratio:.3f}).")
            else:
                reasons.append(f"Correction-field spread is wide (p95/median={corr_ratio:.3f}).")
        else:
            reasons.append("Correction-field metrics missing.")

        if ts_delta is not None:
            abs_ts = abs(ts_delta)
            if abs_ts <= 5_000_000:
                score += 3
                reasons.append("Timestamp delta is very close to zero.")
            elif abs_ts <= 500_000_000:
                score += 2
                reasons.append("Timestamp delta is moderate.")
            elif abs_ts <= 5_000_000_000:
                score += 1
                reasons.append("Timestamp delta is elevated.")
            else:
                reasons.append("Timestamp delta is very large.")
        else:
            reasons.append("Timestamp delta not available.")

        if sync_interval_median is not None and sync_interval_p95 is not None and sync_interval_median > 0:
            interval_ratio = sync_interval_p95 / sync_interval_median
            if interval_ratio <= 1.20:
                score += 2
                reasons.append(f"Sync interval is stable (p95/median={interval_ratio:.3f}).")
            elif interval_ratio <= 1.50:
                score += 1
                reasons.append(f"Sync interval has some jitter (p95/median={interval_ratio:.3f}).")
            else:
                reasons.append(f"Sync interval jitter is high (p95/median={interval_ratio:.3f}).")
        else:
            reasons.append("Sync interval stats not available.")

        if score >= 7:
            likelihood = "high"
        elif score >= 4:
            likelihood = "medium"
        else:
            likelihood = "low"

        wireshark_rows.append(
            {
                "name": str(report.get("name", "-")),
                "provider": str(report.get("backend", "-")),
                "model": str(report.get("model", "-")),
                "sync_packets": sync_packets,
                "follow_up_packets": follow_up_packets,
                "announce_packets": announce_packets,
                "two_step_pct": two_step_pct,
                "correction_ns_median": correction_median,
                "correction_ns_p95": correction_p95,
                "timestamp_delta_ns_median": ts_delta,
                "sync_interval_ms_median": sync_interval_median,
                "sync_interval_ms_p95": sync_interval_p95,
                "lock_score": score,
                "lock_likelihood": likelihood,
                "reasons": reasons,
            }
        )

    if len(wireshark_rows) < 2:
        return {"supported": False}

    best_lock = max(wireshark_rows, key=lambda r: int(r.get("lock_score", 0)))
    min_corr_p95 = min((r for r in wireshark_rows if r.get("correction_ns_p95") is not None), key=lambda r: float(r["correction_ns_p95"]), default=None)
    min_abs_delta = min(
        (r for r in wireshark_rows if r.get("timestamp_delta_ns_median") is not None),
        key=lambda r: abs(float(r["timestamp_delta_ns_median"])),
        default=None,
    )
    pair_delta: dict[str, Any] = {}
    if len(wireshark_rows) == 2:
        a, b = wireshark_rows[0], wireshark_rows[1]
        pair_delta = {
            "left": a["name"],
            "right": b["name"],
            "lock_score_delta": int(a["lock_score"]) - int(b["lock_score"]),
            "correction_p95_delta": (
                (float(a["correction_ns_p95"]) - float(b["correction_ns_p95"]))
                if a.get("correction_ns_p95") is not None and b.get("correction_ns_p95") is not None
                else None
            ),
            "timestamp_delta_abs_diff": (
                (abs(float(a["timestamp_delta_ns_median"])) - abs(float(b["timestamp_delta_ns_median"])))
                if a.get("timestamp_delta_ns_median") is not None and b.get("timestamp_delta_ns_median") is not None
                else None
            ),
        }

    return {
        "supported": True,
        "rows": wireshark_rows,
        "best_lock": best_lock,
        "best_correction": min_corr_p95,
        "best_timestamp_alignment": min_abs_delta,
        "pair_delta": pair_delta,
    }


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
        # Auto-version: strip any existing _v<n> suffix, then find the next free version
        base_id = re.sub(r"_v\d+$", "", report_type_id)
        for version in range(2, 1000):
            candidate = f"{base_id}_v{version}"
            if not (report_types_dir / f"{candidate}.yaml").exists():
                report_type_id = candidate
                break
        else:
            raise ValueError(f"Could not find a free versioned id for '{base_id}'.")
    normalized["report_type_id"] = report_type_id
    return normalized


def _enhance_prompt_with_classification_context(
    prompt_text: str,
    hint_domain: str | None = None,
) -> str:
    """
    Inject domain-specific contextual questions into the prompt to guide AI generation.
    
    This enriches generic prompts with domain-specific clarifications so the AI can
    generate more contextually-aware initial drafts (e.g., asking about SLA thresholds
    for telecom data, incident classification for security data, etc.).
    
    Args:
        prompt_text: Original user-provided prompt
        hint_domain: Domain classification (telecom, observability, security, operations, finance)
    
    Returns:
        Enhanced prompt with domain-specific context questions appended
    """
    if not hint_domain:
        return prompt_text

    domain_context = {
        "telecom": """
Before generating the report type, consider these clarifying questions:
- What are your key SLA/performance thresholds? (e.g., latency < 10ms is good, > 50ms is critical)
- Should analysis be per-flow (per source-destination pair) or aggregate (whole network)?
- How do you define "degraded" vs "critical" performance states?
- Are you tracking absolute metrics (delay, jitter, loss %) or relative trends?
- Which metrics matter most for your use case? (delay, jitter, packet loss, reordering)
- What time windows are most relevant? (per-hour, per-day trends)
""",
        "observability": """
Before generating the report type, consider these clarifying questions:
- What are the key health indicators for your systems? (uptime %, response time, error rate)
- How do you want to detect anomalies? (static thresholds, baselines/percentiles, ML models)
- What time windows matter? (real-time, hourly, daily baselines)
- Are there dependencies between services that should affect health calculation?
- What triggers an alert vs. warning vs. info level in your system?
- How should you handle missing/sparse data?
""",
        "security": """
Before generating the report type, consider these clarifying questions:
- What threat patterns are you tracking? (brute force, lateral movement, data exfiltration, etc.)
- How do you classify incidents vs. vulnerabilities vs. behavioral anomalies?
- What confidence or severity thresholds should trigger escalation?
- Are you tracking root cause (attribution) or just detection?
- What time windows matter for threat analysis? (per-incident, hourly trends, weekly summaries)
- Should analysis focus on individual events or aggregated patterns?
""",
        "operations": """
Before generating the report type, consider these clarifying questions:
- What business SLAs or KPIs are you tracking? (uptime %, throughput, capacity utilization)
- Should you focus on availability (is it working?), capacity (can it handle more?), or cost efficiency?
- What escalation procedures should this report trigger? (auto-remediate, alert owner, page on-call)
- Are there capacity thresholds (e.g., > 85% utilization = warning)?
- What time windows matter? (hourly spikes, daily capacity trends, weekly planning)
- How should seasonal patterns affect baselines?
""",
        "finance": """
Before generating the report type, consider these clarifying questions:
- What are the key financial metrics? (revenue, costs, margin, variance vs. budget)
- Should you drill down by business unit, cost center, product line, or region?
- What variance thresholds trigger investigation? (1%, 5%, 10% off budget?)
- How should you handle timing differences (accrual vs. cash, period-end vs. actual close)?
- What stakeholder views matter most? (CFO, business owners, controllers)
- Should analysis show month-to-date, year-to-date, or trailing 12-month trends?
""",
    }

    context_questions = domain_context.get(hint_domain, "")
    if context_questions:
        return f"{prompt_text}{context_questions}"
    
    return prompt_text


def _generate_report_type_agent_draft(
    *,
    prompt_text: str,
    clone_from: str | None,
    clone_yaml: str,
    report_types: list[str],
    model: str,
    source_profile: dict[str, Any] | None,
    hint_domain: str | None = None,
    hint_family: str | None = None,
    hint_mode: str | None = None,
) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for AI report type generation.")

    try:
        from openai import OpenAI  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError("Install openai extra: pip install -e .[openai]") from exc

    # Enhance the prompt with domain-specific context questions
    enhanced_prompt = _enhance_prompt_with_classification_context(
        prompt_text=prompt_text,
        hint_domain=hint_domain,
    )

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=_report_type_agent_instructions(),
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(
                            {
                                "prompt": enhanced_prompt,
                                "clone_from": clone_from or "",
                                "clone_yaml": clone_yaml,
                                "existing_report_types": report_types,
                                "allowed_families": sorted(FAMILIES),
                                "allowed_domains": sorted(DOMAINS),
                                "allowed_modes": sorted(MODES),
                                "source_profile": source_profile,
                                "classification_hints": {
                                    "domain": hint_domain or "",
                                    "family": hint_family or "",
                                    "mode": hint_mode or "",
                                },
                            }
                        ),
                    }
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "report_type_agent_draft",
                "schema": _report_type_agent_response_schema(),
                "strict": False,
            }
        },
    )
    draft = json.loads(response.output_text)
    if not isinstance(draft, dict):
        raise ValueError("AI draft payload was not an object.")
    return draft


def _materialize_report_type_agent_draft(
    *,
    draft: dict[str, Any],
    config_dir: Path,
    plugin_root: Path,
    clone_from: str | None,
    source_profile: dict[str, Any] | None,
) -> str:
    report_type_id = _normalize_generated_report_type_id(
        str(draft.get("report_type_id", "")).strip() or str(draft.get("title", "")).strip(),
        config_dir,
    )
    title = str(draft.get("title", "")).strip() or _friendly_report_type_label(report_type_id)
    # If auto-versioned (ends with _v<n>), reflect that in the title too
    version_match = re.search(r"_v(\d+)$", report_type_id)
    if version_match:
        version_num = version_match.group(1)
        if not re.search(r"\bv\d+\b", title, re.IGNORECASE):
            title = f"{title} v{version_num}"
    family = str(draft.get("family", "")).strip()
    domain = str(draft.get("domain", "")).strip()
    mode = str(draft.get("mode", "")).strip()
    description = str(draft.get("description", "")).strip() or f"{title} plugin."
    prompt_instructions = str(draft.get("prompt_instructions", "")).strip()
    required_columns = [str(item).strip() for item in draft.get("required_columns", []) if str(item).strip()]
    required_columns = _sanitize_required_columns_for_source_profile(required_columns, source_profile)
    if not prompt_instructions:
        raise ValueError("AI draft is missing prompt_instructions.")
    if not required_columns:
        raise ValueError("AI draft required_columns did not match the uploaded source columns.")

    default_prefs = draft.get("default_prefs", {})
    if not isinstance(default_prefs, dict):
        raise ValueError("AI draft default_prefs must be an object.")
    for key, value in _classification_default_prefs(family=family, domain=domain, mode=mode).items():
        default_prefs.setdefault(key, value)

    scaffold_result = scaffold_report_type(
        report_type_id=report_type_id,
        title=title,
        family=family,
        domain=domain,
        mode=mode,
        required_columns=required_columns,
        version="1.0.0",
        description=description,
        owner="ai-agent",
        generator="openai_sdk",
        inherits_from=clone_from or "",
        status="draft",
        create_report_type_yaml=True,
        config_dir=config_dir,
        plugin_root=plugin_root,
        force=False,
    )

    report_yaml = {
        "report_type_id": report_type_id,
        "version": "1.0.0",
        "title": title,
        "required_columns": required_columns,
        "metrics_profile": report_type_id,
        "default_prefs": default_prefs,
        "prompt_instructions": prompt_instructions,
        "output_schema": _default_report_output_schema(),
    }
    if scaffold_result.report_type_yaml is not None:
        scaffold_result.report_type_yaml.write_text(yaml.safe_dump(report_yaml, sort_keys=False), encoding="utf-8")

    manifest = yaml.safe_load(scaffold_result.plugin_manifest.read_text(encoding="utf-8"))
    manifest["description"] = description
    if clone_from:
        manifest["inherits_from"] = clone_from
    extensions = draft.get("manifest_extensions", {})
    if isinstance(extensions, dict) and extensions:
        manifest["extensions"] = extensions
    scaffold_result.plugin_manifest.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    plugin_code = _strip_markdown_code_fences(str(draft.get("plugin_code", "")).strip())
    smoke_test_code = _strip_markdown_code_fences(str(draft.get("smoke_test_code", "")).strip())
    if plugin_code:
        scaffold_result.plugin_code.write_text(plugin_code.rstrip() + "\n", encoding="utf-8")
    if smoke_test_code:
        # Use agent-generated smoke test if it has valid imports; fall back to
        # canonical template for broken AI output.
        if "from rv_reporter.report_types.plugins import ReportPluginManager" in smoke_test_code:
            scaffold_result.plugin_smoke_test.write_text(
                smoke_test_code.rstrip() + "\n",
                encoding="utf-8",
            )
        else:
            scaffold_result.plugin_smoke_test.write_text(
                _canonical_smoke_test_template(report_type_id=report_type_id),
                encoding="utf-8",
            )
    # New plugin files were written — invalidate the default manager cache so the
    # next generation request rescans and finds this plugin.
    invalidate_plugin_cache()
    return report_type_id


def _normalize_generated_report_type_id(candidate: str, report_types_dir: Path) -> str:
    # Use title as fallback so we get a clean slug regardless
    normalized = _normalize_report_type_payload({"report_type_id": candidate, "title": candidate}, report_types_dir)
    return str(normalized["report_type_id"])


def _strip_markdown_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", stripped)
        stripped = re.sub(r"\n```$", "", stripped)
    return stripped.strip()


def _report_type_agent_instructions() -> str:
    return (
        "You are drafting a new report type for a schema-first reporting system. "
        "Return a JSON object only. Choose family, domain, and mode from the allowed sets. "
        "Use snake_case for report_type_id. Keep generator-compatible values. "
        "Use family to describe source shape: time_series for ordered timestamps, event for activity streams, "
        "entity_snapshot for per-entity status tables, relational for multi-key/tabular joins, log_text for raw text logs, "
        "tabular_statistical for mostly numeric/categorical tables, and hybrid when multiple shapes matter. "
        "Use domain=generic when the source is not strongly tied to a specific industry. "
        "Use mode=issue_detection for problem-finding, anomaly_detection for unusual outliers, statistical_summary for descriptive stats, "
        "overview_summary for broad status reports, trend_analysis for time movement, and root_cause_triage when likely drivers matter. "
        "If source_profile is present, use it to infer family, domain, mode, required_columns, and likely metrics. "
        "When source_profile is present, required_columns must be copied only from source_profile.source_metadata.columns. "
        "Never invent missing columns, never paraphrase header names, and never rename them. "
        "Copy column names exactly, including spaces, punctuation, case, and parentheses. "
        "If a field is not present in source_profile.source_metadata.columns, do not include it in required_columns. "
        "If the prompt contains domain-specific clarifying questions (e.g., 'What are your SLA thresholds?', 'How do you define degraded performance?'), "
        "read those carefully and use them to guide your understanding of the report's purpose. Answer these questions in your mind as you design the metrics and plugin. "
        "Ensure default_prefs and prompt_instructions reflect the answers to these clarifying questions. "
        "Generate plugin.py code that defines get_spec() and build(df, prefs, ctx). "
        "The build function should compute deterministic metrics from the dataframe and return a dict. "
        "Generate smoke_test_code as a pytest file that loads the plugin through ReportPluginManager. "
        "Do not wrap code in Markdown fences."
    )


def _report_type_agent_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "report_type_id",
            "title",
            "family",
            "domain",
            "mode",
            "description",
            "required_columns",
            "default_prefs",
            "prompt_instructions",
            "plugin_code",
            "smoke_test_code",
            "manifest_extensions",
        ],
        "properties": {
            "report_type_id": {"type": "string"},
            "title": {"type": "string"},
            "family": {"type": "string", "enum": sorted(FAMILIES)},
            "domain": {"type": "string", "enum": sorted(DOMAINS)},
            "mode": {"type": "string", "enum": sorted(MODES)},
            "description": {"type": "string"},
            "required_columns": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "default_prefs": {"type": "object", "additionalProperties": True},
            "prompt_instructions": {"type": "string"},
            "plugin_code": {"type": "string"},
            "smoke_test_code": {"type": "string"},
            "manifest_extensions": {"type": "object", "additionalProperties": True},
        },
    }


def _build_report_type_source_profile(
    path_value: str | Path,
    sheet_name: str | None = None,
    sample_percent: int = 100,
) -> dict[str, Any]:
    path = _absolute_path(path_value)
    metadata = describe_tabular_source(path, sheet_name=sheet_name)
    row_count = metadata.get("row_count")
    # Ensure row_count is safe to use in comparisons (convert string to int if needed)
    if isinstance(row_count, str):
        try:
            row_count = int(row_count)
        except (ValueError, TypeError):
            row_count = None
    normalized_percent = min(100, max(10, int(sample_percent or 100)))
    sample_row_limit = 25
    if isinstance(row_count, int) and row_count > 0:
        sample_row_limit = max(1, int((row_count * normalized_percent + 99) // 100))
    frame = load_csv_with_limit(path, row_limit=sample_row_limit, sheet_name=sheet_name)
    profiled = profile_dataframe(frame)
    sample_rows = _dataframe_sample_rows(frame, limit=5)
    return {
        "path": str(path),
        "source_metadata": metadata,
        "profile": profiled,
        "sample_rows": sample_rows,
        "sampling": {
            "sample_percent": normalized_percent,
            "sampled_row_count": int(len(frame.index)),
            "total_row_count": int(row_count) if isinstance(row_count, int) else None,
        },
    }


def _improve_report_type_prompt(
    *,
    prompt_text: str,
    model: str,
    hint_domain: str | None,
    hint_family: str | None,
    hint_mode: str | None,
    source_columns: list[str],
) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for AI prompt improvement.")

    try:
        from openai import OpenAI  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError("Install openai extra: pip install -e .[openai]") from exc

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=(
            "You improve prompts used to create new report types for a schema-first reporting system. "
            "Rewrite the prompt so it is clearer, more concrete, and more actionable for generating YAML, plugin logic, and tests. "
            "Preserve the original intent, but add precision around columns, computed metrics, thresholds, analysis scope, and expected outputs. "
            "Return JSON only."
        ),
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(
                            {
                                "prompt_text": prompt_text,
                                "hint_domain": hint_domain or "",
                                "hint_family": hint_family or "",
                                "hint_mode": hint_mode or "",
                                "source_columns": source_columns,
                            }
                        ),
                    }
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "improved_report_type_prompt",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["improved_prompt", "changes"],
                    "properties": {
                        "improved_prompt": {"type": "string"},
                        "changes": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
                "strict": False,
            }
        },
    )
    payload = json.loads(response.output_text)
    if not isinstance(payload, dict):
        raise ValueError("AI prompt improvement returned an invalid payload.")
    improved_prompt = str(payload.get("improved_prompt", "")).strip()
    changes = [str(item).strip() for item in payload.get("changes", []) if str(item).strip()]
    if not improved_prompt:
        raise ValueError("AI prompt improvement did not return an improved prompt.")
    return {"improved_prompt": improved_prompt, "changes": changes}


def _sanitize_required_columns_for_source_profile(
    required_columns: list[str],
    source_profile: dict[str, Any] | None,
) -> list[str]:
    if not source_profile or not isinstance(source_profile, dict):
        return required_columns

    source_metadata = source_profile.get("source_metadata", {})
    if not isinstance(source_metadata, dict):
        return required_columns

    available_columns = [str(item).strip() for item in source_metadata.get("columns", []) if str(item).strip()]
    if not available_columns:
        return required_columns

    exact_set = set(available_columns)
    normalized_lookup = {re.sub(r"\s+", " ", col).strip().lower(): col for col in available_columns}

    sanitized: list[str] = []
    for column in required_columns:
        if column in exact_set:
            if column not in sanitized:
                sanitized.append(column)
            continue
        normalized = re.sub(r"\s+", " ", column).strip().lower()
        mapped = normalized_lookup.get(normalized)
        if mapped and mapped not in sanitized:
            sanitized.append(mapped)
    return sanitized


def _dataframe_sample_rows(frame: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    sample = frame.head(limit).copy()
    sample = sample.astype(object)
    rows: list[dict[str, Any]] = []
    for row in sample.to_dict(orient="records"):
        rows.append({str(key): _json_safe_value(value) for key, value in row.items()})
    return rows


def _json_safe_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:  # noqa: BLE001
        pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # noqa: BLE001
            return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _safe_describe_source(path_value: str, sheet_name: str | None = None) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = _absolute_path(path_value)
    if not path.exists():
        return None
    try:
        return describe_tabular_source(path, sheet_name=sheet_name)
    except Exception:  # noqa: BLE001
        return None


def _safe_preflight_source(
    path_value: str,
    *,
    sheet_name: str | None = None,
    required_columns: list[str] | None = None,
) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = _absolute_path(path_value)
    if not path.exists():
        return None
    try:
        return preflight_tabular_source(
            path,
            sheet_name=sheet_name,
            required_columns=required_columns,
        )
    except Exception:  # noqa: BLE001
        return None


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
    estimated_output_tokens: int | None,
    output_root: Path,
    report_type_id: str,
) -> dict[str, float | int | str | None]:
    if estimated_output_tokens is None:
        return {
            "model": model,
            "input_tokens_est": estimate_tokens(prompt_text),
            "output_tokens_est": None,
            "output_tokens_user_budget": None,
            "output_tokens_scale_factor": None,
            "input_cost_usd_est": None,
            "output_cost_usd_est": None,
            "total_cost_usd_est": None,
            "raw_total_cost_usd_est": None,
            "pricing_source_url": _provider_pricing_url(provider_name),
            "pricing_verified_date": PRICING_VERIFIED_DATE,
            "calibration_applied": False,
            "calibration_factor": 1.0,
            "calibration_sample_size": 0,
            "calibration_scope": "unbounded_output",
        }

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
