from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich import print

from rv_reporter.orchestrator import run_pipeline
from rv_reporter.providers.anthropic_provider import AnthropicMessagesProvider
from rv_reporter.providers.mock_provider import MockProvider
from rv_reporter.providers.openai_chat_provider import OpenAIChatCompletionsProvider
from rv_reporter.providers.openai_provider import OpenAIResponsesProvider
from rv_reporter.report_types.registry import ReportTypeRegistry
from rv_reporter.web import create_app

app = typer.Typer(help="CSV -> report pipeline")


@app.command("list-report-types")
def list_report_types(config_dir: str = "configs/report_types") -> None:
    registry = ReportTypeRegistry(config_dir=config_dir)
    for rid in registry.list_report_types():
        print(rid)


@app.command("build-report")
def build_report(
    csv: str = typer.Option(..., help="Path to input file (.csv/.xlsx/.xls/.pcap/.pcapng)"),
    report_type: str = typer.Option(..., help="Report type id"),
    prefs: Optional[str] = typer.Option(None, help="Path to user prefs JSON"),
    output_dir: str = typer.Option("outputs", help="Output directory"),
    provider: str = typer.Option(
        "local",
        help="local|openai|claude|gemini|xai (legacy: mock)",
    ),
    model: str = typer.Option("gpt-5-mini", help="Model for openai provider"),
    api_key: Optional[str] = typer.Option(None, help="Provider API token override"),
    api_base_url: Optional[str] = typer.Option(None, help="Provider base URL override"),
    row_limit: Optional[int] = typer.Option(None, help="Optional max number of rows to load from source"),
    sheet_name: Optional[str] = typer.Option(None, help="Excel sheet name for .xlsx/.xls inputs"),
    ignore_column: list[str] = typer.Option(
        [],
        "--ignore-column",
        help="Column name to drop before validation/metrics. Repeat for multiple columns.",
    ),
) -> None:
    user_prefs = {}
    if prefs:
        user_prefs = json.loads(Path(prefs).read_text(encoding="utf-8"))

    provider_name = provider.strip().lower()
    if provider_name == "mock":
        provider_name = "local"

    if provider_name in {"openai", "xai", "gemini", "claude"}:
        env_key = {
            "openai": "OPENAI_API_KEY",
            "xai": "XAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
        }[provider_name]
        default_base = {
            "openai": "",
            "xai": "https://api.x.ai/v1",
            "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
            "claude": "https://api.anthropic.com/v1",
        }[provider_name]
        resolved_key = (api_key or "").strip() or os.getenv(env_key, "").strip()
        if not resolved_key:
            raise typer.BadParameter(f"Missing API token. Set {env_key} or pass --api-key.")
        resolved_base_url = (api_base_url or "").strip() or default_base or None
        if provider_name == "openai":
            provider_impl = OpenAIResponsesProvider(
                model=model,
                api_key=resolved_key,
                base_url=resolved_base_url,
                default_headers=None,
            )
        elif provider_name == "claude":
            provider_impl = AnthropicMessagesProvider(
                model=model,
                api_key=resolved_key,
                base_url=resolved_base_url,
            )
        else:
            provider_impl = OpenAIChatCompletionsProvider(
                model=model,
                api_key=resolved_key,
                base_url=resolved_base_url,
                default_headers=None,
            )
    else:
        provider_impl = MockProvider()

    report_json_file, report_html_file = run_pipeline(
        csv_path=csv,
        report_type_id=report_type,
        user_prefs=user_prefs,
        output_dir=output_dir,
        provider=provider_impl,
        row_limit=row_limit,
        sheet_name=sheet_name,
        ignore_columns=ignore_column or None,
    )
    print(f"[green]Report JSON:[/green] {report_json_file}")
    print(f"[green]Report HTML:[/green] {report_html_file}")


@app.command("run-web")
def run_web(
    host: str = typer.Option("127.0.0.1", help="Flask host"),
    port: int = typer.Option(5000, help="Flask port"),
    debug: bool = typer.Option(True, help="Enable debug mode"),
) -> None:
    web_app = create_app()
    web_app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    app()
