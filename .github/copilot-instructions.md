# rv-reporter — Workspace Instructions

Schema-first CSV reporting pipeline that produces validated JSON, styled HTML, and charted PDF reports. Supports deterministic local generation and OpenAI/Anthropic-backed narrative generation.

## Build & Test

```powershell
# Install (first time)
python -m pip install -e ".[dev,openai]"
python -m playwright install chromium   # required for PDF export

# Run all tests
pytest

# Run specific test
pytest tests/test_pipeline.py -v

# Start web UI (sandbox)
rv-reporter-web                         # or: rv-reporter run-web
# or use sandbox launcher:
powershell -ExecutionPolicy Bypass -File .\run_sandbox.ps1
```

Environment: copy `.env.sandbox.example` → `.env.sandbox` before first run. Set `OPENAI_API_KEY` for OpenAI-backed generation. PCAP ingestion requires `tshark` on PATH.

## Architecture

Seven-step pipeline in `src/rv_reporter/orchestrator.py` (`run_pipeline()`):

1. **Registry** — load YAML from `configs/report_types/<id>.yaml` → `ReportTypeRegistry`
2. **Ingest** — `services/ingest.py`: load CSV/Excel/pcap, validate required columns
3. **Profile** — `services/profiler.py`: shape, dtypes, missing rates
4. **Metrics** — `ReportPluginManager` (dynamic plugin) or `services/metrics.py` (legacy fallback)
5. **Generate** — provider (`MockProvider`, `OpenAIResponsesProvider`, `AnthropicMessagesProvider`, …)
6. **Validate** — `services/validator.py`: JSON Schema against `output_schema` in YAML
7. **Render** — `rendering/html_renderer.py` (Jinja2) → `rendering/pdf_renderer.py` (Playwright)

Key files:
- `src/rv_reporter/models.py` — `ReportOutput` Pydantic model (canonical output shape)
- `src/rv_reporter/report_types/registry.py` — `ReportTypeRegistry`
- `src/rv_reporter/report_types/plugins.py` — `ReportPluginManager` (dynamic discovery)
- `src/rv_reporter/web.py` — Flask routes (upload, generate, cost estimation, artifact browsing)
- `src/rv_reporter/cli.py` — Typer CLI (`build-report`, `scaffold-report-type`, `list-report-types`)

## Plugin System

A **report type plugin** lives in `report_type_plugins/<plugin_id>/` and requires two files:

**`manifest.yaml`** — validated against `report_type_plugins/manifest.schema.yaml`:
```yaml
plugin_id: my_plugin
api_version: 1
version: "1.0.0"
title: "My Plugin"
family: time_series          # time_series | tabular_statistical | event | log_text | entity_snapshot | relational | hybrid
domain: generic              # generic | networking | observability | project_management | healthcare | operations | security | finance | product | sales | customer_support | supply_chain | manufacturing | energy | telecom | research | education | government
mode: issue_detection        # issue_detection | anomaly_detection | statistical_summary | trend_analysis | root_cause_triage | ...
metrics_profile: my_plugin
entrypoint: plugin.py
status: active               # active | draft | deprecated
owner: team-name
generator: openai_sdk
```

**`plugin.py`** — must expose two callables:
```python
def get_spec() -> dict:
    return {"metrics_profile": "my_plugin", "api_version": 1, "title": "...", "description": "..."}

def build(df: pd.DataFrame, prefs: dict, ctx) -> dict:
    # ctx is ReportPluginContext(report_type_id=...)
    # return dict compatible with ReportOutput fields
    ...
```

Plugin discovery is automatic. Override directory via `RV_REPORT_PLUGINS_DIR` env var.

## Report Type Config (YAML)

`configs/report_types/<id>.yaml` structure:
```yaml
report_type_id: my_report      # must match filename
version: "1.0.0"
title: "Display Name"
required_columns:
  - "Column Name"              # exact header names from CSV
metrics_profile: my_plugin     # plugin_id or legacy profile key
default_prefs:
  alert_threshold: 0.05
prompt_instructions: |
  Multi-line AI generation instructions...
output_schema:                 # JSON Schema; always include all ReportOutput fields
  type: object
  additionalProperties: false
  required: [report_type_id, report_title, summary, sections, alerts, recommendations, tables, charts, metadata]
  properties: { ... }
```

Protected (cannot be deleted via UI): `network_queue_congestion`, `twamp_session_health`, `pm_export_health`, `jira_issue_portfolio`, `ms_biomarker_registry_health`.

## Testing Conventions

- Tests use `pytest` + `tmp_path` for isolated output; no mocking framework
- Naming: `test_<feature>_<scenario>(tmp_path)`
- Plugin tests seed `manifest.schema.yaml` + `manifest.yaml` + `plugin.py` into `tmp_path` before loading
- Pipeline tests call `run_pipeline()` directly and assert on JSON payload keys and file existence
- No network calls in tests; use `MockProvider` (local deterministic generation)

## Output Artifacts

Each run writes stamped files + latest aliases in the output directory:
```
<report_type>.<run_id>.report.json    # structured data
<report_type>.<run_id>.report.html    # styled HTML
<report_type>.<run_id>.report.pdf     # Playwright-rendered PDF
<report_type>.report.json             # latest alias
```

## Common Pitfalls

- **PDF missing charts**: run `python -m playwright install chromium`
- **PCAP ingestion**: requires system `tshark` on PATH
- **OpenAI costs**: provider always prompts for cost confirmation before generation
- **Column names**: `required_columns` must match exact CSV headers (case-sensitive)
- **New report type**: add YAML config → add plugin (or legacy metrics) → add test
- **Scaffold a new plugin**: `rv-reporter scaffold-report-type --report-type-id <id> --title "..." --family time_series --domain networking --mode trend_analysis`

## References

- [docs/architecture.md](docs/architecture.md)
- [docs/UI_GUIDE.md](docs/UI_GUIDE.md)
- [report_type_plugins/README.md](report_type_plugins/README.md)
- [INSTALL.md](INSTALL.md)
