# CLI README

`rv-reporter` provides pipeline, web, and report-type scaffolding commands.

## Commands

## `list-report-types`

List report types from YAML configs.

Options:
- `--config-dir` (default: `configs/report_types`)

Example:

```bash
rv-reporter list-report-types --config-dir configs/report_types
```

## `build-report`

Build report artifacts from CSV/Excel/pcap input.

Options:
- `--csv` (required)
- `--report-type` (required)
- `--prefs` path to JSON preferences
- `--output-dir` (default: `outputs`)
- `--provider` `local|openai|claude|gemini|xai|mock` (default: `local`)
- `--model` (default: `gpt-5-mini`)
- `--api-key` provider token override
- `--api-base-url` provider base URL override
- `--row-limit` optional row cap
- `--sheet-name` Excel sheet
- `--ignore-column` repeatable

Examples:

```bash
rv-reporter build-report --csv samples/network_queues.csv --report-type network_queue_congestion --provider local --output-dir outputs/network
```

```bash
rv-reporter build-report --csv samples/ETX2i_twamp.csv --report-type twamp_session_health --provider openai --model gpt-5-mini --row-limit 1000
```

## `run-web`

Run Flask web app.

Options:
- `--host` (default: `127.0.0.1`)
- `--port` (default: `5000`)
- `--debug` (default: `true`)

Example:

```bash
rv-reporter run-web --host 0.0.0.0 --port 5000 --debug
```

## `scaffold-report-type`

Generate plugin package and optional report-type YAML for agent workflows.

Options:
- `--report-type-id` (required) `[a-z0-9_]+`
- `--title` (required)
- `--family` (required): `time_series|tabular_statistical|event|log_text|entity_snapshot|relational|hybrid`
- `--domain` (required): `generic|networking|observability|project_management|healthcare|operations|security|finance|product|sales|customer_support|supply_chain|manufacturing|energy|telecom|research|education|government`
- `--mode` (required): `overview_summary|health_score|issue_detection|anomaly_detection|trend_analysis|statistical_summary|threshold_sla|burst_detection|correlation_analysis|distribution_analysis|variance_analysis|change_detection|segmentation_analysis|ranking_prioritization|forecast_outlook|top_n_hotspots|flow_bottleneck|root_cause_triage`
- `--required-column` (required, repeatable)
- `--version` (default: `1.0.0`)
- `--description` (optional)
- `--owner` (default: `platform`)
- `--generator` (default: `manual`) e.g. `openai_sdk`, `n8n`
- `--inherits-from` (optional)
- `--status` `draft|active|deprecated` (default: `draft`)
- `--create-report-yaml` bool (default: `true`)
- `--config-dir` (default: `configs/report_types`)
- `--plugin-root` (default: `report_type_plugins`)
- `--force` overwrite existing files

Examples:

```bash
rv-reporter scaffold-report-type --report-type-id net_usage --title "Network Usage" --family time_series --domain networking --mode trend_analysis --required-column timestamp --required-column interface --required-column bytes_in --required-column bytes_out --generator openai_sdk
```

```bash
rv-reporter scaffold-report-type --report-type-id jira_flow_health --title "Jira Flow Health" --family event --domain project_management --mode flow_bottleneck --required-column issue_key --required-column status --required-column created --required-column updated --generator n8n
```

```bash
rv-reporter scaffold-report-type --report-type-id dataset_overview --title "Dataset Overview" --family entity_snapshot --domain generic --mode statistical_summary --required-column entity_id --required-column status --generator openai_sdk
```

## Guardrail Tests (recommended per change)

```bash
py scripts/run_report_type_guardrails.py
```
