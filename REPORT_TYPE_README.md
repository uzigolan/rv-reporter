# REPORT TYPE README

This guide explains how an AI agent creates, updates, or inherits report types safely.

## Architecture

Core stays stable:
- ingestion/profiling
- rendering
- orchestration
- validation

Per-report logic lives in plugins:
- `report_type_plugins/<report_type_id>/plugin.py`
- `report_type_plugins/<report_type_id>/manifest.yaml`
- `report_type_plugins/<report_type_id>/tests/test_smoke.py`

Report definition lives in config:
- `configs/report_types/<report_type_id>.yaml`

## Required Contracts

1. `manifest.yaml` must pass `report_type_plugins/manifest.schema.yaml`.
2. `plugin.py` must implement:
- `get_spec()`
- `build(df, prefs, ctx)`
3. These values must match:
- config `metrics_profile`
- manifest `metrics_profile`
- `get_spec()["metrics_profile"]`

## Agent Workflow: Create New Report Type

### Option A — Web UI (recommended)

1. Open `http://localhost:5000/report-types/new`.
2. (Optional) Pick an existing type to clone from.
3. (Optional) Upload source CSV/Excel — gives the AI real column names, types, and sample rows.
4. Enter a plain-language description of your report type.
5. Click **Generate Draft With AI**.
6. The UI calls OpenAI, runs the scaffold, and redirects to the YAML config view.
7. Review the four generated files:
   - `configs/report_types/<id>.yaml`
   - `report_type_plugins/<id>/manifest.yaml`
   - `report_type_plugins/<id>/plugin.py`
   - `report_type_plugins/<id>/tests/test_smoke.py`
8. Run guardrails to validate:

```bash
py scripts/run_report_type_guardrails.py
```

### Option B — CLI scaffold (manual)

1. Scaffold:

```bash
rv-reporter scaffold-report-type \
  --report-type-id <id> \
  --title "<title>" \
  --family <family> \
  --domain <domain> \
  --mode <mode> \
  --required-column <col1> \
  --required-column <col2> \
  --generator <openai_sdk|n8n>
```

2. Implement logic in `plugin.py`.
3. Add/adjust smoke test.
4. Run:

```bash
py scripts/run_report_type_guardrails.py
```

## Agent Workflow: Modify Existing Report Type

1. Edit plugin code and/or manifest.
2. Keep taxonomy + profile consistent.
3. Run guardrail tests.
4. Validate output by building a sample report with `rv-reporter build-report`.

## Agent Workflow: Inherit Existing Report Type

1. Scaffold new type with new id.
2. Set in new manifest:

```yaml
inherits_from: existing_type_id
```

3. Start `plugin.py` by reusing parent behavior, then add deltas.
4. Keep `metrics_profile` equal to new id.

## Taxonomy

- Family = source-data shape:
  `time_series`, `tabular_statistical`, `event`, `log_text`, `entity_snapshot`, `relational`, `hybrid`
- Domain = business context. Use `generic` if the source is not tied to a specific vertical:
  `generic`, `networking`, `observability`, `project_management`, `healthcare`, `operations`, `security`, `finance`, `product`, `sales`, `customer_support`, `supply_chain`, `manufacturing`, `energy`, `telecom`, `research`, `education`, `government`
- Mode = primary analysis intent:
  `overview_summary`, `health_score`, `issue_detection`, `anomaly_detection`, `trend_analysis`, `statistical_summary`, `threshold_sla`, `burst_detection`, `correlation_analysis`, `distribution_analysis`, `variance_analysis`, `change_detection`, `segmentation_analysis`, `ranking_prioritization`, `forecast_outlook`, `top_n_hotspots`, `flow_bottleneck`, `root_cause_triage`

Quick mapping:
- Find general problems: `issue_detection`
- Find outliers: `anomaly_detection`
- Show descriptive stats: `statistical_summary`
- Show movement over time: `trend_analysis`
- Explain likely drivers: `root_cause_triage`
- Rank biggest offenders: `top_n_hotspots` or `ranking_prioritization`

## Future Unknown Requirements

Use manifest `extensions` for fields the core does not yet model:

```yaml
extensions:
  feature_flags:
    experimental_metric: true
  runtime_hints:
    max_rows: 250000
```

Rule:
- Put new metadata in `extensions` first.
- Promote to first-class schema/core fields only after repeated use and explicit core support.
