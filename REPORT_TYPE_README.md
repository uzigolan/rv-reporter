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

- Families: `time_series`, `tabular_statistical`, `event`, `log_text`, `hybrid`
- Domains: `networking`, `project_management`, `healthcare`, `operations`, `security`, `finance`
- Modes: `health_score`, `anomaly_detection`, `trend_analysis`, `threshold_sla`, `burst_detection`, `correlation_analysis`, `distribution_analysis`, `top_n_hotspots`, `flow_bottleneck`

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
