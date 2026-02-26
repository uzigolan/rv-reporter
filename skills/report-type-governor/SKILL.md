---
name: report-type-governor
description: Create, update, or inherit report type plugins in this repo with schema-safe manifests and regression checks. Use when adding a new report type, changing an existing plugin, creating agent-generated report types, validating plugin manifests, or running report-type guardrail tests after code changes.
---

# Report Type Governor

Execute this workflow when touching `configs/report_types/*.yaml` or `report_type_plugins/**`.

## Workflow

1. Scaffold or update report type files.
2. Validate plugin manifest and plugin contract.
3. Run guardrail tests.
4. Summarize created/changed files and next actions.

## Create A New Report Type

Run:

```powershell
rv-reporter scaffold-report-type `
  --report-type-id <id> `
  --title "<title>" `
  --family <family> `
  --domain <domain> `
  --mode <mode> `
  --required-column <col1> `
  --required-column <col2> `
  --generator <openai_sdk|n8n|manual>
```

This generates:
- `configs/report_types/<id>.yaml` (unless disabled)
- `report_type_plugins/<id>/manifest.yaml`
- `report_type_plugins/<id>/plugin.py`
- `report_type_plugins/<id>/tests/test_smoke.py`

## Inherit Or Modify Existing Report Type

1. Copy or edit `report_type_plugins/<existing>/plugin.py`.
2. Set `inherits_from` in `manifest.yaml` when derived.
3. Keep `metrics_profile` equal across:
- config YAML `metrics_profile`
- plugin `get_spec()["metrics_profile"]`
- manifest `metrics_profile`

## Guardrail Checks

Run:

```powershell
py scripts/run_report_type_guardrails.py
```

This executes:
- `tests/test_report_type_guardrails.py`
- `tests/test_plugins.py`
- `tests/test_scaffold.py`

## Reference

Read `references/agent-workflows.md` for OpenAI SDK and n8n generation patterns.
