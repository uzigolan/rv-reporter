# REPORT AGENT WORKFLOW

## Goal

Support two agent paths for creating/updating report types:
- OpenAI Agent Builder + exported Python integration
- n8n automation flow

Both paths must call the same repo contract:
- `rv-reporter scaffold-report-type ...`
- `py scripts/run_report_type_guardrails.py`

## Recommended Architecture (Hybrid)

1. Design and test the agent behavior in OpenAI Agent Builder UI.
2. Export/integrate Python code from the Agent Builder Code tab (Advanced integration).
3. Store production code in this repo.
4. Use n8n to execute the same scaffold + guardrail commands.

This keeps one source of truth for report generation rules and tests.

## OpenAI Agent Builder Path

1. Open: `https://platform.openai.com/agent-builder/edit`
2. Configure instructions and actions for report generation:
- create new report type
- modify existing report type
- inherit existing report type
3. Test in Builder UI.
4. Publish a version.
5. Open **Code** tab -> **Advanced integration** -> Python.
6. Copy code into repo, for example:
- `agents/openai_agent/client.py`
- `agents/openai_agent/workflow.py`
7. In workflow code, execute:
- `rv-reporter scaffold-report-type ... --generator openai_sdk`
- `py scripts/run_report_type_guardrails.py`
8. If guardrails pass, create PR/commit flow.

## n8n Path

1. Input node (Webhook/Form) collects:
- `report_type_id`, `title`
- `family`, `domain`, `mode`
- `required_columns[]`
- optional `inherits_from`
2. Execute Command node:
- `rv-reporter scaffold-report-type ... --generator n8n`
3. Execute Command node:
- `py scripts/run_report_type_guardrails.py`
4. IF node:
- pass: continue to publish/PR
- fail: send diagnostics to user/channel

## Safe Extensibility Rules

1. Keep unknown metadata in `manifest.yaml -> extensions`.
2. Do not change core plugin API unless necessary.
3. Keep `metrics_profile` aligned across:
- `configs/report_types/<id>.yaml`
- `report_type_plugins/<id>/manifest.yaml`
- `report_type_plugins/<id>/plugin.py:get_spec()`

## Standard Validation Command

```bash
py scripts/run_report_type_guardrails.py
```

Includes:
- plugin manifest/schema checks
- plugin contract tests
- scaffold tests

## Example Command (New Report Type)

```bash
rv-reporter scaffold-report-type \
  --report-type-id net_usage \
  --title "Network Usage" \
  --family time_series \
  --domain networking \
  --mode trend_analysis \
  --required-column timestamp \
  --required-column interface \
  --required-column bytes_in \
  --required-column bytes_out \
  --generator openai_sdk
```
