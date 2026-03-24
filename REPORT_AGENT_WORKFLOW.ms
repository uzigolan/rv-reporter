# REPORT AGENT WORKFLOW

## Goal

Support multiple agent paths for creating/updating report types:
- **Built-in Web UI** (primary, recommended): prompt → AI spec → scaffold, all in-browser
- OpenAI Agent Builder + exported Python integration
- n8n automation flow

All paths call the same repo contract:
- `rv-reporter scaffold-report-type ...`
- `py scripts/run_report_type_guardrails.py`

## Recommended Architecture

### Primary path: Built-in Web UI

The Web UI at `/report-types/new` provides a fully integrated AI agent flow:

1. Open `http://localhost:5000/report-types/new`.
2. (Optional) Pick an existing report type to clone taxonomy/column hints from.
3. (Optional) Upload a source CSV/Excel file — the AI inspects real column names, types, and sample rows.
4. Describe your report type in plain language in the prompt field.
5. Click **Generate Draft With AI**.
6. The server calls OpenAI (`REPORT_TYPE_AGENT_MODEL`, default `gpt-4o-mini`) with a structured output schema.
7. The AI returns: `report_type_id`, `title`, `family`, `domain`, `mode`, `required_columns`, `prompt_instructions`, and starter `plugin_code`.
8. The scaffold system writes four files automatically:
   - `configs/report_types/<id>.yaml`
   - `report_type_plugins/<id>/manifest.yaml`
   - `report_type_plugins/<id>/plugin.py` (AI-generated code inserted)
   - `report_type_plugins/<id>/tests/test_smoke.py`
9. Browser redirects to the YAML config view for inspection.

Env vars:

| Var | Default | Purpose |
|---|---|---|
| `REPORT_TYPE_AGENT_MODEL` | `gpt-4o-mini` | Model for report-type design |
| `OPENAI_API_KEY` | — | Required |

### Hybrid path: Agent Builder + repo

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
