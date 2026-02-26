# Agent Workflows

## OpenAI SDK Agent

1. Call `rv-reporter scaffold-report-type` with user intent mapped to:
- `report_type_id`
- `family`, `domain`, `mode`
- `required_column[]`
- `generator=openai_sdk`
2. Edit generated `plugin.py` with report logic.
3. Run `py scripts/run_report_type_guardrails.py`.
4. If checks pass, commit and open PR.

## n8n Agent

1. Collect form inputs for taxonomy and columns.
2. Execute shell node with `rv-reporter scaffold-report-type ... --generator n8n`.
3. Execute shell node with `py scripts/run_report_type_guardrails.py`.
4. Parse test output and gate downstream publish/merge steps.

## Safe Extension Pattern

When the agent needs unsupported metadata, write it under manifest `extensions`:

```yaml
extensions:
  data_contract:
    source: jira_export_v2
    pii_policy: masked
  runtime_hints:
    timeout_seconds: 20
    max_rows: 200000
```

Keep `extensions` read-only for core behavior unless explicit support is added in core code.
