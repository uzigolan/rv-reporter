# Report Type Plugins

Each subfolder is an isolated metrics plugin for a report-type `metrics_profile`.

## Contract

- File path: `report_type_plugins/<plugin_id>/plugin.py`
- Manifest path: `report_type_plugins/<plugin_id>/manifest.yaml`
- Required functions:
  - `get_spec() -> dict`
  - `build(df, prefs, ctx) -> dict`

`manifest.yaml` must validate against `report_type_plugins/manifest.schema.yaml`.
This is the canonical structure your generator agents (OpenAI SDK or n8n) should produce.

`get_spec()` must return:

- `metrics_profile`: unique profile id used by YAML report types.
- `api_version`: currently `1`.
- `title`: display name.
- `description`: optional text.

`build()` receives:

- `df`: report input dataframe.
- `prefs`: merged default/user prefs.
- `ctx`: context object with `report_type_id`.

Plugins are discovered by `rv_reporter.report_types.plugins.ReportPluginManager`.
