# UI Guide

This guide documents the rv-reporter web UI workflows, pages, fields, and actions.

## Navigation Overview

Top navigation:
- `Generate`: main report generation page (`/`)
- `New Type`: create a custom report type (`/report-types/new`)
- `Manage Types`: list and delete custom report types (`/report-types`)
- `Reports`: generated report history (`/reports`)
- `About`: app overview (`/about`)

## 1) Generate Page (`/`)

Purpose:
- Build a report from a tabular source (`.csv`, `.xlsx`, `.xls`).
- Choose local vs OpenAI generation.
- Apply preferences and optional thresholds.

### Fields

- `Report Type`
  - Selects the report schema + metrics profile.
  - Includes protected built-ins and your custom types.

- `Provider`
  - `local`: deterministic local generation, no OpenAI API use.
  - `openai`: uses OpenAI for narrative generation.

- `Model`
  - Used only when `Provider = openai`.
  - Automatically disabled/greyed when `Provider = local`.

- `CSV Source`
  - `sample`: use a bundled sample file from `samples/`.
  - `upload`: upload your own local file.

- `Sample File`
  - Active only when source is `sample`.
  - Shows available sample files.

- `Upload File`
  - Active only when source is `upload`.
  - Accepts `.csv`, `.xlsx`, `.xls`.

- `Sheet (for .xlsx/.xls)`
  - For Excel files with multiple sheets, select one sheet.
  - For single-sheet files, sheet is auto-selected.
  - For CSV, this is not required.

- `Tone`
  - Writing style: `concise`, `executive`, `technical`.

- `Audience`
  - Target reader: `leadership`, `engineering`, `customer`.

- `Focus`
  - Emphasis: `trends`, `anomalies`, `cost`.

- `Threshold Key (optional)` + `Threshold Value (optional)`
  - Optional override for report-specific thresholds.
  - Value must be numeric.

- `Row Limit (cost control)`
  - Max rows loaded from source.
  - Useful for runtime and OpenAI cost control.

- `Output Tokens Budget (OpenAI estimate)`
  - Expected response size used for OpenAI cost estimation.

### Help Popups

- Each field has a `?` icon.
- Hovering/focusing `?` shows popup help.
- Help popup auto-hides on mouse leave / blur.

### Generate Action

- `Generate Report` button submits the form.
- While running:
  - Button is disabled.
  - Loading message appears.
- On browser back/forward navigation, submit UI resets correctly.

## 2) OpenAI Cost Confirmation (`/generate` intermediate)

Shown when:
- `Provider = openai`
- before final report generation

Displays:
- Report type
- Source file (+ sheet when relevant)
- Rows used
- Model
- Estimated input/output tokens
- Estimated total cost

Actions:
- `Continue and Generate`: proceeds with generation.
- `Abort`: returns to generate page.

## 3) Result Page (post-generation)

Displays:
- Success message for generated report type.
- Artifact buttons:
  - `View JSON`
  - `View HTML`
  - `View PDF` (when available)
  - `View OpenAI Raw` (OpenAI runs only)
  - `All reports`

Also shows:
- Summary
- Alerts
- Metadata chips

## 4) Reports Page (`/reports`)

Purpose:
- Browse report history and artifacts.

### Columns

- `Name`
- `Date/Time` (format `YY-MM-DD, HH:MM`)
- `Engine`
- `Model`
- `Source` (+ sheet sub-line if present)
- `Rows`
- `Time Took` (human format: `X sec`, `Y min Z sec`)
- `JSON`
- `HTML`
- `PDF`
- `OpenAI Raw`
- `Action`

### Artifact icons

Legend:
- `↗` = open
- `⤓` = download

Behavior:
- JSON: open icon
- HTML: open + download icons
- PDF: open + download icons
- OpenAI Raw: open icon

### Delete Action

- `Delete` removes report artifacts for that run:
  - `.report.json`
  - `.report.html`
  - `.report.pdf`
  - `.openai.raw.json` (if exists)

## 5) Create Report Type (`/report-types/new`)

Purpose:
- Create a custom report type by submitting YAML.

Behavior:
- Validates required fields and schema basics.
- If `report_type_id` missing, app auto-generates it from title.
- Saves into `configs/report_types/<report_type_id>.yaml`.

## 6) Manage Report Types (`/report-types`)

Purpose:
- List all report types and remove custom ones.

Behavior:
- Protected built-ins cannot be deleted from UI.
- Custom types can be deleted.

## 7) About Page (`/about`)

Shows:
- App summary
- Protected report type list
- priced model list

## Source + Excel sheet workflow notes

- For sample Excel files:
  - sheet list is fetched automatically when selecting the sample.

- For uploaded Excel files:
  - selected file is uploaded in background.
  - sheet list is fetched and shown immediately.
  - selected uploaded path is reused for generation.

## Output locations

Default output root:
- `outputs/web`

Per report run (timestamped):
- `<report_type>.<run_id>.report.json`
- `<report_type>.<run_id>.report.html`
- `<report_type>.<run_id>.report.pdf`
- `<report_type>.<run_id>.openai.raw.json` (OpenAI)

Latest aliases:
- `<report_type>.report.json`
- `<report_type>.report.html`
- `<report_type>.report.pdf`

## Common tips

- If UI changes do not appear, hard refresh (`Ctrl+F5`).
- Restart Flask after code changes.
- For PDF charts, install Playwright Chromium:
  - `python -m playwright install chromium`
