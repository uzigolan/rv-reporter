# Session Handoff (2026-02-16)

## Project
- Repo: `rv-ai-skills-2`
- App: Flask UI for CSV -> metrics -> report JSON/HTML
- Default report type: `network_queue_congestion`

## What Was Implemented

### Core pipeline
- Schema-driven report registry from `configs/report_types/*.yaml`
- CSV ingest + required-column validation
- Metrics engine with `network_queue_congestion` profile
- OpenAI provider integration (Responses API) + local provider
- JSON schema validation before save

### UI
- Generate page (`/`)
- Cost confirmation page for OpenAI (`Continue` / `Abort`)
- Reports history page (`/reports`)
- Raw OpenAI response viewer (`/reports/raw`)
- New report type page (`/report-types/new`)
- Manage report types page (`/report-types`) with delete action for custom types

### Saving/history
- Reports now save with run ID format:
  - `report_type_id.YYMMDD_HHMM_microseconds.report.json`
  - `report_type_id.YYMMDD_HHMM_microseconds.report.html`
  - `report_type_id.YYMMDD_HHMM_microseconds.openai.raw.json` (OpenAI only)
- Latest compatibility files are also maintained:
  - `report_type_id.report.json/.html/.openai.raw.json`
- Report metadata includes:
  - `metadata.report_id`
  - `metadata.generated_at_utc`

### Cost controls
- Row limit input in UI and pipeline (`row_limit`)
- OpenAI preflight estimate before generation
- Pricing map in `src/rv_reporter/services/cost_estimator.py`

### Charts/visuals (network report)
- Modernized report style
- Added charts:
  1. Drop Ratio Over Time (%)
  2. Dropped Bytes Over Time
  3. Top Queues by Mean Drop Ratio (%) (bar)
  4. Dropped Bytes Share (donut)
- Added axis titles/legends and chart safety improvements (downsampling, fixed chart container height)

## Current Built-in Report Types
- `network_queue_congestion` only
- Removed:
  - `ops_kpi_summary`
  - `finance_monthly_variance`

## Important Behavior Notes
- `local` provider: no API calls, deterministic local output
- `openai` provider: calls OpenAI, cost confirmation shown first
- For network reports, deterministic `metrics_payload` is force-injected to avoid empty charts even when model omits `tables`

## Run Commands

### Windows
```powershell
.\run_sandbox.ps1
```

### Linux/macOS
```bash
./run_sandbox.sh
```

Both scripts install `.[dev,openai]`, load `.env.sandbox`, and run Flask.

## URLs
- Generate: `http://127.0.0.1:5000/`
- New report type: `http://127.0.0.1:5000/report-types/new`
- Manage types: `http://127.0.0.1:5000/report-types`
- Reports history: `http://127.0.0.1:5000/reports`

## Last Known Status
- Tests passing in session: `12 passed`
- App functional with report generation, history, raw output, and chart rendering.

## Suggested Next Enhancements
1. Add export buttons for charts (PNG).
2. Add report-history deletion from UI (artifacts, not just type configs).
3. Add richer type editor UX (form-based instead of raw YAML textarea).
