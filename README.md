# rv-reporter

Schema-first CSV reporting app that turns datasets into:

- validated JSON reports
- styled HTML reports with charts
- PDF reports (browser-rendered from HTML)

It supports deterministic local generation and OpenAI-backed narrative generation with cost-estimation approval.

## Features

- report type registry: `configs/report_types/*.yaml`
- pluggable metrics pipeline per report type
- OpenAI/local provider switch
- row-limit and token-budget cost controls
- cost confirmation before OpenAI generation
- report history + JSON/HTML/PDF/raw artifact views
- UI creation/deletion of custom report types

## Built-in report types

- `network_queue_congestion` (protected)
- `twamp_session_health` (protected)
- `pm_export_health` (protected)

## Quick start (sandbox scripts, recommended)

Windows (PowerShell):

```powershell
Copy-Item .env.sandbox.example .env.sandbox
powershell -ExecutionPolicy Bypass -File .\run_sandbox.ps1
```

Linux/macOS:

```bash
cp .env.sandbox.example .env.sandbox
chmod +x ./run_sandbox.sh ./scripts/run_sandbox.sh
./run_sandbox.sh
```

What the sandbox launcher does:

- creates `.venv` if missing
- installs `.[dev,openai]`
- installs Playwright Chromium (for charted PDFs)
- loads `.env.sandbox`
- starts Flask UI at `http://127.0.0.1:5000`

## Manual setup

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\activate
python -m pip install -e ".[dev,openai]"
python -m playwright install chromium
rv-reporter-web
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev,openai]"
python -m playwright install chromium
rv-reporter-web
```

## OpenAI configuration

Set `OPENAI_API_KEY` in `.env.sandbox`:

```env
OPENAI_API_KEY=sk-...
```

In UI:

- choose `Provider = openai`
- set `Row Limit` and `Output Tokens Budget`
- review cost estimate
- click `Continue and Generate`

## UI options

Generate page (`/`) fields:

- `Report Type`: choose schema/metrics template
- `Provider`: `local` or `openai`
- `Model`: OpenAI model used for `openai` provider
- `CSV Source`: `sample` or `upload`
- `Sample CSV` / `Upload CSV`: input dataset
- `Tone`: `concise`, `executive`, `technical`
- `Audience`: `leadership`, `engineering`, `customer`
- `Focus`: `trends`, `anomalies`, `cost`
- `Threshold Key` + `Threshold Value`: optional metric override
- `Row Limit`: cap rows processed (cost/performance control)
- `Output Tokens Budget`: output-size estimate for OpenAI cost preview

Cost confirmation page (`/generate` with `openai`):

- shows `Report Type`, `CSV Source`, `Rows Used`
- shows `Model`, estimated input/output tokens, and estimated cost
- actions: `Continue and Generate` or `Abort`

Other UI pages:

- `/reports`: report history with JSON/HTML/PDF/OpenAI-raw links
- `/report-types/new`: create custom report type YAML
- `/report-types`: list/delete custom report types (protected types cannot be deleted)

## CLI examples

```bash
rv-reporter build-report \
  --csv samples/network_queues.csv \
  --report-type network_queue_congestion \
  --output-dir outputs/network_queue \
  --row-limit 1000
```

```bash
rv-reporter build-report \
  --csv samples/ETX2i_twamp.csv \
  --report-type twamp_session_health \
  --output-dir outputs/twamp \
  --row-limit 1000
```

```bash
rv-reporter build-report \
  --csv samples/pm-csv-es.csv \
  --report-type pm_export_health \
  --output-dir outputs/pm \
  --row-limit 2000
```

## Output artifacts

Per run, the app writes:

- `<report_type>.<run_id>.report.json`
- `<report_type>.<run_id>.report.html`
- `<report_type>.<run_id>.report.pdf`
- `<report_type>.<run_id>.openai.raw.json` (OpenAI runs)

Also writes latest aliases:

- `<report_type>.report.json`
- `<report_type>.report.html`
- `<report_type>.report.pdf`

Default UI output root: `outputs/web`.

## Project layout

- `src/rv_reporter/web.py`: Flask app/routes
- `src/rv_reporter/orchestrator.py`: pipeline orchestration
- `src/rv_reporter/services`: ingest/profile/metrics/cost estimation
- `src/rv_reporter/providers`: local + OpenAI providers
- `src/rv_reporter/rendering`: HTML/PDF rendering
- `configs/report_types`: report type definitions
- `samples`: sample CSVs
- `tests`: automated tests

## Troubleshooting

- If UI changes do not appear: hard refresh browser (`Ctrl+F5`)
- If PDF misses charts: ensure Playwright Chromium is installed:
  - `python -m playwright install chromium`
- If OpenAI fails: verify `OPENAI_API_KEY` and provider/model selection
- If report not in history: open `/reports` and check `History source` path

## More docs

- [INSTALL.md](INSTALL.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/UI_GUIDE.md](docs/UI_GUIDE.md)
