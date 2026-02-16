# Architecture

## Workflow

1. Load report type definition from `configs/report_types`.
2. Ingest CSV and validate required columns.
3. Profile dataset (shape, dtypes, missing rates).
4. Compute deterministic metrics based on `metrics_profile`.
5. Generate report JSON with provider (`local` or `openai`).
   For OpenAI provider, UI includes an estimated cost confirmation step before sending the request.
6. Validate against report type output schema.
7. Render HTML and persist JSON + HTML outputs.

## Services

- `services/ingest.py`: CSV load and schema prechecks
- `services/profiler.py`: generic dataframe profile
- `services/metrics.py`: domain metrics computation
- `providers/`: report JSON generation abstraction
- `rendering/html_renderer.py`: deterministic HTML rendering
- `web.py`: Flask UI routes for upload, generation, and artifact browsing

## Extending with new report types

1. Add `configs/report_types/<new_id>.yaml`.
2. Add metric logic in `services/metrics.py`.
3. Add sample CSV and test coverage in `tests/`.
