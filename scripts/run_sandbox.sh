#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install -e ".[dev,openai]"
python -m playwright install chromium || echo "Warning: Playwright Chromium install failed. PDFs may fallback without charts."

if [ ! -f ".env.sandbox" ] && [ -f ".env.sandbox.example" ]; then
  cp .env.sandbox.example .env.sandbox
fi

if [ -f ".env.sandbox" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env.sandbox
  set +a
fi

python -m rv_reporter.web
