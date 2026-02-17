#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Linux system deps needed by pycairo (pulled by PDF stack).
if [ "$(uname -s)" = "Linux" ]; then
  if command -v pkg-config >/dev/null 2>&1; then
    if ! pkg-config --exists cairo; then
      if command -v apt-get >/dev/null 2>&1; then
        echo "Installing Linux build dependencies (cairo, pkg-config, python headers)..."
        SUDO_CMD=""
        if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
          SUDO_CMD="sudo"
        fi
        $SUDO_CMD apt-get update
        $SUDO_CMD apt-get install -y \
          pkg-config \
          libcairo2-dev \
          python3-dev \
          build-essential \
          libffi-dev \
          libjpeg-dev \
          zlib1g-dev
      else
        echo "Warning: cairo dev packages missing and apt-get not found."
        echo "Install cairo + pkg-config + python dev headers via your distro package manager."
      fi
    fi
  fi
fi

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
