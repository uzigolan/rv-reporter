# INSTALL

This guide installs and runs `rv-reporter` on Windows or Linux/macOS.

## Requirements

- Python 3.10+
- Internet access (for package install)
- For charted PDFs: Playwright Chromium

Linux also needs system libraries for `pycairo`:

- `pkg-config`
- cairo development headers
- python development headers
- compiler toolchain

## Clone repository

### Windows

```powershell
git clone https://github.com/uzigolan/rv-reporter.git
cd rv-reporter
```

### Linux/macOS

```bash
git clone https://github.com/uzigolan/rv-reporter.git
cd rv-reporter
```

## Option A: Sandbox launcher (recommended)

### Windows

```powershell
Copy-Item .env.sandbox.example .env.sandbox
powershell -ExecutionPolicy Bypass -File .\run_sandbox.ps1
```

### Linux/macOS

```bash
cp .env.sandbox.example .env.sandbox
chmod +x ./run_sandbox.sh ./scripts/run_sandbox.sh
./run_sandbox.sh
```

The launcher:

- creates `.venv`
- installs `.[dev,openai]`
- installs Chromium via Playwright
- loads `.env.sandbox`
- starts the web app on `127.0.0.1:5000`

## Option B: Manual install

### 1) Create and activate virtualenv

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

Linux (install system packages first):

Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y pkg-config libcairo2-dev python3-dev build-essential libffi-dev libjpeg-dev zlib1g-dev
```

Fedora/RHEL:

```bash
sudo dnf install -y pkgconf-pkg-config cairo-devel python3-devel gcc gcc-c++ libffi-devel libjpeg-turbo-devel zlib-devel
```

Then install Python dependencies:

```bash
python -m pip install -e ".[dev,openai]"
python -m playwright install chromium
```

### 3) Configure environment

Create `.env.sandbox` from sample:

```bash
cp .env.sandbox.example .env.sandbox
```

Set OpenAI key if you want OpenAI generation:

```env
OPENAI_API_KEY=sk-...
```

### 4) Run app

```bash
python -m rv_reporter.web
```

Open: `http://127.0.0.1:5000`

## Verify install

Run tests:

```bash
python -m pytest -q
```

You should see all tests passing.

## Common issues

- `playwright`/PDF issues:
  - rerun `python -m playwright install chromium`
- stale UI/CSS:
  - hard refresh browser (`Ctrl+F5`)
- OpenAI calls fail:
  - confirm `OPENAI_API_KEY` exists in `.env.sandbox`
  - ensure provider is `openai` in UI
