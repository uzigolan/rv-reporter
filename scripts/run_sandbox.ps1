param(
  [switch]$OpenBrowser
)

$ErrorActionPreference = "Stop"

Set-Location (Resolve-Path "$PSScriptRoot\..")

if (-not (Test-Path ".venv")) {
  py -m venv .venv
}

$python = ".\.venv\Scripts\python.exe"
$pip = ".\.venv\Scripts\pip.exe"

& $pip install -e ".[dev,openai]"
try {
  & $python -m playwright install chromium
} catch {
  Write-Warning "Playwright Chromium install failed. PDFs may fallback without charts."
}

if (-not (Test-Path ".env.sandbox") -and (Test-Path ".env.sandbox.example")) {
  Copy-Item ".env.sandbox.example" ".env.sandbox"
}

if (Test-Path ".env.sandbox") {
  Get-Content ".env.sandbox" | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
      $parts = $line.Split("=", 2)
      $name = $parts[0].Trim()
      $value = $parts[1].Trim().Trim('"').Trim("'")
      Set-Item -Path "Env:$name" -Value $value
    }
  }
}

if ($OpenBrowser) {
  Start-Process "http://127.0.0.1:5000"
}

& $python -m rv_reporter.web
