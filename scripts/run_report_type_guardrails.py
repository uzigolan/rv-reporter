from __future__ import annotations

import subprocess
import sys


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_report_type_guardrails.py",
        "tests/test_plugins.py",
        "tests/test_scaffold.py",
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
