"""Diagnostic: full create+publish+generate run with tmp plugin dir patch."""
from __future__ import annotations
import os, tempfile
from pathlib import Path

os.environ["GENERATION_HIDDEN_REPORT_TYPES"] = "dummy"

import rv_reporter.web as wm
import rv_reporter.report_types.plugins as pm
from rv_reporter.web import create_app

PLUGIN_CODE = (
    "from __future__ import annotations\n"
    "from typing import Any\n"
    "import pandas as pd\n"
    "\n"
    "def get_spec() -> dict:\n"
    "    return {'metrics_profile': 'ct', 'api_version': 1, 'title': 'CT', 'description': 'CT'}\n"
    "\n"
    "def build(df: pd.DataFrame, prefs: dict, ctx: Any) -> dict:\n"
    "    return {'rows': int(len(df))}\n"
)


def fake_draft(**_):  # type: ignore[return]
    return {
        "report_type_id": "ct",
        "title": "CT",
        "family": "time_series",
        "domain": "networking",
        "mode": "issue_detection",
        "description": "CT.",
        "required_columns": ["DateTimeUTC", "DiscardRatePct_Emulated"],
        "default_prefs": {},
        "prompt_instructions": "Test.",
        "plugin_code": PLUGIN_CODE,
        "smoke_test_code": "",
        "manifest_extensions": {},
    }


with tempfile.TemporaryDirectory() as tmp:
    t = Path(tmp)
    pr = t / "report_type_plugins"
    pr.mkdir()
    (pr / "manifest.schema.yaml").write_text(
        Path("report_type_plugins/manifest.schema.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    app = create_app({
        "TESTING": True,
        "REPORT_TYPES_DIR": str(t / "rt"),
        "PLUGIN_ROOT": str(pr),
        "UPLOAD_FOLDER": str(t / "u"),
        "OUTPUT_FOLDER": str(t / "o"),
    })

    msgs: list = []
    orig_draft = wm._generate_report_type_agent_draft
    orig_flash = wm.flash

    def cap_flash(m, c="message"):  # type: ignore[misc]
        msgs.append((c, m))
        orig_flash(m, c)

    wm._generate_report_type_agent_draft = fake_draft  # type: ignore[assignment]
    wm.flash = cap_flash  # type: ignore[assignment]

    orig_dir = pm._DEFAULT_MANAGER._plugin_dir
    orig_schema = pm._DEFAULT_MANAGER._manifest_schema_path
    orig_cache = pm._DEFAULT_MANAGER._plugin_cache
    pm._DEFAULT_MANAGER._plugin_dir = pr
    pm._DEFAULT_MANAGER._manifest_schema_path = pr / "manifest.schema.yaml"
    pm._DEFAULT_MANAGER._plugin_cache = None

    try:
        with app.test_client() as client:
            client.post(
                "/report-types/new",
                data={"prompt_text": "test", "existing_csv_path": "samples/ETX2i_twamp.csv"},
                follow_redirects=True,
            )
            print("PLUGIN MANIFEST STATUS:", __import__("yaml").safe_load(
                (pr / "ct" / "manifest.yaml").read_text(encoding="utf-8")
            ).get("status"))
            client.post("/report-types/publish", data={"report_type_id": "ct"}, follow_redirects=True)
            r = client.post(
                "/generate",
                data={
                    "report_type_id": "ct",
                    "provider": "local",
                    "existing_csv_path": "samples/ETX2i_twamp.csv",
                },
                follow_redirects=False,
            )
            print("GEN STATUS:", r.status_code, "LOC:", r.headers.get("Location"))
            print("FLASH:", msgs[-5:])
            print("OUTPUT FILES:", list((t / "o" / "ct").glob("*.json")))
    finally:
        wm._generate_report_type_agent_draft = orig_draft  # type: ignore[assignment]
        wm.flash = orig_flash  # type: ignore[assignment]
        pm._DEFAULT_MANAGER._plugin_dir = orig_dir
        pm._DEFAULT_MANAGER._manifest_schema_path = orig_schema
        pm._DEFAULT_MANAGER._plugin_cache = orig_cache
