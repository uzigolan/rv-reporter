"""Diagnostic: test create+publish flow and print flash area."""
from __future__ import annotations
import os, tempfile, sys
from pathlib import Path

os.environ["GENERATION_HIDDEN_REPORT_TYPES"] = "some_other_type"

import rv_reporter.web as wm
from rv_reporter.web import create_app

PLUGIN_CODE = (
    "from __future__ import annotations\n"
    "from typing import Any\n"
    "import pandas as pd\n"
    "\n"
    "def get_spec() -> dict:\n"
    "    return {'metrics_profile': 'custom_test', 'api_version': 1, 'title': 'Custom Test', 'description': 'Test'}\n"
    "\n"
    "def build(df: pd.DataFrame, prefs: dict, ctx: Any) -> dict:\n"
    "    return {'rows': int(len(df))}\n"
)

def fake_draft(**_):  # type: ignore[return]
    return {
        "report_type_id": "custom_test",
        "title": "Custom Test",
        "family": "time_series",
        "domain": "networking",
        "mode": "issue_detection",
        "description": "Test.",
        "required_columns": ["DateTimeUTC", "DiscardRatePct_Emulated"],
        "default_prefs": {},
        "prompt_instructions": "Test.",
        "plugin_code": PLUGIN_CODE,
        "smoke_test_code": "",
        "manifest_extensions": {},
    }


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    plugin_root = tmp_path / "report_type_plugins"
    plugin_root.mkdir(parents=True)
    (plugin_root / "manifest.schema.yaml").write_text(
        Path("report_type_plugins/manifest.schema.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    app = create_app({
        "TESTING": True,
        "REPORT_TYPES_DIR": str(tmp_path / "report_types"),
        "PLUGIN_ROOT": str(plugin_root),
        "UPLOAD_FOLDER": str(tmp_path / "uploads"),
        "OUTPUT_FOLDER": str(tmp_path / "outputs"),
    })

    orig = wm._generate_report_type_agent_draft
    wm._generate_report_type_agent_draft = fake_draft  # type: ignore[assignment]
    try:
        with app.test_client() as client:
            create_resp = client.post(
                "/report-types/new",
                data={"prompt_text": "test", "existing_csv_path": "samples/ETX2i_twamp.csv"},
                follow_redirects=True,
            )
            print("CREATE STATUS:", create_resp.status_code)
            manifest_path = plugin_root / "custom_test" / "manifest.yaml"
            print("MANIFEST EXISTS:", manifest_path.exists())
            if manifest_path.exists():
                print("MANIFEST STATUS:", __import__("yaml").safe_load(manifest_path.read_text(encoding="utf-8")).get("status"))

            pub_resp = client.post(
                "/report-types/publish",
                data={"report_type_id": "custom_test"},
                follow_redirects=True,
            )
            print("PUBLISH STATUS:", pub_resp.status_code)
            pub_text = pub_resp.data.decode("utf-8", errors="replace")
            idx = pub_text.find("<main")
            section = pub_text[idx : idx + 1000]
            print("RESPONSE AROUND MAIN:")
            print(section)
            print("---")
            print("'Published' in response:", "Published" in pub_text)
            print("flash div present:", 'class="flash' in pub_text)
    finally:
        wm._generate_report_type_agent_draft = orig  # type: ignore[assignment]
