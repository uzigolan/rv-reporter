from pathlib import Path
import io
import json
import os
from urllib.parse import urlencode

from rv_reporter.web import create_app, _build_report_type_source_profile, _parse_source_labels_text, _prepare_pipeline_source, _resolve_source_label


def test_index_renders() -> None:
    app = create_app({"TESTING": True})
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Tabular File to Structured Report" in response.data
    assert b"Describe Report (Natural Language - Optional)" not in response.data
    assert b"<span>Tone</span>" not in response.data
    assert b"<span>Audience</span>" not in response.data
    assert b"<span>Focus</span>" not in response.data
    assert b"<span>Threshold Key (optional)</span>" not in response.data
    assert b"<span>Threshold Value (optional)</span>" not in response.data
    assert b"Leave blank for unlimited rows." in response.data
    assert b"Leave blank for unlimited output." in response.data


def test_about_includes_doc_links() -> None:
    app = create_app({"TESTING": True})
    client = app.test_client()
    response = client.get("/about")
    assert response.status_code == 200
    assert b"/docs/install" in response.data
    assert b"/docs/architecture" in response.data
    assert b"/docs/ui-guide" in response.data


def test_logic_page_renders_new_type_diagram_tab() -> None:
    app = create_app({"TESTING": True})
    client = app.test_client()
    response = client.get("/logic")
    assert response.status_code == 200
    assert b"New Type" in response.data
    assert b"Sample Source" in response.data
    assert b"Generate" in response.data


def test_doc_view_renders_markdown() -> None:
    app = create_app({"TESTING": True})
    client = app.test_client()
    response = client.get("/docs/ui-guide")
    assert response.status_code == 200
    assert b"UI Guide" in response.data


def test_report_type_yaml_api_returns_yaml(tmp_path: Path) -> None:
    report_types_dir = tmp_path / "report_types"
    report_types_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = report_types_dir / "demo_type.yaml"
    yaml_path.write_text(
        "report_type_id: demo_type\nversion: '1.0.0'\ntitle: Demo\nrequired_columns: []\nmetrics_profile: ops_kpi\noutput_schema: {type: object}\n",
        encoding="utf-8",
    )
    app = create_app(
        {
            "TESTING": True,
            "REPORT_TYPES_DIR": str(report_types_dir),
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()
    response = client.get("/api/report-type-yaml?report_type_id=demo_type")
    assert response.status_code == 200
    assert "yaml" in response.json
    assert "report_type_id: demo_type" in response.json["yaml"]


def test_generate_from_sample(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    app = create_app(
        {
            "TESTING": True,
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(output_root),
        }
    )
    client = app.test_client()
    response = client.post(
        "/generate",
        data={
            "report_type_id": "twamp_session_health",
            "provider": "local",
            "model": "gpt-4.1-mini",
            "existing_csv_path": "samples/ETX2i_twamp.csv",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"report generated" in response.data.lower()
    report_files = sorted((output_root / "twamp_session_health").glob("*.report.json"))
    assert report_files
    payload = json.loads(report_files[-1].read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    assert "generation_duration_seconds" in metadata
    assert "agent_workflow" not in metadata
    assert metadata.get("tone") == "technical"
    assert metadata.get("audience") == "engineering"
    assert metadata.get("focus") == "anomalies"


def test_generate_blocks_when_source_preflight_fails(tmp_path: Path, monkeypatch) -> None:
    output_root = tmp_path / "outputs"
    app = create_app(
        {
            "TESTING": True,
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(output_root),
        }
    )
    client = app.test_client()

    monkeypatch.setattr(
        "rv_reporter.web.preflight_tabular_source",
        lambda *args, **kwargs: {
            "ok": False,
            "issues": ["Parsed data still contains repeated header rows."],
            "warnings": [],
        },
    )

    response = client.post(
        "/generate",
        data={
            "report_type_id": "twamp_session_health",
            "provider": "local",
            "model": "local-metrics",
            "existing_csv_path": "samples/ETX2i_twamp.csv",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Source preflight failed: Parsed data still contains repeated header rows." in response.data
    assert not sorted((output_root / "twamp_session_health").glob("*.report.json"))


def test_report_type_review_page_shows_draft_workflow(tmp_path: Path) -> None:
    report_type_id = "draft_type"
    report_types_dir = tmp_path / "report_types"
    plugin_root = tmp_path / "plugins"
    report_types_dir.mkdir(parents=True, exist_ok=True)
    plugin_root.mkdir(parents=True, exist_ok=True)
    (report_types_dir / f"{report_type_id}.yaml").write_text(
        "report_type_id: draft_type\n"
        "version: '1.0.0'\n"
        "title: Draft Type\n"
        "required_columns: [timestamp]\n"
        "metrics_profile: draft_type\n"
        "default_prefs: {}\n"
        "prompt_instructions: test\n"
        "output_schema: {type: object}\n",
        encoding="utf-8",
    )
    (plugin_root / report_type_id / "tests").mkdir(parents=True, exist_ok=True)
    (plugin_root / report_type_id / "manifest.yaml").write_text(
        "plugin_id: draft_type\napi_version: 1\nversion: '1.0.0'\nstatus: draft\n",
        encoding="utf-8",
    )
    (plugin_root / report_type_id / "plugin.py").write_text("def get_spec(): return {}\ndef build(df, prefs, ctx): return {}\n", encoding="utf-8")
    (plugin_root / report_type_id / "tests" / "test_smoke.py").write_text("def test_smoke():\n    assert True\n", encoding="utf-8")

    app = create_app(
        {
            "TESTING": True,
            "REPORT_TYPES_DIR": str(report_types_dir),
            "PLUGIN_ROOT": str(plugin_root),
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["report_type_review_contexts"] = {
            report_type_id: {
                "draft_workflow": json.dumps(
                    [
                        {
                            "agent": "report_type_request",
                            "status": "completed",
                            "summary": "Captured the natural-language request for a new report type.",
                            "details": {"prompt_text": "create a KPI report"},
                        }
                    ]
                )
            }
        }

    response = client.get(f"/report-types/view?report_type_id={report_type_id}")

    assert response.status_code == 200
    assert b"Draft Workflow" in response.data
    assert b"same report type can be reused with different source data" in response.data
    assert b"report_type_request" in response.data


def test_openai_generate_shows_cost_confirmation(tmp_path: Path) -> None:
    app = create_app(
        {
            "TESTING": True,
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()
    response = client.post(
        "/generate",
        data={
            "report_type_id": "twamp_session_health",
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "existing_csv_path": "samples/ETX2i_twamp.csv",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Cost Estimate" in response.data
    assert b"Unlimited" in response.data


def test_benchmark_report_type_filter_accepts_legacy_alias(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    network_dir = output_root / "network_queue_congestion"
    twamp_dir = output_root / "twamp_session_health"
    network_dir.mkdir(parents=True, exist_ok=True)
    twamp_dir.mkdir(parents=True, exist_ok=True)

    network_path = network_dir / "network_queue_congestion.20260222_1200_000001.report.json"
    twamp_path = twamp_dir / "twamp_session_health.20260222_1200_000002.report.json"
    base_payload = {
        "summary": "",
        "sections": [],
        "alerts": [],
        "recommendations": [],
        "metadata": {},
    }
    network_path.write_text(
        json.dumps({**base_payload, "report_type_id": "network_queue_congestion"}, ensure_ascii=False),
        encoding="utf-8",
    )
    twamp_path.write_text(
        json.dumps({**base_payload, "report_type_id": "twamp_session_health"}, ensure_ascii=False),
        encoding="utf-8",
    )

    app = create_app(
        {
            "TESTING": True,
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(output_root),
        }
    )
    client = app.test_client()
    response = client.get("/benchmark?report_type_id=network")
    assert response.status_code == 200
    assert b"network.20260222_1200_000001" in response.data
    assert b"twamp.20260222_1200_000002" not in response.data


def test_prepare_pipeline_source_combines_multiple_wireshark_csvs(tmp_path: Path) -> None:
    p1 = tmp_path / "a.csv"
    p2 = tmp_path / "b.csv"
    p1.write_text(
        "frame_time_epoch,frame_len,src_ip,dst_ip,transport,src_port,dst_port,frame_protocols\n"
        "1.0,64,10.0.0.1,224.0.1.129,PTP,319,319,eth:ip:udp:ptp\n",
        encoding="utf-8",
    )
    p2.write_text(
        "frame_time_epoch,frame_len,src_ip,dst_ip,transport,src_port,dst_port,frame_protocols\n"
        "2.0,64,10.0.0.2,224.0.1.129,PTP,319,319,eth:ip:udp:ptp\n",
        encoding="utf-8",
    )
    combined_path, source_names, was_combined = _prepare_pipeline_source(
        csv_paths=[str(p1), str(p2)],
        report_type_id="wireshark_capture_health",
        row_limit=None,
        sheet_name=None,
        upload_dir=tmp_path,
        source_labels={"a.csv": "Port 4", "b.csv": "Port 3"},
    )
    assert source_names == ["a.csv", "b.csv"]
    assert was_combined is True
    combined = Path(combined_path)
    assert combined.exists()
    content = combined.read_text(encoding="utf-8")
    assert "source_file" in content
    assert "source_label" in content
    assert "a.csv" in content
    assert "b.csv" in content
    assert "Port 4" in content
    assert "Port 3" in content


def test_prepare_pipeline_source_marks_precombined_file() -> None:
    combined = Path("outputs/web/combined_aac1dedda6.csv")
    path, names, was_combined = _prepare_pipeline_source(
        csv_paths=[str(combined)],
        report_type_id="wireshark_capture_health",
        row_limit=1000,
        sheet_name=None,
        upload_dir=Path("."),
        source_labels={},
    )
    assert path.endswith("combined_aac1dedda6.csv")
    assert names == ["combined_aac1dedda6.csv"]
    assert was_combined is True


def test_benchmark_includes_wireshark_ptp_panel_for_two_reports(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    ws_dir = output_root / "wireshark_capture_health"
    ws_dir.mkdir(parents=True, exist_ok=True)

    payload_base = {
        "report_type_id": "wireshark_capture_health",
        "summary": "",
        "sections": [],
        "alerts": [],
        "recommendations": [],
        "charts": [],
        "metadata": {"generation_backend": "local", "generation_model": "local-metrics"},
    }
    p1 = ws_dir / "wireshark_capture_health.260222_1200_000001.report.json"
    p2 = ws_dir / "wireshark_capture_health.260222_1201_000002.report.json"
    p1.write_text(
        json.dumps(
            {
                **payload_base,
                "tables": [
                    {
                        "name": "metrics_payload",
                        "rows": [
                            {
                                "ptp_summary": {
                                    "sync_packets": 600,
                                    "follow_up_packets": 0,
                                    "announce_packets": 300,
                                    "correction_ns_median": 10000.0,
                                    "correction_ns_p95": 10400.0,
                                    "timestamp_delta_ns_median": -20_000_000.0,
                                },
                                "ptp_port_health": [
                                    {"sync_interval_ms_median": 62.5, "sync_interval_ms_p95": 70.0}
                                ],
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    p2.write_text(
        json.dumps(
            {
                **payload_base,
                "tables": [
                    {
                        "name": "metrics_payload",
                        "rows": [
                            {
                                "ptp_summary": {
                                    "sync_packets": 500,
                                    "follow_up_packets": 0,
                                    "announce_packets": 250,
                                    "correction_ns_median": 10000.0,
                                    "correction_ns_p95": 11500.0,
                                    "timestamp_delta_ns_median": -2_000_000_000.0,
                                },
                                "ptp_port_health": [
                                    {"sync_interval_ms_median": 62.5, "sync_interval_ms_p95": 90.0}
                                ],
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    app = create_app(
        {
            "TESTING": True,
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(output_root),
        }
    )
    client = app.test_client()
    query = urlencode({"json_path": [str(p1), str(p2)]}, doseq=True)
    response = client.get(f"/benchmark?{query}")
    assert response.status_code == 200
    assert b"PTP Benchmark (Wireshark)" in response.data
    assert b"Lock Likelihood" in response.data


def test_benchmark_baseline_includes_tone_audience_focus_checks(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    report_dir = output_root / "network_queue_congestion"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "report_type_id": "network_queue_congestion",
        "summary": "",
        "sections": [],
        "alerts": [],
        "recommendations": [],
        "tables": [],
        "charts": [],
        "metadata": {
            "generation_backend": "local",
            "generation_model": "local-metrics",
            "tone": "technical",
            "audience": "engineering",
            "focus": "anomalies",
            "source_rows_used": 100,
            "source_csv": "sample.csv",
        },
    }
    p1 = report_dir / "network_queue_congestion.260222_1300_000001.report.json"
    p2 = report_dir / "network_queue_congestion.260222_1301_000002.report.json"
    p1.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    p2.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    app = create_app(
        {
            "TESTING": True,
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(output_root),
        }
    )
    client = app.test_client()
    query = urlencode({"json_path": [str(p1), str(p2)]}, doseq=True)
    response = client.get(f"/benchmark?{query}")
    assert response.status_code == 200
    assert b"Same tone" in response.data
    assert b"Same audience" in response.data
    assert b"Same focus" in response.data


def test_parse_source_labels_text() -> None:
    parsed = _parse_source_labels_text(
        "Master_ETX-205A_port-3_nok.pcapng=Port 3 (NOK)\n"
        "Master_ETX-205A_port-4_not-sure-if-ok=Port 4\n"
        "badline\n"
    )
    assert parsed["Master_ETX-205A_port-3_nok.pcapng"] == "Port 3 (NOK)"
    assert parsed["Master_ETX-205A_port-4_not-sure-if-ok"] == "Port 4"


def test_resolve_source_label_matches_original_name_after_upload_suffix() -> None:
    labels = {
        "Master_ETX-205A_port-4_not-sure-if-ok.pcapng": "Port 4 (OK)",
        "Master_ETX-205A_port-3_nok": "Port 3 (NOK)",
    }
    assert (
        _resolve_source_label("Master_ETX-205A_port-4_not-sure-if-ok_81c3ca59.pcapng", labels)
        == "Port 4 (OK)"
    )
    assert _resolve_source_label("Master_ETX-205A_port-3_nok_9fd4a112.pcapng", labels) == "Port 3 (NOK)"


def test_create_report_type_from_ui(tmp_path: Path) -> None:
    report_types_dir = tmp_path / "report_types"
    plugin_root = tmp_path / "report_type_plugins"
    plugin_root.mkdir(parents=True, exist_ok=True)
    (plugin_root / "manifest.schema.yaml").write_text(
        Path("report_type_plugins/manifest.schema.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    app = create_app(
        {
            "TESTING": True,
            "REPORT_TYPES_DIR": str(report_types_dir),
            "PLUGIN_ROOT": str(plugin_root),
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()
    get_resp = client.get("/report-types/new")
    assert get_resp.status_code == 200
    assert b"Create Report Type With AI" in get_resp.data
    assert b"Preview Sample" in get_resp.data
    assert b"Source Sampling For AI Drafting" in get_resp.data
    assert b'value="100" checked' in get_resp.data

    def _fake_agent_draft(**_: object) -> dict[str, object]:
        return {
            "report_type_id": "temp_custom_report",
            "title": "Temp Custom Report",
            "family": "time_series",
            "domain": "networking",
            "mode": "trend_analysis",
            "description": "Temp Custom Report plugin.",
            "required_columns": ["DateTimeUTC", "DiscardRatePct_Emulated"],
            "default_prefs": {},
            "prompt_instructions": "Summarize latency drift and spikes.",
            "plugin_code": (
                "from __future__ import annotations\n\n"
                "from typing import Any\n\n"
                "import pandas as pd\n\n"
                "def get_spec() -> dict[str, Any]:\n"
                "    return {\"metrics_profile\": \"temp_custom_report\", \"api_version\": 1, \"title\": \"Temp Custom Report\", \"description\": \"Temp Custom Report plugin.\"}\n\n"
                "def build(df: pd.DataFrame, prefs: dict[str, Any], ctx: Any) -> dict[str, Any]:\n"
                "    return {\"row_count\": int(len(df)), \"columns\": list(df.columns)}\n"
            ),
            "smoke_test_code": (
                "from __future__ import annotations\n\n"
                "import pandas as pd\n\n"
                "from rv_reporter.report_types.plugins import ReportPluginManager\n\n"
                "def test_temp_custom_report_smoke_build() -> None:\n"
                "    manager = ReportPluginManager(plugin_dir=\"report_type_plugins\")\n"
                "    result = manager.compute_metrics(metrics_profile=\"temp_custom_report\", df=pd.DataFrame({\"DiscardRatePct_Emulated\": [1, 2, 3]}), prefs={}, report_type_id=\"temp_custom_report\")\n"
                "    assert isinstance(result, dict)\n"
            ),
            "manifest_extensions": {},
        }

    from rv_reporter import web as web_module

    original = web_module._generate_report_type_agent_draft
    web_module._generate_report_type_agent_draft = _fake_agent_draft
    try:
        post_resp = client.post(
            "/report-types/new",
            data={
                "prompt_text": "Create a latency trend report.",
                "existing_csv_path": "samples/ETX2i_twamp.csv",
                "source_sample_percent": "20",
            },
            follow_redirects=True,
        )
    finally:
        web_module._generate_report_type_agent_draft = original
    assert post_resp.status_code == 200
    assert (report_types_dir / "temp_custom_report.yaml").exists()
    assert (plugin_root / "temp_custom_report" / "plugin.py").exists()
    assert b"Review Report Type" in post_resp.data
    assert b"Preview Sample Report" in post_resp.data
    assert b"Publish Report Type" not in post_resp.data
    assert b"automatically be marked as planned" in post_resp.data


def test_improve_prompt_api_uses_ai_helper(tmp_path: Path, monkeypatch) -> None:
    app = create_app(
        {
            "TESTING": True,
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()

    def _fake_improve(**_: object) -> dict[str, object]:
        return {
            "improved_prompt": "Create a queue health report using queue_id, timestamp, drop_pct, and avg_delay_ms. Flag alerts above 5% drop.",
            "changes": ["Added exact column names.", "Added a concrete threshold."],
        }

    monkeypatch.setattr("rv_reporter.web._improve_report_type_prompt", _fake_improve)

    response = client.post(
        "/api/improve-prompt",
        json={
            "prompt_text": "make a queue report",
            "hint_domain": "networking",
            "source_columns": ["queue_id", "drop_pct"],
        },
    )

    assert response.status_code == 200
    assert response.json["improved_prompt"].startswith("Create a queue health report")
    assert response.json["changes"] == ["Added exact column names.", "Added a concrete threshold."]


def test_build_report_type_source_profile_respects_sample_percentage(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "col_a,col_b\n"
        "1,10\n"
        "2,20\n"
        "3,30\n"
        "4,40\n"
        "5,50\n"
        "6,60\n"
        "7,70\n"
        "8,80\n"
        "9,90\n"
        "10,100\n",
        encoding="utf-8",
    )

    profile = _build_report_type_source_profile(csv_path, sample_percent=20)

    assert profile["sampling"]["sample_percent"] == 20
    assert profile["sampling"]["total_row_count"] == 10
    assert profile["sampling"]["sampled_row_count"] == 2


def test_create_report_type_auto_generates_id(tmp_path: Path) -> None:
    report_types_dir = tmp_path / "report_types"
    plugin_root = tmp_path / "report_type_plugins"
    plugin_root.mkdir(parents=True, exist_ok=True)
    (plugin_root / "manifest.schema.yaml").write_text(
        Path("report_type_plugins/manifest.schema.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    app = create_app(
        {
            "TESTING": True,
            "REPORT_TYPES_DIR": str(report_types_dir),
            "PLUGIN_ROOT": str(plugin_root),
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()

    def _fake_agent_draft(**_: object) -> dict[str, object]:
        return {
            "report_type_id": "",
            "title": "Queue Health Snapshot",
            "family": "time_series",
            "domain": "networking",
            "mode": "trend_analysis",
            "description": "Queue Health Snapshot plugin.",
            "required_columns": ["timestamp", "queue", "drop_ratio"],
            "default_prefs": {},
            "prompt_instructions": "Describe queue health changes and risks.",
            "plugin_code": "from __future__ import annotations\n\nfrom typing import Any\n\nimport pandas as pd\n\ndef get_spec() -> dict[str, Any]:\n    return {\"metrics_profile\": \"queue_health_snapshot\", \"api_version\": 1, \"title\": \"Queue Health Snapshot\", \"description\": \"Queue Health Snapshot plugin.\"}\n\ndef build(df: pd.DataFrame, prefs: dict[str, Any], ctx: Any) -> dict[str, Any]:\n    return {\"rows\": int(len(df))}\n",
            "smoke_test_code": "from __future__ import annotations\n\nimport pandas as pd\n\nfrom rv_reporter.report_types.plugins import ReportPluginManager\n\ndef test_queue_health_snapshot_smoke_build() -> None:\n    manager = ReportPluginManager(plugin_dir=\"report_type_plugins\")\n    result = manager.compute_metrics(metrics_profile=\"queue_health_snapshot\", df=pd.DataFrame({\"drop_ratio\": [0.1]}), prefs={}, report_type_id=\"queue_health_snapshot\")\n    assert isinstance(result, dict)\n",
            "manifest_extensions": {},
        }

    from rv_reporter import web as web_module

    original = web_module._generate_report_type_agent_draft
    web_module._generate_report_type_agent_draft = _fake_agent_draft
    try:
        post_resp = client.post(
            "/report-types/new",
            data={"prompt_text": "Create a queue health snapshot report."},
            follow_redirects=True,
        )
    finally:
        web_module._generate_report_type_agent_draft = original
    assert post_resp.status_code == 200
    assert (report_types_dir / "queue_health_snapshot.yaml").exists()
    assert b"Review Report Type" in post_resp.data
    assert b"Queue Health Snapshot" in post_resp.data
    assert b"Preview Sample Report" in post_resp.data


def test_create_report_type_filters_required_columns_to_uploaded_source(tmp_path: Path) -> None:
    report_types_dir = tmp_path / "report_types"
    plugin_root = tmp_path / "report_type_plugins"
    upload_dir = tmp_path / "uploads"
    plugin_root.mkdir(parents=True, exist_ok=True)
    upload_dir.mkdir(parents=True, exist_ok=True)
    (plugin_root / "manifest.schema.yaml").write_text(
        Path("report_type_plugins/manifest.schema.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    source_csv = upload_dir / "snmp.csv"
    source_csv.write_text(
        "Entry OID,Device ID,InOctets\n1,router-a,100\n",
        encoding="utf-8",
    )
    app = create_app(
        {
            "TESTING": True,
            "REPORT_TYPES_DIR": str(report_types_dir),
            "PLUGIN_ROOT": str(plugin_root),
            "UPLOAD_FOLDER": str(upload_dir),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()

    def _fake_agent_draft(**_: object) -> dict[str, object]:
        return {
            "report_type_id": "snmp_performance",
            "title": "SNMP Performance",
            "family": "time_series",
            "domain": "networking",
            "mode": "trend_analysis",
            "description": "SNMP Performance plugin.",
            "required_columns": ["Entry OID", "Date And Time (UTC)", "Device ID", "InOctets"],
            "default_prefs": {},
            "prompt_instructions": "Summarize interface throughput.",
            "plugin_code": (
                "def get_spec():\n"
                "    return {'metrics_profile': 'snmp_performance', 'api_version': 1, 'title': 'SNMP Performance'}\n\n"
                "def build(df, prefs, ctx):\n"
                "    return {'columns': list(df.columns)}\n"
            ),
            "smoke_test_code": "def test_smoke():\n    assert True\n",
            "manifest_extensions": {},
        }

    from rv_reporter import web as web_module

    original = web_module._generate_report_type_agent_draft
    web_module._generate_report_type_agent_draft = _fake_agent_draft
    try:
        post_resp = client.post(
            "/report-types/new",
            data={
                "prompt_text": "Create an SNMP report.",
                "existing_csv_path": str(source_csv),
            },
            follow_redirects=True,
        )
    finally:
        web_module._generate_report_type_agent_draft = original

    assert post_resp.status_code == 200
    yaml_text = (report_types_dir / "snmp_performance.yaml").read_text(encoding="utf-8")
    assert "Entry OID" in yaml_text
    assert "Device ID" in yaml_text
    assert "InOctets" in yaml_text
    assert "Date And Time (UTC)" not in yaml_text


def test_create_report_type_replaces_bad_ai_smoke_test_with_canonical_template(tmp_path: Path) -> None:
    report_types_dir = tmp_path / "report_types"
    plugin_root = tmp_path / "report_type_plugins"
    plugin_root.mkdir(parents=True, exist_ok=True)
    (plugin_root / "manifest.schema.yaml").write_text(
        Path("report_type_plugins/manifest.schema.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    app = create_app(
        {
            "TESTING": True,
            "REPORT_TYPES_DIR": str(report_types_dir),
            "PLUGIN_ROOT": str(plugin_root),
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()

    def _fake_agent_draft(**_: object) -> dict[str, object]:
        return {
            "report_type_id": "snmp_perf",
            "title": "SNMP Perf",
            "family": "time_series",
            "domain": "networking",
            "mode": "trend_analysis",
            "description": "SNMP Perf plugin.",
            "required_columns": ["sample"],
            "default_prefs": {},
            "prompt_instructions": "Summarize throughput.",
            "plugin_code": (
                "from __future__ import annotations\n\n"
                "def get_spec():\n"
                "    return {'metrics_profile': 'snmp_perf', 'api_version': 1, 'title': 'SNMP Perf'}\n\n"
                "def build(df, prefs, ctx):\n"
                "    return {'ok': True}\n"
            ),
            "smoke_test_code": (
                "from reporter.plugins import ReportPluginManager\n"
                "def test_bad():\n"
                "    assert False\n"
            ),
            "manifest_extensions": {},
        }

    from rv_reporter import web as web_module

    original = web_module._generate_report_type_agent_draft
    web_module._generate_report_type_agent_draft = _fake_agent_draft
    try:
        post_resp = client.post(
            "/report-types/new",
            data={"prompt_text": "Create an SNMP perf report."},
            follow_redirects=True,
        )
    finally:
        web_module._generate_report_type_agent_draft = original

    assert post_resp.status_code == 200
    smoke_text = (plugin_root / "snmp_perf" / "tests" / "test_smoke.py").read_text(encoding="utf-8")
    assert "from rv_reporter.report_types.plugins import ReportPluginManager" in smoke_text
    assert "from reporter.plugins import ReportPluginManager" not in smoke_text


def test_publish_rewrites_broken_smoke_test_before_running(tmp_path: Path) -> None:
    report_types_dir = tmp_path / "report_types"
    plugin_root = tmp_path / "report_type_plugins"
    report_types_dir.mkdir(parents=True, exist_ok=True)
    plugin_root.mkdir(parents=True, exist_ok=True)

    report_type_id = "snmp_perf"
    (report_types_dir / f"{report_type_id}.yaml").write_text(
        "report_type_id: snmp_perf\nversion: '1.0.0'\ntitle: 'SNMP Perf'\nrequired_columns: [sample]\nmetrics_profile: snmp_perf\noutput_schema: {type: object}\n",
        encoding="utf-8",
    )
    (plugin_root / report_type_id / "tests").mkdir(parents=True, exist_ok=True)
    (plugin_root / report_type_id / "manifest.yaml").write_text(
        "plugin_id: snmp_perf\nmetrics_profile: snmp_perf\napi_version: 1\nversion: '1.0.0'\ntitle: 'SNMP Perf'\ndescription: 'SNMP Perf plugin.'\nfamily: time_series\ndomain: networking\nmode: trend_analysis\nentrypoint: plugin.py\nowner: ai-agent\ngenerator: openai_sdk\nstatus: draft\n",
        encoding="utf-8",
    )
    (plugin_root / report_type_id / "plugin.py").write_text(
        "from __future__ import annotations\n\n"
        "def get_spec():\n"
        "    return {'metrics_profile': 'snmp_perf', 'api_version': 1, 'title': 'SNMP Perf'}\n\n"
        "def build(df, prefs, ctx):\n"
        "    return {'ok': True}\n",
        encoding="utf-8",
    )
    (plugin_root / report_type_id / "tests" / "test_smoke.py").write_text(
        "from reporter.plugins import ReportPluginManager\n"
        "def test_bad_imports():\n"
        "    assert False\n",
        encoding="utf-8",
    )
    (plugin_root / "manifest.schema.yaml").write_text(
        Path("report_type_plugins/manifest.schema.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    app = create_app(
        {
            "TESTING": True,
            "REPORT_TYPES_DIR": str(report_types_dir),
            "PLUGIN_ROOT": str(plugin_root),
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()

    response = client.post(
        "/report-types/publish",
        data={"report_type_id": report_type_id},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Published report type" in response.data
    smoke_text = (plugin_root / report_type_id / "tests" / "test_smoke.py").read_text(encoding="utf-8")
    assert "from rv_reporter.report_types.plugins import ReportPluginManager" in smoke_text
    assert "from reporter.plugins import ReportPluginManager" not in smoke_text


def test_published_ai_report_type_is_visible_and_can_generate(tmp_path: Path) -> None:
    previous_visible = os.environ.get("GENERATION_HIDDEN_REPORT_TYPES")
    os.environ["GENERATION_HIDDEN_REPORT_TYPES"] = "some_other_hidden_type"
    try:
        report_types_dir = tmp_path / "report_types"
        plugin_root = tmp_path / "report_type_plugins"
        output_root = tmp_path / "outputs"
        plugin_root.mkdir(parents=True, exist_ok=True)
        (plugin_root / "manifest.schema.yaml").write_text(
            Path("report_type_plugins/manifest.schema.yaml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        app = create_app(
            {
                "TESTING": True,
                "REPORT_TYPES_DIR": str(report_types_dir),
                "PLUGIN_ROOT": str(plugin_root),
                "UPLOAD_FOLDER": str(tmp_path / "uploads"),
                "OUTPUT_FOLDER": str(output_root),
            }
        )
        client = app.test_client()

        def _fake_agent_draft(**_: object) -> dict[str, object]:
            return {
                "report_type_id": "custom_twamp_review",
                "title": "Custom TWAMP Review",
                "family": "time_series",
                "domain": "networking",
                "mode": "issue_detection",
                "description": "Custom TWAMP Review plugin.",
                "required_columns": ["DateTimeUTC", "DiscardRatePct_Emulated"],
                "default_prefs": {},
                "prompt_instructions": "Describe TWAMP risks and recommended actions.",
                "plugin_code": (
                    "from __future__ import annotations\n\n"
                    "from typing import Any\n\n"
                    "import pandas as pd\n\n"
                    "def get_spec() -> dict[str, Any]:\n"
                    "    return {\"metrics_profile\": \"custom_twamp_review\", \"api_version\": 1, \"title\": \"Custom TWAMP Review\", \"description\": \"Custom TWAMP Review plugin.\"}\n\n"
                    "def build(df: pd.DataFrame, prefs: dict[str, Any], ctx: Any) -> dict[str, Any]:\n"
                    "    return {\"rows\": int(len(df)), \"mean_discard_pct\": float(df[\"DiscardRatePct_Emulated\"].astype(float).mean())}\n"
                ),
                "smoke_test_code": (
                    "from __future__ import annotations\n\n"
                    "import pandas as pd\n\n"
                    "from rv_reporter.report_types.plugins import ReportPluginManager\n\n"
                    "def test_custom_twamp_review_smoke_build() -> None:\n"
                    "    manager = ReportPluginManager(plugin_dir=\"report_type_plugins\")\n"
                    "    result = manager.compute_metrics(metrics_profile=\"custom_twamp_review\", df=pd.DataFrame({\"DiscardRatePct_Emulated\": [0.5, 1.0]}), prefs={}, report_type_id=\"custom_twamp_review\")\n"
                    "    assert isinstance(result, dict)\n"
                ),
                "manifest_extensions": {},
            }

        from rv_reporter import web as web_module

        original = web_module._generate_report_type_agent_draft
        web_module._generate_report_type_agent_draft = _fake_agent_draft
        try:
            create_resp = client.post(
                "/report-types/new",
                data={
                    "prompt_text": "Create a TWAMP issue report.",
                    "existing_csv_path": "samples/ETX2i_twamp.csv",
                },
                follow_redirects=True,
            )
        finally:
            web_module._generate_report_type_agent_draft = original

        assert create_resp.status_code == 200
        publish_resp = client.post(
            "/report-types/publish",
            data={"report_type_id": "custom_twamp_review"},
            follow_redirects=True,
        )
        assert publish_resp.status_code == 200
        assert b"Published report type &#39;custom_twamp_review&#39;" in publish_resp.data
        assert b"Status:</strong> <span class=\"muted\">active" in publish_resp.data

        index_resp = client.get("/")
        assert index_resp.status_code == 200
        assert b"custom_twamp_review" in index_resp.data

        generate_resp = client.post(
            "/generate",
            data={
                "report_type_id": "custom_twamp_review",
                "provider": "local",
                "model": "local-metrics",
                "existing_csv_path": "samples/ETX2i_twamp.csv",
            },
            follow_redirects=True,
        )
        assert generate_resp.status_code == 200
        assert b"&#39;custom_twamp_review&#39; is not enabled for generation" not in generate_resp.data
    finally:
        if previous_visible is None:
            os.environ.pop("GENERATION_HIDDEN_REPORT_TYPES", None)
        else:
            os.environ["GENERATION_HIDDEN_REPORT_TYPES"] = previous_visible


def test_ai_created_type_gets_classification_default_prefs(tmp_path: Path) -> None:
    report_types_dir = tmp_path / "report_types"
    plugin_root = tmp_path / "report_type_plugins"
    plugin_root.mkdir(parents=True, exist_ok=True)
    (plugin_root / "manifest.schema.yaml").write_text(
        Path("report_type_plugins/manifest.schema.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    app = create_app(
        {
            "TESTING": True,
            "REPORT_TYPES_DIR": str(report_types_dir),
            "PLUGIN_ROOT": str(plugin_root),
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()

    def _fake_agent_draft(**_: object) -> dict[str, object]:
        return {
            "report_type_id": "generic_issue_monitor",
            "title": "Generic Issue Monitor",
            "family": "event",
            "domain": "operations",
            "mode": "issue_detection",
            "description": "Generic Issue Monitor plugin.",
            "required_columns": ["timestamp", "entity", "status"],
            "default_prefs": {},
            "prompt_instructions": "Describe operational problems and likely actions.",
            "plugin_code": "from __future__ import annotations\n\nfrom typing import Any\n\nimport pandas as pd\n\ndef get_spec() -> dict[str, Any]:\n    return {\"metrics_profile\": \"generic_issue_monitor\", \"api_version\": 1, \"title\": \"Generic Issue Monitor\", \"description\": \"Generic Issue Monitor plugin.\"}\n\ndef build(df: pd.DataFrame, prefs: dict[str, Any], ctx: Any) -> dict[str, Any]:\n    return {\"rows\": int(len(df))}\n",
            "smoke_test_code": "from __future__ import annotations\n\nimport pandas as pd\n\nfrom rv_reporter.report_types.plugins import ReportPluginManager\n\ndef test_generic_issue_monitor_smoke_build() -> None:\n    manager = ReportPluginManager(plugin_dir=\"report_type_plugins\")\n    result = manager.compute_metrics(metrics_profile=\"generic_issue_monitor\", df=pd.DataFrame({\"status\": [\"ok\"]}), prefs={}, report_type_id=\"generic_issue_monitor\")\n    assert isinstance(result, dict)\n",
            "manifest_extensions": {},
        }

    from rv_reporter import web as web_module

    original = web_module._generate_report_type_agent_draft
    web_module._generate_report_type_agent_draft = _fake_agent_draft
    try:
        client.post("/report-types/new", data={"prompt_text": "Create an issue detection report."}, follow_redirects=True)
    finally:
        web_module._generate_report_type_agent_draft = original

    import yaml

    payload = yaml.safe_load((report_types_dir / "generic_issue_monitor.yaml").read_text(encoding="utf-8"))
    assert payload["default_prefs"]["tone"] == "technical"
    assert payload["default_prefs"]["audience"] == "engineering"
    assert payload["default_prefs"]["focus"] == "anomalies"


def test_delete_custom_report_type_from_ui(tmp_path: Path) -> None:
    report_types_dir = tmp_path / "report_types"
    report_types_dir.mkdir(parents=True, exist_ok=True)
    (report_types_dir / "custom_temp.yaml").write_text(
        "report_type_id: custom_temp\n"
        "version: '1.0.0'\n"
        "title: 'Custom Temp'\n"
        "required_columns: [NE Name, Resource Name, Queue Block, Queue Number, Time, Dequeued (Bytes), Dequeued (Frames), Dropped (Bytes), Dropped (Frames)]\n"
        "metrics_profile: network_queue_congestion\n"
        "output_schema: {type: object, additionalProperties: false, required: [report_type_id, report_title, summary, sections, alerts, recommendations, tables, charts, metadata], properties: {report_type_id: {type: string}, report_title: {type: string}, summary: {type: string}, sections: {type: array, items: {type: object, additionalProperties: false, required: [title, body], properties: {title: {type: string}, body: {type: string}}}}, alerts: {type: array, items: {type: object, additionalProperties: false, required: [severity, message], properties: {severity: {type: string}, message: {type: string}}}}, recommendations: {type: array, items: {type: object, additionalProperties: false, required: [priority, action], properties: {priority: {type: string}, action: {type: string}}}}, tables: {type: array, items: {type: object, additionalProperties: true}}, charts: {type: array, items: {type: object, additionalProperties: true}}, metadata: {type: object, additionalProperties: true}}}\n",
        encoding="utf-8",
    )
    app = create_app(
        {
            "TESTING": True,
            "REPORT_TYPES_DIR": str(report_types_dir),
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()
    response = client.post(
        "/report-types/delete",
        data={"report_type_id": "custom_temp"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert not (report_types_dir / "custom_temp.yaml").exists()


def test_delete_protected_report_type_is_blocked(tmp_path: Path) -> None:
    report_types_dir = tmp_path / "report_types"
    report_types_dir.mkdir(parents=True, exist_ok=True)
    (report_types_dir / "twamp_session_health.yaml").write_text(
        "report_type_id: twamp_session_health\n"
        "version: '1.0.0'\n"
        "title: 'TWAMP Session Health'\n"
        "required_columns: [DateTimeUTC, DiscardRatePct_Emulated]\n"
        "metrics_profile: twamp_session_health\n"
        "output_schema: {type: object}\n",
        encoding="utf-8",
    )
    app = create_app(
        {
            "TESTING": True,
            "REPORT_TYPES_DIR": str(report_types_dir),
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()
    response = client.post(
        "/report-types/delete",
        data={"report_type_id": "twamp_session_health"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert (report_types_dir / "twamp_session_health.yaml").exists()


def test_report_types_page_shows_publish_and_update_dates(tmp_path: Path) -> None:
    report_types_dir = tmp_path / "report_types"
    plugin_root = tmp_path / "report_type_plugins"
    report_types_dir.mkdir(parents=True, exist_ok=True)
    plugin_root.mkdir(parents=True, exist_ok=True)

    report_type_id = "custom_report"
    (report_types_dir / f"{report_type_id}.yaml").write_text(
        "report_type_id: custom_report\nversion: '1.0.0'\ntitle: 'Custom Report'\nrequired_columns: [a]\nmetrics_profile: custom_report\noutput_schema: {type: object}\n",
        encoding="utf-8",
    )
    (plugin_root / report_type_id).mkdir(parents=True, exist_ok=True)
    (plugin_root / report_type_id / "manifest.yaml").write_text(
        "plugin_id: custom_report\napi_version: 1\nversion: '1.0.0'\nstatus: active\npublished_at: '2026-03-16T10:30:00+00:00'\nupdated_at: '2026-03-16T12:45:00+00:00'\n",
        encoding="utf-8",
    )

    app = create_app(
        {
            "TESTING": True,
            "REPORT_TYPES_DIR": str(report_types_dir),
            "PLUGIN_ROOT": str(plugin_root),
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()

    response = client.get("/report-types")

    assert response.status_code == 200
    assert b"Published" in response.data
    assert b"Updated" in response.data
    assert b"10:30" in response.data or b"12:30" in response.data or b"03-16" in response.data


def test_publish_sets_manifest_timestamps(tmp_path: Path) -> None:
    report_types_dir = tmp_path / "report_types"
    plugin_root = tmp_path / "report_type_plugins"
    report_types_dir.mkdir(parents=True, exist_ok=True)
    plugin_root.mkdir(parents=True, exist_ok=True)

    report_type_id = "publishable_report"
    (report_types_dir / f"{report_type_id}.yaml").write_text(
        "report_type_id: publishable_report\nversion: '1.0.0'\ntitle: 'Publishable Report'\nrequired_columns: [value]\nmetrics_profile: publishable_report\noutput_schema: {type: object}\n",
        encoding="utf-8",
    )
    (plugin_root / report_type_id / "tests").mkdir(parents=True, exist_ok=True)
    (plugin_root / report_type_id / "manifest.yaml").write_text(
        "plugin_id: publishable_report\nmetrics_profile: publishable_report\napi_version: 1\nversion: '1.0.0'\ntitle: 'Publishable Report'\ndescription: 'Publishable Report plugin.'\nfamily: time_series\ndomain: generic\nmode: trend_analysis\nentrypoint: plugin.py\nowner: ai-agent\ngenerator: openai_sdk\nstatus: draft\n",
        encoding="utf-8",
    )
    (plugin_root / report_type_id / "plugin.py").write_text(
        "def get_spec():\n    return {'metrics_profile': 'publishable_report', 'api_version': 1, 'title': 'Publishable Report'}\n"
        "def build(df, prefs, ctx):\n    return {}\n",
        encoding="utf-8",
    )
    (plugin_root / report_type_id / "tests" / "test_smoke.py").write_text(
        "def test_smoke():\n    assert True\n",
        encoding="utf-8",
    )
    (plugin_root / "manifest.schema.yaml").write_text(
        Path("report_type_plugins/manifest.schema.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    app = create_app(
        {
            "TESTING": True,
            "REPORT_TYPES_DIR": str(report_types_dir),
            "PLUGIN_ROOT": str(plugin_root),
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()

    response = client.post(
        "/report-types/publish",
        data={"report_type_id": report_type_id},
        follow_redirects=True,
    )

    assert response.status_code == 200
    manifest_text = (plugin_root / report_type_id / "manifest.yaml").read_text(encoding="utf-8")
    assert "published_at:" in manifest_text
    assert "updated_at:" in manifest_text


def test_rename_draft_report_type(tmp_path: Path) -> None:
    report_types_dir = tmp_path / "report_types"
    plugin_root = tmp_path / "report_type_plugins"
    plugin_root.mkdir(parents=True, exist_ok=True)
    report_types_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy manifest schema
    (plugin_root / "manifest.schema.yaml").write_text(
        Path("report_type_plugins/manifest.schema.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    
    # Create a draft report type
    old_id = "draft_temp_report"
    (report_types_dir / f"{old_id}.yaml").write_text(
        "report_type_id: draft_temp_report\n"
        "version: '1.0.0'\n"
        "title: 'Draft Temp Report'\n"
        "required_columns: [timestamp, value]\n"
        "metrics_profile: custom_profile\n"
        "output_schema: {type: object, additionalProperties: false, required: [report_type_id, report_title, summary, sections, alerts, recommendations, tables, charts, metadata], properties: {report_type_id: {type: string}, report_title: {type: string}, summary: {type: string}, sections: {type: array, items: {type: object}}, alerts: {type: array, items: {type: object}}, recommendations: {type: array, items: {type: object}}, tables: {type: array, items: {type: object}}, charts: {type: array, items: {type: object}}, metadata: {type: object}}}\n",
        encoding="utf-8",
    )
    
    # Create plugin directory
    plugin_dir = plugin_root / old_id / "tests"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_root / old_id / "manifest.yaml").write_text(
        "plugin_id: draft_temp_report\n"
        "api_version: 1\n"
        "version: '1.0.0'\n"
        "title: 'Draft Temp Report'\n"
        "status: draft\n",
        encoding="utf-8",
    )
    (plugin_root / old_id / "plugin.py").write_text(
        "def get_spec():\n    return {'metrics_profile': 'custom_profile', 'api_version': 1, 'title': 'Draft Temp Report'}\n"
        "def build(df, prefs, ctx):\n    return {}\n",
        encoding="utf-8",
    )
    
    app = create_app(
        {
            "TESTING": True,
            "REPORT_TYPES_DIR": str(report_types_dir),
            "PLUGIN_ROOT": str(plugin_root),
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()
    
    # Rename to new ID
    new_id = "better_report_name"
    response = client.post(
        "/report-types/rename",
        data={
            "current_report_type_id": old_id,
            "new_report_type_id": new_id,
        },
        follow_redirects=True,
    )
    
    assert response.status_code == 200
    # Old files should not exist
    assert not (report_types_dir / f"{old_id}.yaml").exists()
    assert not (plugin_root / old_id).exists()
    # New files should exist
    assert (report_types_dir / f"{new_id}.yaml").exists()
    assert (plugin_root / new_id / "manifest.yaml").exists()
    assert (plugin_root / new_id / "plugin.py").exists()
    assert "report_type_id: better_report_name" in (report_types_dir / f"{new_id}.yaml").read_text(encoding="utf-8")
    assert "plugin_id: better_report_name" in (plugin_root / new_id / "manifest.yaml").read_text(encoding="utf-8")
    # Should see success message and be on new draft page
    assert b"Renamed report type from" in response.data
    assert b"better_report_name" in response.data


def test_rename_to_existing_id_fails(tmp_path: Path) -> None:
    report_types_dir = tmp_path / "report_types"
    plugin_root = tmp_path / "report_type_plugins"
    plugin_root.mkdir(parents=True, exist_ok=True)
    report_types_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy manifest schema
    (plugin_root / "manifest.schema.yaml").write_text(
        Path("report_type_plugins/manifest.schema.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    
    # Create two draft report types
    (report_types_dir / "first_draft.yaml").write_text(
        "report_type_id: first_draft\nversion: '1.0.0'\ntitle: 'First Draft'\nrequired_columns: [a]\nmetrics_profile: custom\noutput_schema: {type: object, additionalProperties: false, required: [report_type_id, report_title, summary, sections, alerts, recommendations, tables, charts, metadata], properties: {report_type_id: {type: string}, report_title: {type: string}, summary: {type: string}, sections: {type: array, items: {type: object}}, alerts: {type: array, items: {type: object}}, recommendations: {type: array, items: {type: object}}, tables: {type: array, items: {type: object}}, charts: {type: array, items: {type: object}}, metadata: {type: object}}}\n",
        encoding="utf-8",
    )
    (report_types_dir / "second_draft.yaml").write_text(
        "report_type_id: second_draft\nversion: '1.0.0'\ntitle: 'Second Draft'\nrequired_columns: [a]\nmetrics_profile: custom\noutput_schema: {type: object, additionalProperties: false, required: [report_type_id, report_title, summary, sections, alerts, recommendations, tables, charts, metadata], properties: {report_type_id: {type: string}, report_title: {type: string}, summary: {type: string}, sections: {type: array, items: {type: object}}, alerts: {type: array, items: {type: object}}, recommendations: {type: array, items: {type: object}}, tables: {type: array, items: {type: object}}, charts: {type: array, items: {type: object}}, metadata: {type: object}}}\n",
        encoding="utf-8",
    )
    
    # Create plugin directories
    for draft_id in ["first_draft", "second_draft"]:
        plugin_dir = plugin_root / draft_id / "tests"
        plugin_dir.mkdir(parents=True, exist_ok=True)
        (plugin_root / draft_id / "manifest.yaml").write_text(
            f"plugin_id: {draft_id}\napi_version: 1\nversion: '1.0.0'\nstatus: draft\n",
            encoding="utf-8",
        )
        (plugin_root / draft_id / "plugin.py").write_text("def get_spec(): return {}\ndef build(df, prefs, ctx): return {}\n", encoding="utf-8")
    
    app = create_app(
        {
            "TESTING": True,
            "REPORT_TYPES_DIR": str(report_types_dir),
            "PLUGIN_ROOT": str(plugin_root),
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()
    
    # Try to rename first_draft to second_draft (which already exists)
    response = client.post(
        "/report-types/rename",
        data={
            "current_report_type_id": "first_draft",
            "new_report_type_id": "second_draft",
        },
        follow_redirects=True,
    )
    
    assert response.status_code == 200
    # Should get error message
    assert b"already exists" in response.data
    # Original files should still exist
    assert (report_types_dir / "first_draft.yaml").exists()
    assert (plugin_root / "first_draft").exists()


def test_rename_normalizes_user_input(tmp_path: Path) -> None:
    report_types_dir = tmp_path / "report_types"
    plugin_root = tmp_path / "report_type_plugins"
    plugin_root.mkdir(parents=True, exist_ok=True)
    report_types_dir.mkdir(parents=True, exist_ok=True)

    (plugin_root / "manifest.schema.yaml").write_text(
        Path("report_type_plugins/manifest.schema.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    old_id = "snmp_request_response_performance"
    (report_types_dir / f"{old_id}.yaml").write_text(
        "report_type_id: snmp_request_response_performance\n"
        "version: '1.0.0'\n"
        "title: 'SNMP Request Response Performance'\n"
        "required_columns: [timestamp, value]\n"
        "metrics_profile: snmp_request_response_performance\n"
        "output_schema: {type: object, additionalProperties: false, required: [report_type_id, report_title, summary, sections, alerts, recommendations, tables, charts, metadata], properties: {report_type_id: {type: string}, report_title: {type: string}, summary: {type: string}, sections: {type: array, items: {type: object}}, alerts: {type: array, items: {type: object}}, recommendations: {type: array, items: {type: object}}, tables: {type: array, items: {type: object}}, charts: {type: array, items: {type: object}}, metadata: {type: object}}}\n",
        encoding="utf-8",
    )
    (plugin_root / old_id / "tests").mkdir(parents=True, exist_ok=True)
    (plugin_root / old_id / "manifest.yaml").write_text(
        "plugin_id: snmp_request_response_performance\n"
        "api_version: 1\n"
        "version: '1.0.0'\n"
        "metrics_profile: snmp_request_response_performance\n"
        "status: draft\n",
        encoding="utf-8",
    )
    (plugin_root / old_id / "plugin.py").write_text(
        "def get_spec():\n"
        "    return {'metrics_profile': 'snmp_request_response_performance', 'api_version': 1, 'title': 'SNMP Request Response Performance'}\n\n"
        "def build(df, prefs, ctx):\n"
        "    return {'report_type_id': 'snmp_request_response_performance'}\n",
        encoding="utf-8",
    )
    (plugin_root / old_id / "tests" / "test_smoke.py").write_text(
        "def test_smoke():\n"
        "    assert 'snmp_request_response_performance' == 'snmp_request_response_performance'\n",
        encoding="utf-8",
    )

    app = create_app(
        {
            "TESTING": True,
            "REPORT_TYPES_DIR": str(report_types_dir),
            "PLUGIN_ROOT": str(plugin_root),
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()

    response = client.post(
        "/report-types/rename",
        data={
            "current_report_type_id": old_id,
            "new_report_type_id": "SNMP performance",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert not (report_types_dir / f"{old_id}.yaml").exists()
    assert (report_types_dir / "snmp_performance.yaml").exists()
    assert (plugin_root / "snmp_performance" / "manifest.yaml").exists()
    assert "report_type_id: snmp_performance" in (report_types_dir / "snmp_performance.yaml").read_text(encoding="utf-8")
    assert "metrics_profile: snmp_performance" in (report_types_dir / "snmp_performance.yaml").read_text(encoding="utf-8")
    assert "plugin_id: snmp_performance" in (plugin_root / "snmp_performance" / "manifest.yaml").read_text(encoding="utf-8")
    assert "metrics_profile: snmp_performance" in (plugin_root / "snmp_performance" / "manifest.yaml").read_text(encoding="utf-8")
    assert "snmp_performance" in (plugin_root / "snmp_performance" / "plugin.py").read_text(encoding="utf-8")
    assert b"normalized from &#39;SNMP performance&#39;" in response.data


def test_review_page_blocks_sample_when_source_columns_do_not_match(tmp_path: Path) -> None:
    report_types_dir = tmp_path / "report_types"
    plugin_root = tmp_path / "report_type_plugins"
    report_types_dir.mkdir(parents=True, exist_ok=True)
    plugin_root.mkdir(parents=True, exist_ok=True)

    report_type_id = "snmp_performance"
    (report_types_dir / f"{report_type_id}.yaml").write_text(
        "report_type_id: snmp_performance\n"
        "version: '1.0.0'\n"
        "title: 'SNMP Performance'\n"
        "required_columns: [Entry OID, InOctets, OutOctets]\n"
        "metrics_profile: snmp_performance\n"
        "output_schema: {type: object}\n",
        encoding="utf-8",
    )
    (plugin_root / report_type_id).mkdir(parents=True, exist_ok=True)
    (plugin_root / report_type_id / "manifest.yaml").write_text(
        "plugin_id: snmp_performance\n"
        "api_version: 1\n"
        "version: '1.0.0'\n"
        "status: active\n",
        encoding="utf-8",
    )
    (plugin_root / report_type_id / "plugin.py").write_text("def get_spec(): return {}\ndef build(df, prefs, ctx): return {}\n", encoding="utf-8")
    sample_csv = tmp_path / "sample.csv"
    sample_csv.write_text("timestamp,value\n1,2\n", encoding="utf-8")

    app = create_app(
        {
            "TESTING": True,
            "REPORT_TYPES_DIR": str(report_types_dir),
            "PLUGIN_ROOT": str(plugin_root),
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["report_type_review_contexts"] = {
            report_type_id: {"existing_csv_path": str(sample_csv)}
        }

    response = client.get(f"/report-types/view?report_type_id={report_type_id}")

    assert response.status_code == 200
    assert b"Sample generation is blocked because the stored source file does not match this draft." in response.data
    assert b"Missing required columns: Entry OID, InOctets, OutOctets." in response.data
    assert b"Generate Sample Report</button>" in response.data
    assert b"return_to_report_type_view" in response.data


def test_review_page_blocks_sample_when_source_preflight_fails(tmp_path: Path, monkeypatch) -> None:
    report_types_dir = tmp_path / "report_types"
    plugin_root = tmp_path / "report_type_plugins"
    report_types_dir.mkdir(parents=True, exist_ok=True)
    plugin_root.mkdir(parents=True, exist_ok=True)

    report_type_id = "snmp_performance"
    (report_types_dir / f"{report_type_id}.yaml").write_text(
        "report_type_id: snmp_performance\n"
        "version: '1.0.0'\n"
        "title: 'SNMP Performance'\n"
        "required_columns: [timestamp, value]\n"
        "metrics_profile: snmp_performance\n"
        "output_schema: {type: object}\n",
        encoding="utf-8",
    )
    (plugin_root / report_type_id).mkdir(parents=True, exist_ok=True)
    (plugin_root / report_type_id / "manifest.yaml").write_text(
        "plugin_id: snmp_performance\n"
        "api_version: 1\n"
        "version: '1.0.0'\n"
        "status: active\n",
        encoding="utf-8",
    )
    (plugin_root / report_type_id / "plugin.py").write_text("def get_spec(): return {}\ndef build(df, prefs, ctx): return {}\n", encoding="utf-8")
    sample_csv = tmp_path / "sample.csv"
    sample_csv.write_text("timestamp,value\n1,2\n", encoding="utf-8")

    app = create_app(
        {
            "TESTING": True,
            "REPORT_TYPES_DIR": str(report_types_dir),
            "PLUGIN_ROOT": str(plugin_root),
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["report_type_review_contexts"] = {
            report_type_id: {"existing_csv_path": str(sample_csv)}
        }

    monkeypatch.setattr(
        "rv_reporter.web.preflight_tabular_source",
        lambda *args, **kwargs: {
            "ok": False,
            "issues": ["Parsed data still contains metadata rows."],
            "warnings": [],
        },
    )

    response = client.get(f"/report-types/view?report_type_id={report_type_id}")

    assert response.status_code == 200
    assert b"could not be safely parsed" in response.data
    assert b"Parsed data still contains metadata rows." in response.data
    assert b"Generate Sample Report</button>" in response.data
    assert b"return_to_report_type_view" in response.data


def test_preview_sample_failure_returns_to_review_page(tmp_path: Path, monkeypatch) -> None:
    report_types_dir = tmp_path / "report_types"
    plugin_root = tmp_path / "report_type_plugins"
    report_types_dir.mkdir(parents=True, exist_ok=True)
    plugin_root.mkdir(parents=True, exist_ok=True)

    report_type_id = "snmp_performance"
    (report_types_dir / f"{report_type_id}.yaml").write_text(
        "report_type_id: snmp_performance\n"
        "version: '1.0.0'\n"
        "title: 'SNMP Performance'\n"
        "required_columns: [timestamp, value]\n"
        "metrics_profile: snmp_performance\n"
        "output_schema: {type: object}\n",
        encoding="utf-8",
    )
    (plugin_root / report_type_id).mkdir(parents=True, exist_ok=True)
    (plugin_root / report_type_id / "manifest.yaml").write_text(
        "plugin_id: snmp_performance\n"
        "api_version: 1\n"
        "version: '1.0.0'\n"
        "status: active\n",
        encoding="utf-8",
    )
    (plugin_root / report_type_id / "plugin.py").write_text(
        "def get_spec(): return {}\ndef build(df, prefs, ctx): return {}\n",
        encoding="utf-8",
    )
    sample_csv = tmp_path / "sample.csv"
    sample_csv.write_text("timestamp,value\n1,2\n", encoding="utf-8")

    app = create_app(
        {
            "TESTING": True,
            "REPORT_TYPES_DIR": str(report_types_dir),
            "PLUGIN_ROOT": str(plugin_root),
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["report_type_review_contexts"] = {
            report_type_id: {"existing_csv_path": str(sample_csv)}
        }

    monkeypatch.setattr(
        "rv_reporter.web.preflight_tabular_source",
        lambda *args, **kwargs: {
            "ok": False,
            "issues": ["Parsed data still contains metadata rows."],
            "warnings": [],
        },
    )

    response = client.post(
        "/generate",
        data={
            "report_type_id": report_type_id,
            "existing_csv_path": str(sample_csv),
            "provider": "local",
            "model": "local-metrics",
            "return_to_report_type_view": "1",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Review Report Type" in response.data
    assert b"Source preflight failed: Parsed data still contains metadata rows." in response.data
    assert b"Preview Sample Report" in response.data


def test_excel_sheets_api_for_non_excel_returns_empty() -> None:
    app = create_app({"TESTING": True})
    client = app.test_client()
    response = client.get("/api/excel-sheets?path=samples/network_queues.csv")
    assert response.status_code == 200
    assert response.json == {"sheets": []}


def test_generate_upload_excel_without_sheet_prompts_sheet_selection(tmp_path: Path, monkeypatch) -> None:
    app = create_app(
        {
            "TESTING": True,
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()

    def _fake_sheet_list(_path: str) -> list[str]:
        return ["Sheet1", "Sheet2"]

    monkeypatch.setattr("rv_reporter.web.list_excel_sheets", _fake_sheet_list)

    response = client.post(
        "/generate",
        data={
            "report_type_id": "network_queue_congestion",
            "provider": "local",
            "model": "gpt-4.1-mini",
            "csv_source": "upload",
            "csv_upload": (io.BytesIO(b"dummy"), "test.xlsx"),
            "tone": "technical",
            "audience": "engineering",
            "focus": "anomalies",
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Excel file has multiple sheets" in response.data
    assert b"Sheet1" in response.data
    assert b"Sheet2" in response.data


def test_upload_excel_sheets_api_returns_sheets(tmp_path: Path, monkeypatch) -> None:
    app = create_app(
        {
            "TESTING": True,
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()

    def _fake_sheet_list(_path: str) -> list[str]:
        return ["Main", "Summary"]

    monkeypatch.setattr("rv_reporter.web.list_excel_sheets", _fake_sheet_list)

    response = client.post(
        "/api/upload-excel-sheets",
        data={"file": (io.BytesIO(b"dummy"), "book.xlsx")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.json["sheets"] == ["Main", "Summary"]
    assert response.json["path"]
