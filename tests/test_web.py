from pathlib import Path
import io
import json
from urllib.parse import urlencode

from rv_reporter.web import create_app, _parse_source_labels_text, _prepare_pipeline_source, _resolve_source_label


def test_index_renders() -> None:
    app = create_app({"TESTING": True})
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Tabular File to Structured Report" in response.data


def test_about_includes_doc_links() -> None:
    app = create_app({"TESTING": True})
    client = app.test_client()
    response = client.get("/about")
    assert response.status_code == 200
    assert b"/docs/install" in response.data
    assert b"/docs/architecture" in response.data
    assert b"/docs/ui-guide" in response.data


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
            "report_type_id": "network_queue_congestion",
            "provider": "local",
            "model": "gpt-4.1-mini",
            "csv_source": "sample",
            "sample_csv": "samples/network_queues.csv",
            "tone": "technical",
            "audience": "engineering",
            "focus": "anomalies",
            "threshold_name": "alert_drop_ratio",
            "threshold_value": "0.2",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"report generated" in response.data.lower()
    report_files = sorted((output_root / "network_queue_congestion").glob("*.report.json"))
    assert report_files
    payload = json.loads(report_files[-1].read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    assert "generation_duration_seconds" in metadata


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
            "report_type_id": "network_queue_congestion",
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "csv_source": "sample",
            "sample_csv": "samples/network_queues.csv",
            "tone": "technical",
            "audience": "engineering",
            "focus": "anomalies",
            "row_limit": "8",
            "output_token_budget": "1000",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"OpenAI Cost Estimate" in response.data


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
    app = create_app(
        {
            "TESTING": True,
            "REPORT_TYPES_DIR": str(report_types_dir),
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()
    get_resp = client.get("/report-types/new")
    assert get_resp.status_code == 200

    yaml_body = """
report_type_id: temp_custom_report
version: "1.0.0"
title: "Temp Custom Report"
required_columns: [NE Name, Resource Name, Queue Block, Queue Number, Time, Dequeued (Bytes), Dequeued (Frames), Dropped (Bytes), Dropped (Frames)]
metrics_profile: network_queue_congestion
default_prefs:
  tone: concise
  audience: leadership
  focus: trends
prompt_instructions: "test"
output_schema:
  type: object
  additionalProperties: false
  required: [report_type_id, report_title, summary, sections, alerts, recommendations, tables, charts, metadata]
  properties:
    report_type_id: {type: string}
    report_title: {type: string}
    summary: {type: string}
    sections:
      type: array
      items:
        type: object
        additionalProperties: false
        required: [title, body]
        properties:
          title: {type: string}
          body: {type: string}
    alerts:
      type: array
      items:
        type: object
        additionalProperties: false
        required: [severity, message]
        properties:
          severity: {type: string}
          message: {type: string}
    recommendations:
      type: array
      items:
        type: object
        additionalProperties: false
        required: [priority, action]
        properties:
          priority: {type: string}
          action: {type: string}
    tables:
      type: array
      items: {type: object, additionalProperties: true}
    charts:
      type: array
      items: {type: object, additionalProperties: true}
    metadata: {type: object, additionalProperties: true}
""".strip()
    post_resp = client.post(
        "/report-types/new",
        data={"report_type_yaml": yaml_body},
        follow_redirects=True,
    )
    assert post_resp.status_code == 200
    assert (report_types_dir / "temp_custom_report.yaml").exists()


def test_create_report_type_auto_generates_id(tmp_path: Path) -> None:
    report_types_dir = tmp_path / "report_types"
    app = create_app(
        {
            "TESTING": True,
            "REPORT_TYPES_DIR": str(report_types_dir),
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "OUTPUT_FOLDER": str(tmp_path / "outputs"),
        }
    )
    client = app.test_client()
    yaml_body = """
version: "1.0.0"
title: "Queue Health Snapshot"
required_columns: [NE Name, Resource Name, Queue Block, Queue Number, Time, Dequeued (Bytes), Dequeued (Frames), Dropped (Bytes), Dropped (Frames)]
metrics_profile: network_queue_congestion
output_schema:
  type: object
  additionalProperties: false
  required: [report_type_id, report_title, summary, sections, alerts, recommendations, tables, charts, metadata]
  properties:
    report_type_id: {type: string}
    report_title: {type: string}
    summary: {type: string}
    sections:
      type: array
      items:
        type: object
        additionalProperties: false
        required: [title, body]
        properties:
          title: {type: string}
          body: {type: string}
    alerts:
      type: array
      items:
        type: object
        additionalProperties: false
        required: [severity, message]
        properties:
          severity: {type: string}
          message: {type: string}
    recommendations:
      type: array
      items:
        type: object
        additionalProperties: false
        required: [priority, action]
        properties:
          priority: {type: string}
          action: {type: string}
    tables:
      type: array
      items: {type: object, additionalProperties: true}
    charts:
      type: array
      items: {type: object, additionalProperties: true}
    metadata: {type: object, additionalProperties: true}
""".strip()
    post_resp = client.post(
        "/report-types/new",
        data={"report_type_yaml": yaml_body},
        follow_redirects=True,
    )
    assert post_resp.status_code == 200
    assert (report_types_dir / "queue_health_snapshot.yaml").exists()


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
