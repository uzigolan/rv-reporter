from pathlib import Path
import io

from rv_reporter.web import create_app


def test_index_renders() -> None:
    app = create_app({"TESTING": True})
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Tabular File to Structured Report" in response.data


def test_generate_from_sample(tmp_path: Path) -> None:
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
