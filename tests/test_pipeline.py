import json
import re

from rv_reporter.orchestrator import run_pipeline
from rv_reporter.providers.base import ReportProvider
from rv_reporter.report_types.registry import ReportTypeDefinition


def test_pipeline_network_queue_generates_outputs(tmp_path) -> None:
    json_path, html_path = run_pipeline(
        csv_path="samples/network_queues.csv",
        report_type_id="network_queue_congestion",
        user_prefs={"alert_drop_ratio": 0.2},
        output_dir=tmp_path,
    )
    assert json_path.exists()
    assert html_path.exists()
    assert json_path.with_name(json_path.name.replace(".report.json", ".report.pdf")).exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["report_type_id"] == "network_queue_congestion"
    assert isinstance(payload["alerts"], list)
    assert "generated_at_utc" in payload["metadata"]
    assert "report_id" in payload["metadata"]
    assert re.search(r"\.\d{6}_\d{4}_\d{6}\.report\.json$", str(json_path).replace("\\", "/"))
    html = html_path.read_text(encoding="utf-8")
    assert "Top Congested Queues" in html
    assert "Drop Ratio Trend" in html
    assert "L2/QoS Insights" in html
    assert "Hotspot Intervals" in html


def test_pipeline_with_row_limit(tmp_path) -> None:
    json_path, _ = run_pipeline(
        csv_path="samples/network_queues.csv",
        report_type_id="network_queue_congestion",
        output_dir=tmp_path,
        row_limit=4,
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    metrics_payload = payload["tables"][0]["rows"][0]
    assert metrics_payload["summary"]["interval_samples"] <= 4


def test_pipeline_twamp_generates_outputs(tmp_path) -> None:
    json_path, html_path = run_pipeline(
        csv_path="samples/ETX2i_twamp.csv",
        report_type_id="twamp_session_health",
        output_dir=tmp_path,
        row_limit=16,
    )
    assert json_path.exists()
    assert html_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["report_type_id"] == "twamp_session_health"
    assert "generated_at_utc" in payload["metadata"]
    html = html_path.read_text(encoding="utf-8")
    assert "TWAMP Health Charts" in html
    assert "Top Discard Hotspots" in html
    assert "L2/QoS Insights" in html
    assert "Risk Score" in html


class _SparseOpenAIStyleProvider(ReportProvider):
    def generate_report_json(self, definition: ReportTypeDefinition, csv_profile, metrics, user_prefs):  # noqa: ANN001
        return {
            "report_type_id": definition.report_type_id,
            "report_title": definition.title,
            "summary": "model output without tables/charts",
            "sections": [{"title": "x", "body": "y"}],
            "alerts": [],
            "recommendations": [],
            "metadata": {},
        }


def test_pipeline_normalizes_missing_required_report_keys(tmp_path) -> None:
    json_path, _ = run_pipeline(
        csv_path="samples/network_queues.csv",
        report_type_id="network_queue_congestion",
        output_dir=tmp_path,
        provider=_SparseOpenAIStyleProvider(),
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "tables" in payload and isinstance(payload["tables"], list)
    assert "charts" in payload and isinstance(payload["charts"], list)


def test_pipeline_pm_export_health_generates_outputs(tmp_path) -> None:
    json_path, html_path = run_pipeline(
        csv_path="samples/pm-csv-es.csv",
        report_type_id="pm_export_health",
        output_dir=tmp_path,
    )
    assert json_path.exists()
    assert html_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["report_type_id"] == "pm_export_health"
    metrics_payload = payload["tables"][0]["rows"][0]
    assert "summary" in metrics_payload
    assert "table_counts" in metrics_payload
    html = html_path.read_text(encoding="utf-8")
    assert "PM Health Charts" in html
    assert "Top Interface" in html
