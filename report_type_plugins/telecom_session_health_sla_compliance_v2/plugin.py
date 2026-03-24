import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _twamp_shared import build_twamp_sla_metrics


def get_spec():
    return {
        "metrics_profile": "telecom_session_health_sla_compliance_v2",
        "api_version": 1,
        "title": "Telecom Session Health SLA Compliance v2",
        "description": "Analyzes telecom session data to assess compliance with SLA thresholds.",
    }


def build(df, prefs, ctx):
    return build_twamp_sla_metrics(df, prefs)
