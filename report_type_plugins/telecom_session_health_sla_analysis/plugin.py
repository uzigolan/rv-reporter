import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _twamp_shared import build_twamp_sla_metrics


def get_spec():
    return {
        "metrics_profile": "telecom_session_health_sla_analysis",
        "api_version": 1,
        "title": "Telecom Session Health SLA Analysis",
        "description": "This report evaluates telecom session health metrics against defined SLA thresholds to detect compliance and performance issues.",
    }


def build(df, prefs, ctx):
    return build_twamp_sla_metrics(df, prefs)
