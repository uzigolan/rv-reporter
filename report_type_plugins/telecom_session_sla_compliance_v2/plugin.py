import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _twamp_shared import build_twamp_sla_metrics


def get_spec():
    return {
        "metrics_profile": "telecom_session_sla_compliance_v2",
        "api_version": 1,
        "title": "Telecom Session SLA Compliance v2",
        "description": "Evaluates telecom session metrics against specified SLA thresholds to determine session health and compliance.",
    }


def build(df, prefs, ctx):
    return build_twamp_sla_metrics(df, prefs)
