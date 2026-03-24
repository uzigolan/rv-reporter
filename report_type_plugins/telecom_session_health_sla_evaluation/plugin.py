import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _twamp_shared import build_twamp_sla_metrics


def get_spec():
    return {
        "metrics_profile": "telecom_session_health_sla_evaluation",
        "api_version": 1,
        "title": "Telecom Session Health SLA Evaluation",
        "description": "Evaluates the compliance of telecom sessions against defined SLA thresholds for delay, jitter, packet loss, and other performance metrics.",
    }


def build(df, prefs, ctx):
    return build_twamp_sla_metrics(df, prefs)
