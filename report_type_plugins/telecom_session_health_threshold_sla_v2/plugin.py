import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _twamp_shared import build_twamp_sla_metrics


def get_spec():
    return {
        "metrics_profile": "telecom_session_health_threshold_sla_v2",
        "api_version": 1,
        "title": "Telecom Session Health SLA Threshold Analysis v2",
        "description": "Evaluates telecom session compliance with SLA thresholds for latency, jitter, and packet loss with configurable focus metrics and aggregate analysis.",
    }


def build(df, prefs, ctx):
    return build_twamp_sla_metrics(df, prefs)
