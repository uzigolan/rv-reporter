import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _twamp_shared import build_twamp_sla_metrics


def get_spec():
    return {
        "metrics_profile": "telecom_session_health_sla_eval_v2",
        "api_version": 1,
        "title": "Telecom Session Health SLA Evaluation v2",
        "description": "Evaluate telecom session health metrics against predefined SLA thresholds, focusing on delay, jitter, and packet loss. Provides insights on session performance and compliance with SLAs.",
    }


def build(df, prefs, ctx):
    return build_twamp_sla_metrics(df, prefs)
