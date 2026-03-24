import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _twamp_shared import build_twamp_sla_metrics


def get_spec():
    return {
        "metrics_profile": "telecom_session_health_sla_evaluation_v3",
        "api_version": 1,
        "title": "Telecom Session Health SLA Evaluation v3",
        "description": "Evaluates session health against SLA thresholds for delay, jitter, and packet loss.",
    }


def build(df, prefs, ctx):
    return build_twamp_sla_metrics(df, prefs)
