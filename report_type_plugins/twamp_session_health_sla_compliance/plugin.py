import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _twamp_shared import build_twamp_sla_metrics


def get_spec():
    return {
        "metrics_profile": "twamp_session_health_sla_compliance",
        "api_version": 1,
        "title": "TWAMP Session Health SLA Compliance",
        "description": "Evaluates TWAMP session health against SLA thresholds.",
    }


def build(df, prefs, ctx):
    return build_twamp_sla_metrics(df, prefs)
