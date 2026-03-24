import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _twamp_shared import build_twamp_sla_metrics


def get_spec():
    return {
        "metrics_profile": "telecom_session_health_thresold_sla",
        "api_version": 1,
        "title": "Telecom Session Health Threshold SLA Report",
        "description": "Evaluates telecom session health compliance against defined SLA thresholds for key metrics in time series data.",
    }


def build(df, prefs, ctx):
    return build_twamp_sla_metrics(df, prefs)
