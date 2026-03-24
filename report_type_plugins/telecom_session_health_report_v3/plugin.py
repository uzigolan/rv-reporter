import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _twamp_shared import build_twamp_sla_metrics


def get_spec():
    return {
        "metrics_profile": "telecom_session_health_report_v3",
        "api_version": 1,
        "title": "Telecom Session Health Report v3",
        "description": "Comprehensive telecom session health analysis with SLA compliance, signal quality, and SNR metrics for network performance monitoring.",
    }


def build(df, prefs, ctx):
    return build_twamp_sla_metrics(df, prefs)
