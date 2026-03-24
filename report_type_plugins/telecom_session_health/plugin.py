import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _twamp_shared import build_twamp_sla_metrics


def get_spec():
    return {
        "metrics_profile": "telecom_session_health",
        "api_version": 1,
        "title": "Telecom Session Health",
        "description": "Analyze session-level time-series measurements to detect degraded sessions and signal radio/signal anomalies. Computes RSSI/SNR/handovers statistics, flags sessions below configured thresholds, and returns top degraded sessions for triage.",
    }


def build(df, prefs, ctx):
    return build_twamp_sla_metrics(df, prefs)
