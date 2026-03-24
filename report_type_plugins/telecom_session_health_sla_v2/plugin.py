import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _twamp_shared import build_twamp_sla_metrics


def get_spec():
    return {
        "metrics_profile": "telecom_session_health_sla_v2",
        "api_version": 1,
        "title": "Telecom Session Health SLA v2",
        "description": "This report analyzes telecom session data to assess compliance with defined Service Level Agreements (SLAs). It evaluates session parameters against SLA thresholds to identify performance issues.",
    }


def build(df, prefs, ctx):
    return build_twamp_sla_metrics(df, prefs)
