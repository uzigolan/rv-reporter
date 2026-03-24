import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _twamp_shared import build_twamp_sla_metrics


def get_spec():
    return {
        "metrics_profile": "telecom_session_sla_compliance",
        "api_version": 1,
        "title": "Telecom Session SLA Compliance Report",
        "description": "This report evaluates telecom session data against predefined SLA thresholds to identify compliance issues and potential degradations based on key performance metrics over specific time windows.",
    }


def build(df, prefs, ctx):
    return build_twamp_sla_metrics(df, prefs)
