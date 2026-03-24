import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _assisted_living_shared import build_assisted_living_metrics


def get_spec():
    return {
        "metrics_profile": "assisted_living_sla_compliance_analysis",
        "api_version": 1,
        "title": "Assisted Living SLA Compliance Analysis",
        "description": "Analyzes assisted living incidents for SLA compliance, risk scoring, and staffing optimization.",
    }


def build(df, prefs, ctx):
    return build_assisted_living_metrics(df, prefs)
