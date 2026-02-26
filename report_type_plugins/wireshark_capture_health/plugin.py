from __future__ import annotations

from typing import Any

import pandas as pd

from rv_reporter.services.metrics import compute_legacy_metrics


def get_spec() -> dict[str, Any]:
    return {
        "metrics_profile": "wireshark_capture_health",
        "api_version": 1,
        "title": "Wireshark Capture Health",
        "description": "Wireshark report metrics plugin.",
    }


def build(df: pd.DataFrame, prefs: dict[str, Any], ctx: Any) -> dict[str, Any]:
    # This plugin boundary isolates Wireshark logic from the core dispatcher.
    return compute_legacy_metrics("wireshark_capture_health", df, prefs)
