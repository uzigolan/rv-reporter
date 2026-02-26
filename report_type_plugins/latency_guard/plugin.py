from __future__ import annotations

from typing import Any

import pandas as pd


def get_spec() -> dict[str, Any]:
    return {
        "metrics_profile": "latency_guard",
        "api_version": 1,
        "title": "Latency Guard",
        "description": "Sample plugin profile for isolated report-type metrics logic.",
    }


def build(df: pd.DataFrame, prefs: dict[str, Any], ctx: Any) -> dict[str, Any]:
    working = df.copy()
    latency_col = str(prefs.get("latency_column", "latency_ms"))
    threshold_ms = float(prefs.get("alert_latency_ms", 200))

    if latency_col not in working.columns:
        return {
            "summary": {"rows": int(len(working)), "latency_column": latency_col, "available": False},
            "rows": [],
            "alerts": [
                {
                    "severity": "high",
                    "message": f"Column '{latency_col}' is required for latency_guard.",
                }
            ],
        }

    numeric = pd.to_numeric(working[latency_col], errors="coerce").dropna()
    if numeric.empty:
        return {
            "summary": {"rows": int(len(working)), "latency_column": latency_col, "available": False},
            "rows": [],
            "alerts": [{"severity": "medium", "message": "No numeric latency samples were found."}],
        }

    p95 = float(numeric.quantile(0.95))
    avg = float(numeric.mean())
    alerts = []
    if p95 > threshold_ms:
        alerts.append(
            {
                "severity": "high",
                "message": f"p95 latency {p95:.2f}ms exceeds threshold {threshold_ms:.2f}ms.",
            }
        )

    return {
        "summary": {
            "rows": int(len(working)),
            "samples": int(numeric.shape[0]),
            "avg_latency_ms": round(avg, 2),
            "p95_latency_ms": round(p95, 2),
            "threshold_ms": threshold_ms,
            "report_type_id": getattr(ctx, "report_type_id", ""),
        },
        "rows": [
            {"metric": "avg_latency_ms", "value": round(avg, 2)},
            {"metric": "p95_latency_ms", "value": round(p95, 2)},
        ],
        "alerts": alerts,
    }
