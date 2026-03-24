"""Shared TWAMP SLA analytics used by all TWAMP/telecom session plugins.

Every TWAMP plugin that operates on PM CSV-ES data with twampReportCurrent*
columns can call ``build_twamp_sla_metrics(df, prefs)`` from its ``build()``
function instead of reimplementing the same analysis.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

_SENTINEL = 4294967295  # 0xFFFFFFFF — "not available" in PM CSV-ES exports


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convert to numeric and replace PM CSV-ES sentinel values with NaN."""
    vals = pd.to_numeric(series, errors="coerce")
    return vals.replace(_SENTINEL, float("nan"))


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a safe-numeric column, or zeros if the column is missing."""
    if name in df.columns:
        return _safe_numeric(df[name])
    return pd.Series(0.0, index=df.index)


def build_twamp_sla_metrics(df: pd.DataFrame, prefs: dict[str, Any]) -> dict[str, Any]:
    """Full TWAMP SLA analysis — summary, compliance, per-interval detail,
    charts, and alerts.  Returned dict is compatible with MockProvider
    forwarding (keys: summary, compliance, per_interval, charts, alerts).
    """
    work = df.copy()
    sla = prefs.get("sla_thresholds", prefs.get("state_definitions", {}))

    # ── Thresholds (accept several naming conventions) ───────────────
    lat_good = float(sla.get("latency_good", sla.get("delay_degraded", sla.get("degraded", {}).get("latency", 10))))
    lat_crit = float(sla.get("latency_critical", sla.get("delay_critical", sla.get("critical", {}).get("latency", 50))))
    jit_good = float(sla.get("jitter_good", sla.get("jitter_degraded", sla.get("degraded", {}).get("jitter", 5))))
    jit_crit = float(sla.get("jitter_critical", sla.get("critical", {}).get("jitter", 20)))
    loss_good = float(sla.get("packet_loss_good", sla.get("packet_loss_degraded", sla.get("degraded", {}).get("packet_loss", 0))))
    loss_crit = float(sla.get("packet_loss_critical", sla.get("critical", {}).get("packet_loss", 1)))

    # ── Numeric columns ──────────────────────────────────────────────
    delay_avg = _col(work, "twampReportCurrentDelayAverage")
    delay_min = _col(work, "twampReportCurrentDelayMin")
    delay_max = _col(work, "twampReportCurrentDelayMax")
    pdv_max = _col(work, "twampReportCurrentPdvMax")
    ipdv_max = _col(work, "twampReportCurrentIpdvMax")
    tx_pkts = _col(work, "twampReportCurrentTxPackets")
    rx_pkts = _col(work, "twampReportCurrentRxValidPackets")
    loss_pkts = _col(work, "twampReportCurrentLossPackets")
    elapsed = _col(work, "twampReportCurrentElapsedTime")
    delay_fwd_max = _col(work, "twampReportCurrentDelayFwdMax")
    delay_bck_max = _col(work, "twampReportCurrentDelayBckMax")

    total_intervals = len(work)
    active_mask = tx_pkts > 0
    active_count = int(active_mask.sum())
    idle_count = total_intervals - active_count

    # ── Derived metrics ──────────────────────────────────────────────
    loss_pct = (loss_pkts / tx_pkts.replace(0, float("nan")) * 100).fillna(0.0)
    jitter_combined = pdv_max.combine(ipdv_max, max).fillna(0.0)

    # ── Per-interval classification ──────────────────────────────────
    def _classify(val: float, good_th: float, crit_th: float) -> str:
        if pd.isna(val):
            return "no_data"
        if val <= good_th:
            return "good"
        if val <= crit_th:
            return "degraded"
        return "critical"

    lat_status = delay_avg.apply(lambda v: _classify(v, lat_good, lat_crit))
    jit_status = jitter_combined.apply(lambda v: _classify(v, jit_good, jit_crit))
    loss_status = loss_pct.apply(lambda v: _classify(v, loss_good, loss_crit))

    def _counts(series: pd.Series) -> dict[str, int]:
        c = series.value_counts().to_dict()
        return {k: int(c.get(k, 0)) for k in ("good", "degraded", "critical", "no_data")}

    # ── Overall verdict ──────────────────────────────────────────────
    worst = "good"
    for s in (lat_status, jit_status, loss_status):
        if (s == "critical").any():
            worst = "critical"
            break
        if (s == "degraded").any():
            worst = "degraded"

    compliant_count = int(
        ((lat_status == "good") & (jit_status == "good") & (loss_status == "good")).sum()
    )
    sla_compliance_pct = round(compliant_count / max(1, total_intervals) * 100, 1)

    # ── Summary ──────────────────────────────────────────────────────
    session_col = "twampContSessionId" if "twampContSessionId" in work.columns else None
    controller_col = "twampControllerId" if "twampControllerId" in work.columns else None
    peer_col = "twampPeerAddr" if "twampPeerAddr" in work.columns else None

    summary: dict[str, Any] = {
        "total_intervals": total_intervals,
        "active_intervals": active_count,
        "idle_intervals": idle_count,
        "unique_sessions": int(work[session_col].nunique()) if session_col else 0,
        "unique_controllers": int(work[controller_col].nunique()) if controller_col else 0,
        "unique_peers": int(work[peer_col].nunique()) if peer_col else 0,
        "sla_compliance_pct": sla_compliance_pct,
        "overall_verdict": worst,
        "delay_avg_mean": round(float(delay_avg.mean()), 2) if delay_avg.notna().any() else 0.0,
        "delay_avg_max": round(float(delay_avg.max()), 2) if delay_avg.notna().any() else 0.0,
        "delay_max_peak": round(float(delay_max.max()), 2) if delay_max.notna().any() else 0.0,
        "jitter_combined_peak": round(float(jitter_combined.max()), 2),
        "ipdv_max_peak": round(float(ipdv_max.max()), 2) if ipdv_max.notna().any() else 0.0,
        "total_tx_packets": int(tx_pkts.sum()),
        "total_rx_packets": int(rx_pkts.sum()),
        "total_lost_packets": int(loss_pkts.sum()),
        "loss_pct_mean": round(float(loss_pct.mean()), 4),
        "total_elapsed_seconds": int(elapsed.sum()),
    }

    # ── Compliance breakdown ─────────────────────────────────────────
    compliance: dict[str, dict[str, int]] = {
        "latency": _counts(lat_status),
        "jitter": _counts(jit_status),
        "packet_loss": _counts(loss_status),
    }

    # ── Per-interval detail rows ─────────────────────────────────────
    per_interval: list[dict[str, Any]] = []
    for i in range(total_intervals):
        row: dict[str, Any] = {"interval": i + 1}
        if session_col:
            row["session_id"] = str(work.iloc[i][session_col])
        row.update({
            "elapsed_sec": int(elapsed.iloc[i]) if pd.notna(elapsed.iloc[i]) else 0,
            "tx_packets": int(tx_pkts.iloc[i]) if pd.notna(tx_pkts.iloc[i]) else 0,
            "rx_packets": int(rx_pkts.iloc[i]) if pd.notna(rx_pkts.iloc[i]) else 0,
            "loss_packets": int(loss_pkts.iloc[i]) if pd.notna(loss_pkts.iloc[i]) else 0,
            "loss_pct": round(float(loss_pct.iloc[i]), 4),
            "delay_avg": round(float(delay_avg.iloc[i]), 2) if pd.notna(delay_avg.iloc[i]) else None,
            "delay_max": round(float(delay_max.iloc[i]), 2) if pd.notna(delay_max.iloc[i]) else None,
            "jitter_combined": round(float(jitter_combined.iloc[i]), 2),
            "latency_status": str(lat_status.iloc[i]),
            "jitter_status": str(jit_status.iloc[i]),
            "loss_status": str(loss_status.iloc[i]),
        })
        per_interval.append(row)

    # ── Charts ───────────────────────────────────────────────────────
    charts: list[dict[str, Any]] = []

    # 1. Compliance doughnut
    comp_good = sum(c["good"] for c in compliance.values())
    comp_deg = sum(c["degraded"] for c in compliance.values())
    comp_crit = sum(c["critical"] for c in compliance.values())
    comp_nd = sum(c["no_data"] for c in compliance.values())
    charts.append({
        "chart_type": "doughnut",
        "title": "SLA Compliance Distribution",
        "labels": ["Good", "Degraded", "Critical", "No Data"],
        "datasets": [{"data": [comp_good, comp_deg, comp_crit, comp_nd],
                       "backgroundColor": ["#4caf50", "#ff9800", "#f44336", "#9e9e9e"]}],
    })

    # 2. Per-metric stacked bar
    charts.append({
        "chart_type": "bar",
        "title": "Compliance by Metric",
        "labels": ["Latency", "Jitter", "Packet Loss"],
        "datasets": [
            {"label": "Good", "data": [compliance["latency"]["good"], compliance["jitter"]["good"], compliance["packet_loss"]["good"]], "backgroundColor": "#4caf50"},
            {"label": "Degraded", "data": [compliance["latency"]["degraded"], compliance["jitter"]["degraded"], compliance["packet_loss"]["degraded"]], "backgroundColor": "#ff9800"},
            {"label": "Critical", "data": [compliance["latency"]["critical"], compliance["jitter"]["critical"], compliance["packet_loss"]["critical"]], "backgroundColor": "#f44336"},
        ],
    })

    int_labels = [f"Int {i + 1}" for i in range(total_intervals)]

    # 3. Delay trend
    charts.append({
        "chart_type": "line",
        "title": "Delay Trend Across Intervals",
        "labels": int_labels,
        "datasets": [
            {"label": "Avg Delay", "data": [round(float(v), 2) if pd.notna(v) else None for v in delay_avg], "borderColor": "#1976d2"},
            {"label": "Max Delay", "data": [round(float(v), 2) if pd.notna(v) else None for v in delay_max], "borderColor": "#e65100"},
            {"label": f"SLA Critical ({lat_crit})", "data": [lat_crit] * total_intervals, "borderColor": "#f44336", "borderDash": [5, 5]},
        ],
    })

    # 4. Jitter + loss trend
    charts.append({
        "chart_type": "line",
        "title": "Jitter & Packet Loss Trend",
        "labels": int_labels,
        "datasets": [
            {"label": "Jitter (combined)", "data": [round(float(v), 2) for v in jitter_combined], "borderColor": "#7b1fa2"},
            {"label": "Loss %", "data": [round(float(v), 4) for v in loss_pct], "borderColor": "#f44336"},
        ],
    })

    # 5. Packet volume bars
    charts.append({
        "chart_type": "bar",
        "title": "Packet Volume per Interval",
        "labels": int_labels,
        "datasets": [
            {"label": "TX", "data": [int(v) if pd.notna(v) else 0 for v in tx_pkts], "backgroundColor": "#1976d2"},
            {"label": "RX", "data": [int(v) if pd.notna(v) else 0 for v in rx_pkts], "backgroundColor": "#4caf50"},
            {"label": "Lost", "data": [int(v) if pd.notna(v) else 0 for v in loss_pkts], "backgroundColor": "#f44336"},
        ],
    })

    # ── Alerts ───────────────────────────────────────────────────────
    alerts: list[dict[str, str]] = []
    if idle_count == total_intervals:
        alerts.append({"severity": "warning", "message": f"All {total_intervals} intervals have zero TX packets — sessions appear idle or not yet started."})
    elif idle_count > 0:
        alerts.append({"severity": "warning", "message": f"{idle_count} of {total_intervals} intervals have zero TX packets (idle)."})
    if (lat_status == "critical").any():
        n = int((lat_status == "critical").sum())
        alerts.append({"severity": "critical", "message": f"{n} interval(s) exceed critical latency threshold ({lat_crit})."})
    if (jit_status == "critical").any():
        n = int((jit_status == "critical").sum())
        alerts.append({"severity": "critical", "message": f"{n} interval(s) exceed critical jitter threshold ({jit_crit})."})
    if (loss_status == "critical").any():
        n = int((loss_status == "critical").sum())
        alerts.append({"severity": "critical", "message": f"{n} interval(s) exceed critical packet-loss threshold ({loss_crit}%)."})
    if sla_compliance_pct < 100 and worst == "degraded":
        alerts.append({"severity": "warning", "message": f"SLA compliance is {sla_compliance_pct}% — some intervals are degraded."})
    if not alerts:
        alerts.append({"severity": "info", "message": "All intervals are within SLA thresholds."})

    return {
        "summary": summary,
        "compliance": compliance,
        "per_interval": per_interval,
        "charts": charts,
        "alerts": alerts,
    }
