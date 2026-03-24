import pandas as pd


_SENTINEL = 4294967295  # 0xFFFFFFFF — "not available" in PM CSV-ES


def get_spec():
    return {
        "metrics_profile": "twamp_session_threshold_sla_compliance",
        "api_version": 1,
        "title": "TWAMP Session Threshold SLA Compliance",
        "description": "Evaluate telecom session health using SLA thresholds for delay, jitter, and packet loss.",
    }


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convert to numeric and replace PM CSV-ES sentinel values with NaN."""
    vals = pd.to_numeric(series, errors="coerce")
    return vals.replace(_SENTINEL, float("nan"))


def build(df, prefs, ctx):
    work = df.copy()

    # ── Numeric conversion ───────────────────────────────────────────
    delay_avg = _safe_numeric(work["twampReportCurrentDelayAverage"])
    delay_min = _safe_numeric(work["twampReportCurrentDelayMin"])
    delay_max = _safe_numeric(work["twampReportCurrentDelayMax"])
    pdv_max = _safe_numeric(work["twampReportCurrentPdvMax"])
    ipdv_max = _safe_numeric(work["twampReportCurrentIpdvMax"])
    tx_pkts = _safe_numeric(work["twampReportCurrentTxPackets"])
    rx_pkts = _safe_numeric(work["twampReportCurrentRxValidPackets"])
    loss_pkts = _safe_numeric(work["twampReportCurrentLossPackets"])
    elapsed = _safe_numeric(work["twampReportCurrentElapsedTime"])
    avail_sec = _safe_numeric(work["twampReportCurrentAvailableSeconds"])
    delay_fwd_max = _safe_numeric(work["twampReportCurrentDelayFwdMax"])
    delay_bck_max = _safe_numeric(work["twampReportCurrentDelayBckMax"])

    total_intervals = len(work)
    active_mask = tx_pkts > 0
    active_count = int(active_mask.sum())
    idle_count = total_intervals - active_count

    # ── SLA thresholds ───────────────────────────────────────────────
    sla = prefs.get("sla_thresholds", {})
    lat_good = float(sla.get("latency_good", 10))
    lat_crit = float(sla.get("latency_critical", 50))
    jit_good = float(sla.get("jitter_good", 5))
    jit_crit = float(sla.get("jitter_critical", 20))
    loss_good = float(sla.get("packet_loss_good", 0))
    loss_crit = float(sla.get("packet_loss_critical", 1))

    # ── Loss percentage (safe division) ──────────────────────────────
    loss_pct = (loss_pkts / tx_pkts.replace(0, float("nan")) * 100).fillna(0.0)

    # ── Per-interval classification ──────────────────────────────────
    def classify(val, good_th, crit_th):
        if pd.isna(val):
            return "no_data"
        if val <= good_th:
            return "good"
        if val <= crit_th:
            return "degraded"
        return "critical"

    lat_status = delay_avg.apply(lambda v: classify(v, lat_good, lat_crit))
    jit_status = pdv_max.apply(lambda v: classify(v, jit_good, jit_crit))
    loss_status = loss_pct.apply(lambda v: classify(v, loss_good, loss_crit))

    def status_counts(series):
        counts = series.value_counts().to_dict()
        return {
            "good": int(counts.get("good", 0)),
            "degraded": int(counts.get("degraded", 0)),
            "critical": int(counts.get("critical", 0)),
            "no_data": int(counts.get("no_data", 0)),
        }

    # ── Overall SLA verdict ──────────────────────────────────────────
    worst = "good"
    for s in [lat_status, jit_status, loss_status]:
        if (s == "critical").any():
            worst = "critical"
            break
        if (s == "degraded").any():
            worst = "degraded"

    sla_compliance_pct = round(
        ((lat_status == "good") & (jit_status == "good") & (loss_status == "good")).sum()
        / max(1, total_intervals) * 100, 1
    )

    # ── Summary statistics ───────────────────────────────────────────
    summary = {
        "total_intervals": total_intervals,
        "active_intervals": active_count,
        "idle_intervals": idle_count,
        "unique_sessions": int(work["twampContSessionId"].nunique()),
        "unique_controllers": int(work["twampControllerId"].nunique()),
        "unique_peers": int(work["twampPeerAddr"].nunique()),
        "sla_compliance_pct": sla_compliance_pct,
        "overall_verdict": worst,
        "delay_avg_mean": round(float(delay_avg.mean()), 2) if delay_avg.notna().any() else 0.0,
        "delay_avg_max": round(float(delay_avg.max()), 2) if delay_avg.notna().any() else 0.0,
        "delay_max_peak": round(float(delay_max.max()), 2) if delay_max.notna().any() else 0.0,
        "jitter_max_peak": round(float(pdv_max.max()), 2) if pdv_max.notna().any() else 0.0,
        "ipdv_max_peak": round(float(ipdv_max.max()), 2) if ipdv_max.notna().any() else 0.0,
        "total_tx_packets": int(tx_pkts.sum()),
        "total_rx_packets": int(rx_pkts.sum()),
        "total_lost_packets": int(loss_pkts.sum()),
        "loss_pct_mean": round(float(loss_pct.mean()), 4),
        "total_elapsed_seconds": int(elapsed.sum()),
    }

    # ── Compliance breakdown ─────────────────────────────────────────
    compliance = {
        "latency": status_counts(lat_status),
        "jitter": status_counts(jit_status),
        "packet_loss": status_counts(loss_status),
    }

    # ── Per-interval detail table ────────────────────────────────────
    per_interval = []
    for i in range(total_intervals):
        per_interval.append({
            "interval": i + 1,
            "session_id": str(work.iloc[i]["twampContSessionId"]),
            "elapsed_sec": int(elapsed.iloc[i]) if pd.notna(elapsed.iloc[i]) else 0,
            "tx_packets": int(tx_pkts.iloc[i]) if pd.notna(tx_pkts.iloc[i]) else 0,
            "rx_packets": int(rx_pkts.iloc[i]) if pd.notna(rx_pkts.iloc[i]) else 0,
            "loss_packets": int(loss_pkts.iloc[i]) if pd.notna(loss_pkts.iloc[i]) else 0,
            "loss_pct": round(float(loss_pct.iloc[i]), 4),
            "delay_avg": round(float(delay_avg.iloc[i]), 2) if pd.notna(delay_avg.iloc[i]) else None,
            "delay_max": round(float(delay_max.iloc[i]), 2) if pd.notna(delay_max.iloc[i]) else None,
            "jitter_max": round(float(pdv_max.iloc[i]), 2) if pd.notna(pdv_max.iloc[i]) else None,
            "latency_status": str(lat_status.iloc[i]),
            "jitter_status": str(jit_status.iloc[i]),
            "loss_status": str(loss_status.iloc[i]),
        })

    # ── Charts data ──────────────────────────────────────────────────
    charts = []
    # Compliance distribution pie chart
    total_checks = total_intervals * 3  # 3 metrics per interval
    good_total = sum(c["good"] for c in compliance.values())
    degraded_total = sum(c["degraded"] for c in compliance.values())
    critical_total = sum(c["critical"] for c in compliance.values())
    no_data_total = sum(c["no_data"] for c in compliance.values())
    charts.append({
        "chart_type": "doughnut",
        "title": "SLA Compliance Distribution",
        "labels": ["Good", "Degraded", "Critical", "No Data"],
        "datasets": [{
            "data": [good_total, degraded_total, critical_total, no_data_total],
            "backgroundColor": ["#4caf50", "#ff9800", "#f44336", "#9e9e9e"],
        }],
    })

    # Delay trend across intervals
    if active_count > 0:
        charts.append({
            "chart_type": "line",
            "title": "Delay Trend Across Intervals",
            "labels": [f"Int {i + 1}" for i in range(total_intervals)],
            "datasets": [
                {
                    "label": "Avg Delay",
                    "data": [round(float(v), 2) if pd.notna(v) else None for v in delay_avg],
                    "borderColor": "#1976d2",
                },
                {
                    "label": "Max Delay",
                    "data": [round(float(v), 2) if pd.notna(v) else None for v in delay_max],
                    "borderColor": "#f44336",
                },
                {
                    "label": f"SLA Good ({lat_good})",
                    "data": [lat_good] * total_intervals,
                    "borderColor": "#4caf50",
                    "borderDash": [5, 5],
                },
                {
                    "label": f"SLA Critical ({lat_crit})",
                    "data": [lat_crit] * total_intervals,
                    "borderColor": "#f44336",
                    "borderDash": [5, 5],
                },
            ],
        })

    # Packet counts bar chart
    charts.append({
        "chart_type": "bar",
        "title": "Packet Counts per Interval",
        "labels": [f"Int {i + 1}" for i in range(total_intervals)],
        "datasets": [
            {
                "label": "TX Packets",
                "data": [int(v) if pd.notna(v) else 0 for v in tx_pkts],
                "backgroundColor": "#1976d2",
            },
            {
                "label": "RX Packets",
                "data": [int(v) if pd.notna(v) else 0 for v in rx_pkts],
                "backgroundColor": "#4caf50",
            },
            {
                "label": "Lost Packets",
                "data": [int(v) if pd.notna(v) else 0 for v in loss_pkts],
                "backgroundColor": "#f44336",
            },
        ],
    })

    # ── Alerts ───────────────────────────────────────────────────────
    alerts = []
    if idle_count == total_intervals:
        alerts.append({
            "severity": "warning",
            "message": f"All {total_intervals} intervals have zero TX packets — sessions appear idle or not yet started.",
        })
    elif idle_count > 0:
        alerts.append({
            "severity": "warning",
            "message": f"{idle_count} of {total_intervals} intervals have zero TX packets (idle sessions).",
        })
    if (lat_status == "critical").any():
        n = int((lat_status == "critical").sum())
        alerts.append({
            "severity": "critical",
            "message": f"{n} interval(s) exceed the critical latency threshold ({lat_crit} units).",
        })
    if (jit_status == "critical").any():
        n = int((jit_status == "critical").sum())
        alerts.append({
            "severity": "critical",
            "message": f"{n} interval(s) exceed the critical jitter threshold ({jit_crit} units).",
        })
    if (loss_status == "critical").any():
        n = int((loss_status == "critical").sum())
        alerts.append({
            "severity": "critical",
            "message": f"{n} interval(s) exceed the critical packet loss threshold ({loss_crit}%).",
        })
    if sla_compliance_pct < 100 and worst != "critical":
        alerts.append({
            "severity": "warning",
            "message": f"SLA compliance is {sla_compliance_pct}% — some intervals are degraded.",
        })
    if not alerts:
        alerts.append({
            "severity": "info",
            "message": "All SLA thresholds are within acceptable limits.",
        })

    return {
        "summary": summary,
        "compliance": compliance,
        "per_interval": per_interval,
        "charts": charts,
        "alerts": alerts,
    }
