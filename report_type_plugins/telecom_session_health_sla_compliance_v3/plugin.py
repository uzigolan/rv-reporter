import pandas as pd


_SENTINEL = 4294967295  # 0xFFFFFFFF — "not available" in PM CSV-ES


def get_spec():
    return {
        "metrics_profile": "telecom_session_health_sla_compliance_v3",
        "api_version": 1,
        "title": "Telecom Session Health SLA Compliance Report v3",
        "description": "Evaluates TWAMP session health against SLA thresholds for delay, jitter, and packet loss.",
    }


def _safe_numeric(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return vals.replace(_SENTINEL, float("nan"))


def build(df, prefs, ctx):
    work = df.copy()
    sla = prefs.get("sla_thresholds", {})

    # ── Thresholds ───────────────────────────────────────────────────
    lat_good = float(sla.get("delay_degraded", sla.get("latency_good", 10)))
    lat_crit = float(sla.get("delay_critical", sla.get("latency_critical", 50)))
    jit_good = float(sla.get("jitter_degraded", sla.get("jitter_good", 10)))
    jit_crit = float(sla.get("jitter_critical", 30))
    loss_good = float(sla.get("packet_loss_degraded", sla.get("packet_loss_good", 1)))
    loss_crit = float(sla.get("packet_loss_critical", 5))

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
    delay_fwd_max = _safe_numeric(work["twampReportCurrentDelayFwdMax"])
    delay_bck_max = _safe_numeric(work["twampReportCurrentDelayBckMax"])

    total_intervals = len(work)
    active_mask = tx_pkts > 0
    active_count = int(active_mask.sum())
    idle_count = total_intervals - active_count

    # ── Loss percentage (safe division) ──────────────────────────────
    loss_pct = (loss_pkts / tx_pkts.replace(0, float("nan")) * 100).fillna(0.0)
    jitter_combined = pdv_max.combine(ipdv_max, max).fillna(0.0)

    # ── Per-interval classification ──────────────────────────────────
    def classify(val, good_th, crit_th):
        if pd.isna(val):
            return "no_data"
        if val <= good_th:
            return "compliant"
        if val <= crit_th:
            return "degraded"
        return "critical"

    lat_status = delay_avg.apply(lambda v: classify(v, lat_good, lat_crit))
    jit_status = jitter_combined.apply(lambda v: classify(v, jit_good, jit_crit))
    loss_status = loss_pct.apply(lambda v: classify(v, loss_good, loss_crit))

    def status_counts(series):
        c = series.value_counts().to_dict()
        return {
            "compliant": int(c.get("compliant", 0)),
            "degraded": int(c.get("degraded", 0)),
            "critical": int(c.get("critical", 0)),
            "no_data": int(c.get("no_data", 0)),
        }

    # ── Overall verdict ──────────────────────────────────────────────
    worst = "compliant"
    for s in [lat_status, jit_status, loss_status]:
        if (s == "critical").any():
            worst = "critical"
            break
        if (s == "degraded").any():
            worst = "degraded"

    compliant_count = int(
        ((lat_status == "compliant") & (jit_status == "compliant") & (loss_status == "compliant")).sum()
    )
    sla_compliance_pct = round(compliant_count / max(1, total_intervals) * 100, 1)

    # ── Summary ──────────────────────────────────────────────────────
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
        "jitter_combined_peak": round(float(jitter_combined.max()), 2),
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

    # ── Per-interval detail rows ─────────────────────────────────────
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
            "jitter_combined": round(float(jitter_combined.iloc[i]), 2),
            "delay_fwd_max": round(float(delay_fwd_max.iloc[i]), 2) if pd.notna(delay_fwd_max.iloc[i]) else None,
            "delay_bck_max": round(float(delay_bck_max.iloc[i]), 2) if pd.notna(delay_bck_max.iloc[i]) else None,
            "latency_status": str(lat_status.iloc[i]),
            "jitter_status": str(jit_status.iloc[i]),
            "loss_status": str(loss_status.iloc[i]),
        })

    # ── Charts ───────────────────────────────────────────────────────
    charts = []

    # 1. Compliance doughnut
    comp_good = sum(c["compliant"] for c in compliance.values())
    comp_deg = sum(c["degraded"] for c in compliance.values())
    comp_crit = sum(c["critical"] for c in compliance.values())
    comp_nd = sum(c["no_data"] for c in compliance.values())
    charts.append({
        "chart_type": "doughnut",
        "title": "SLA Compliance Distribution",
        "labels": ["Compliant", "Degraded", "Critical", "No Data"],
        "datasets": [{
            "data": [comp_good, comp_deg, comp_crit, comp_nd],
            "backgroundColor": ["#4caf50", "#ff9800", "#f44336", "#9e9e9e"],
        }],
    })

    # 2. Per-metric compliance stacked bar
    charts.append({
        "chart_type": "bar",
        "title": "Compliance by Metric",
        "labels": ["Latency", "Jitter", "Packet Loss"],
        "datasets": [
            {"label": "Compliant", "data": [compliance["latency"]["compliant"], compliance["jitter"]["compliant"], compliance["packet_loss"]["compliant"]], "backgroundColor": "#4caf50"},
            {"label": "Degraded", "data": [compliance["latency"]["degraded"], compliance["jitter"]["degraded"], compliance["packet_loss"]["degraded"]], "backgroundColor": "#ff9800"},
            {"label": "Critical", "data": [compliance["latency"]["critical"], compliance["jitter"]["critical"], compliance["packet_loss"]["critical"]], "backgroundColor": "#f44336"},
        ],
    })

    # 3. Delay trend line chart
    charts.append({
        "chart_type": "line",
        "title": "Delay Trend Across Intervals",
        "labels": [f"Int {i + 1}" for i in range(total_intervals)],
        "datasets": [
            {"label": "Avg Delay", "data": [round(float(v), 2) if pd.notna(v) else None for v in delay_avg], "borderColor": "#1976d2"},
            {"label": "Max Delay", "data": [round(float(v), 2) if pd.notna(v) else None for v in delay_max], "borderColor": "#e65100"},
            {"label": f"SLA Threshold ({lat_crit})", "data": [lat_crit] * total_intervals, "borderColor": "#f44336", "borderDash": [5, 5]},
        ],
    })

    # 4. Jitter + loss trend
    charts.append({
        "chart_type": "line",
        "title": "Jitter & Packet Loss Trend",
        "labels": [f"Int {i + 1}" for i in range(total_intervals)],
        "datasets": [
            {"label": "Jitter (combined max)", "data": [round(float(v), 2) for v in jitter_combined], "borderColor": "#7b1fa2"},
            {"label": "Loss %", "data": [round(float(v), 4) for v in loss_pct], "borderColor": "#f44336", "yAxisID": "y1"},
        ],
    })

    # 5. Packet volume bar chart
    charts.append({
        "chart_type": "bar",
        "title": "Packet Volume per Interval",
        "labels": [f"Int {i + 1}" for i in range(total_intervals)],
        "datasets": [
            {"label": "TX", "data": [int(v) if pd.notna(v) else 0 for v in tx_pkts], "backgroundColor": "#1976d2"},
            {"label": "RX", "data": [int(v) if pd.notna(v) else 0 for v in rx_pkts], "backgroundColor": "#4caf50"},
            {"label": "Lost", "data": [int(v) if pd.notna(v) else 0 for v in loss_pkts], "backgroundColor": "#f44336"},
        ],
    })

    # ── Alerts ───────────────────────────────────────────────────────
    alerts = []
    if idle_count == total_intervals:
        alerts.append({"severity": "warning", "message": f"All {total_intervals} intervals have zero TX packets — sessions appear idle or not yet started."})
    elif idle_count > 0:
        alerts.append({"severity": "warning", "message": f"{idle_count} of {total_intervals} intervals have zero TX packets (idle sessions)."})
    if (lat_status == "critical").any():
        n = int((lat_status == "critical").sum())
        alerts.append({"severity": "critical", "message": f"{n} interval(s) exceed the critical latency threshold ({lat_crit})."})
    if (jit_status == "critical").any():
        n = int((jit_status == "critical").sum())
        alerts.append({"severity": "critical", "message": f"{n} interval(s) exceed the critical jitter threshold ({jit_crit})."})
    if (loss_status == "critical").any():
        n = int((loss_status == "critical").sum())
        alerts.append({"severity": "critical", "message": f"{n} interval(s) exceed the critical packet-loss threshold ({loss_crit}%)."})
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
