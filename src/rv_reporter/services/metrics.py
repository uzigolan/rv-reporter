from __future__ import annotations

from typing import Any

import pandas as pd


def compute_metrics(profile: str, df: pd.DataFrame, prefs: dict[str, Any]) -> dict[str, Any]:
    if profile == "ops_kpi":
        return _compute_ops_kpi(df, prefs)
    if profile == "finance_variance":
        return _compute_finance_variance(df, prefs)
    if profile == "network_queue_congestion":
        return _compute_network_queue_congestion(df, prefs)
    if profile == "twamp_session_health":
        return _compute_twamp_session_health(df, prefs)
    if profile == "pm_export_health":
        return _compute_pm_export_health(df, prefs)
    raise ValueError(f"Unsupported metrics profile '{profile}'.")


def _compute_ops_kpi(df: pd.DataFrame, prefs: dict[str, Any]) -> dict[str, Any]:
    working = df.copy()
    if "timestamp" in working.columns:
        working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
        working["date"] = working["timestamp"].dt.date

    threshold = float(prefs.get("alert_error_rate", 0.05))
    grouped = working.groupby("service", dropna=False)
    per_service = []
    alerts = []

    for service, g in grouped:
        requests = float(g["requests"].sum())
        errors = float(g["errors"].sum())
        error_rate = (errors / requests) if requests > 0 else 0.0
        p95_latency = float(g["latency_ms"].quantile(0.95))
        per_service.append(
            {
                "service": str(service),
                "requests": int(requests),
                "errors": int(errors),
                "error_rate": round(error_rate, 4),
                "latency_p95_ms": round(p95_latency, 2),
            }
        )
        if error_rate > threshold:
            alerts.append(
                {
                    "severity": "high",
                    "message": f"{service}: error rate {error_rate:.2%} exceeds {threshold:.2%}",
                }
            )

    daily = []
    if "date" in working.columns:
        daily_grouped = (
            working.groupby("date", dropna=True)[["requests", "errors"]]
            .sum()
            .reset_index()
            .sort_values("date")
        )
        for _, row in daily_grouped.iterrows():
            req = float(row["requests"])
            err = float(row["errors"])
            daily.append(
                {
                    "date": str(row["date"]),
                    "requests": int(req),
                    "errors": int(err),
                    "error_rate": round((err / req) if req > 0 else 0.0, 4),
                }
            )

    return {
        "totals": {
            "requests": int(working["requests"].sum()),
            "errors": int(working["errors"].sum()),
            "avg_latency_ms": round(float(working["latency_ms"].mean()), 2),
        },
        "per_service": per_service,
        "daily_trend": daily,
        "alerts": alerts,
    }


def _compute_finance_variance(df: pd.DataFrame, prefs: dict[str, Any]) -> dict[str, Any]:
    working = df.copy()
    threshold = float(prefs.get("variance_alert_pct", 0.1))
    working["variance"] = working["actual"] - working["budget"]
    working["variance_pct"] = working.apply(
        lambda row: (row["variance"] / row["budget"]) if row["budget"] else 0.0,
        axis=1,
    )
    rows = []
    alerts = []
    for _, row in working.iterrows():
        pct = float(row["variance_pct"])
        item = {
            "month": str(row["month"]),
            "category": str(row["category"]),
            "actual": float(row["actual"]),
            "budget": float(row["budget"]),
            "variance": float(row["variance"]),
            "variance_pct": round(pct, 4),
        }
        rows.append(item)
        if abs(pct) > threshold:
            alerts.append(
                {
                    "severity": "medium",
                    "message": (
                        f"{row['month']} {row['category']}: variance {pct:.2%} exceeds {threshold:.2%}"
                    ),
                }
            )

    return {
        "totals": {
            "actual": float(working["actual"].sum()),
            "budget": float(working["budget"].sum()),
            "variance": float(working["variance"].sum()),
        },
        "rows": rows,
        "alerts": alerts,
    }


def _compute_network_queue_congestion(df: pd.DataFrame, prefs: dict[str, Any]) -> dict[str, Any]:
    working = df.copy()
    working["Time"] = pd.to_datetime(working["Time"], errors="coerce")
    keys = ["NE Name", "Resource Name", "Queue Block", "Queue Number"]
    working = working.sort_values(keys + ["Time"])

    counter_columns = [
        "Dequeued (Bytes)",
        "Dequeued (Frames)",
        "Dropped (Bytes)",
        "Dropped (Frames)",
    ]
    for col in counter_columns:
        working[f"d_{col}"] = working.groupby(keys)[col].diff()

    deltas = working.dropna(subset=[f"d_{col}" for col in counter_columns]).copy()
    deltas = deltas[
        (deltas["d_Dequeued (Bytes)"] >= 0)
        & (deltas["d_Dropped (Bytes)"] >= 0)
        & (deltas["d_Dequeued (Frames)"] >= 0)
        & (deltas["d_Dropped (Frames)"] >= 0)
    ].copy()

    deltas["total_delta_bytes"] = deltas["d_Dequeued (Bytes)"] + deltas["d_Dropped (Bytes)"]
    deltas["total_delta_frames"] = deltas["d_Dequeued (Frames)"] + deltas["d_Dropped (Frames)"]
    deltas = deltas[
        (deltas["total_delta_bytes"] > 0) | (deltas["total_delta_frames"] > 0)
    ].copy()

    if deltas.empty:
        return {
            "summary": {
                "interval_samples": 0,
                "active_queues": 0,
                "overall_drop_ratio_bytes": 0.0,
                "overall_drop_ratio_frames": 0.0,
            },
            "top_queues": [],
            "time_trend": [],
            "alerts": [],
        }

    deltas["drop_ratio_bytes"] = deltas["d_Dropped (Bytes)"] / deltas["total_delta_bytes"].replace(0, pd.NA)
    deltas["drop_ratio_frames"] = deltas["d_Dropped (Frames)"] / deltas["total_delta_frames"].replace(0, pd.NA)
    deltas["drop_ratio_bytes"] = deltas["drop_ratio_bytes"].fillna(0.0)
    deltas["drop_ratio_frames"] = deltas["drop_ratio_frames"].fillna(0.0)

    grouped = (
        deltas.groupby(keys)
        .agg(
            samples=("drop_ratio_bytes", "count"),
            mean_drop_ratio_bytes=("drop_ratio_bytes", "mean"),
            p95_drop_ratio_bytes=("drop_ratio_bytes", lambda s: s.quantile(0.95)),
            max_drop_ratio_bytes=("drop_ratio_bytes", "max"),
            mean_drop_ratio_frames=("drop_ratio_frames", "mean"),
            dropped_bytes=("d_Dropped (Bytes)", "sum"),
            dequeued_bytes=("d_Dequeued (Bytes)", "sum"),
            dropped_frames=("d_Dropped (Frames)", "sum"),
            dequeued_frames=("d_Dequeued (Frames)", "sum"),
        )
        .reset_index()
        .sort_values(["mean_drop_ratio_bytes", "dropped_bytes"], ascending=[False, False])
    )

    threshold = float(prefs.get("alert_drop_ratio", 0.2))
    min_samples = int(prefs.get("alert_min_samples", 3))
    top_n = int(prefs.get("top_n_queues", 10))
    top_n_intervals = int(prefs.get("top_n_intervals", 8))

    top_queues = []
    alerts = []
    for _, row in grouped.head(top_n).iterrows():
        item = {
            "ne_name": str(row["NE Name"]),
            "resource_name": str(row["Resource Name"]),
            "queue_block": str(row["Queue Block"]),
            "queue_number": int(row["Queue Number"]),
            "samples": int(row["samples"]),
            "mean_drop_ratio_bytes": round(float(row["mean_drop_ratio_bytes"]), 4),
            "p95_drop_ratio_bytes": round(float(row["p95_drop_ratio_bytes"]), 4),
            "max_drop_ratio_bytes": round(float(row["max_drop_ratio_bytes"]), 4),
            "mean_drop_ratio_frames": round(float(row["mean_drop_ratio_frames"]), 4),
            "dropped_bytes": float(row["dropped_bytes"]),
            "dequeued_bytes": float(row["dequeued_bytes"]),
            "dropped_frames": float(row["dropped_frames"]),
            "dequeued_frames": float(row["dequeued_frames"]),
        }
        top_queues.append(item)
        if item["samples"] >= min_samples and item["mean_drop_ratio_bytes"] >= threshold:
            alerts.append(
                {
                    "severity": "high",
                    "message": (
                        f"{item['ne_name']} {item['resource_name']} "
                        f"Q{item['queue_block']}/{item['queue_number']}: "
                        f"mean byte drop ratio {item['mean_drop_ratio_bytes']:.2%} "
                        f"exceeds {threshold:.2%}"
                    ),
                }
            )

    trend = (
        deltas.groupby("Time")[["d_Dropped (Bytes)", "d_Dequeued (Bytes)", "d_Dropped (Frames)", "d_Dequeued (Frames)"]]
        .sum()
        .reset_index()
        .sort_values("Time")
    )

    per_time_queue = (
        deltas.groupby(["Time"] + keys)[["d_Dropped (Bytes)", "d_Dequeued (Bytes)"]]
        .sum()
        .reset_index()
    )
    dominant_by_time: dict[str, dict[str, Any]] = {}
    for t, frame in per_time_queue.groupby("Time"):
        total_drop = float(frame["d_Dropped (Bytes)"].sum())
        if frame.empty:
            continue
        top = frame.sort_values("d_Dropped (Bytes)", ascending=False).iloc[0]
        dominant_by_time[str(t)] = {
            "queue_label": f"{top['NE Name']} {top['Resource Name']} {top['Queue Block']}/{int(top['Queue Number'])}",
            "share_pct": round((float(top["d_Dropped (Bytes)"]) / total_drop) * 100, 2) if total_drop > 0 else 0.0,
        }

    time_trend = []
    for _, row in trend.iterrows():
        time_key = str(row["Time"])
        dom = dominant_by_time.get(time_key, {"queue_label": "n/a", "share_pct": 0.0})
        total_bytes = float(row["d_Dropped (Bytes)"] + row["d_Dequeued (Bytes)"])
        total_frames = float(row["d_Dropped (Frames)"] + row["d_Dequeued (Frames)"])
        time_trend.append(
            {
                "time": time_key,
                "drop_ratio_bytes": round(float(row["d_Dropped (Bytes)"] / total_bytes), 4) if total_bytes > 0 else 0.0,
                "drop_ratio_frames": round(float(row["d_Dropped (Frames)"] / total_frames), 4) if total_frames > 0 else 0.0,
                "dropped_bytes": float(row["d_Dropped (Bytes)"]),
                "dequeued_bytes": float(row["d_Dequeued (Bytes)"]),
                "dominant_queue": dom["queue_label"],
                "dominant_queue_share_pct": dom["share_pct"],
            }
        )

    interval_hotspots = sorted(time_trend, key=lambda x: x["drop_ratio_bytes"], reverse=True)[:top_n_intervals]
    for item in interval_hotspots:
        cause, confidence = _infer_network_hotspot_cause(item, threshold)
        item["likely_cause"] = cause
        item["confidence"] = confidence

    total_dropped_bytes = float(deltas["d_Dropped (Bytes)"].sum())
    total_dequeued_bytes = float(deltas["d_Dequeued (Bytes)"].sum())
    total_dropped_frames = float(deltas["d_Dropped (Frames)"].sum())
    total_dequeued_frames = float(deltas["d_Dequeued (Frames)"].sum())
    total_bytes = total_dropped_bytes + total_dequeued_bytes
    total_frames = total_dropped_frames + total_dequeued_frames
    interval_count = max(1, len(time_trend))
    intervals_over_threshold = sum(1 for row in time_trend if row["drop_ratio_bytes"] >= threshold)
    affected_intervals_pct = round((intervals_over_threshold / interval_count) * 100, 2)

    queue_drop_share = []
    for q in top_queues:
        share = (q["dropped_bytes"] / total_dropped_bytes * 100) if total_dropped_bytes > 0 else 0.0
        queue_drop_share.append(
            {
                "queue": f"{q['ne_name']} {q['resource_name']} {q['queue_block']}/{q['queue_number']}",
                "dropped_bytes_share_pct": round(share, 2),
            }
        )

    avg_drop = (total_dropped_bytes / total_bytes) if total_bytes > 0 else 0.0
    p95_drop = sorted([r["drop_ratio_bytes"] for r in time_trend])[int(max(0, round(0.95 * (len(time_trend) - 1))))] if time_trend else 0.0
    dominant_share = queue_drop_share[0]["dropped_bytes_share_pct"] if queue_drop_share else 0.0
    risk_score = _network_risk_score(
        avg_drop_ratio=avg_drop,
        p95_drop_ratio=p95_drop,
        affected_intervals_pct=affected_intervals_pct,
        dominant_queue_share_pct=dominant_share,
    )
    risk_band = _risk_band(risk_score)
    insights = _network_l2_qos_insights(
        dominant_queue_share_pct=dominant_share,
        affected_intervals_pct=affected_intervals_pct,
        avg_drop_ratio=avg_drop,
        threshold=threshold,
    )

    return {
        "summary": {
            "interval_samples": int(len(deltas)),
            "active_queues": int(grouped.shape[0]),
            "overall_drop_ratio_bytes": round((total_dropped_bytes / total_bytes) if total_bytes > 0 else 0.0, 4),
            "overall_drop_ratio_frames": round((total_dropped_frames / total_frames) if total_frames > 0 else 0.0, 4),
            "total_dropped_bytes": total_dropped_bytes,
            "total_dequeued_bytes": total_dequeued_bytes,
            "total_dropped_frames": total_dropped_frames,
            "total_dequeued_frames": total_dequeued_frames,
            "affected_intervals_pct": affected_intervals_pct,
            "risk_score": risk_score,
            "risk_band": risk_band,
        },
        "top_queues": top_queues,
        "queue_drop_share": queue_drop_share,
        "interval_hotspots": interval_hotspots,
        "l2_qos_insights": insights,
        "time_trend": time_trend,
        "alerts": alerts,
    }


def _compute_twamp_session_health(df: pd.DataFrame, prefs: dict[str, Any]) -> dict[str, Any]:
    working = df.copy()
    working["DateTimeUTC"] = pd.to_datetime(working["DateTimeUTC"], errors="coerce")
    working = working.sort_values("DateTimeUTC")

    numeric_cols = [
        "DiscardRatePct_Emulated",
        "YellowTrafficPct_Emulated",
        "FwdTotalPackets_Emulated",
        "ColorDiscardTotalPackets_Emulated",
        "DiscardGreenPackets_Emulated",
        "DiscardYellowPackets_Emulated",
        "DiscardRedPackets_Emulated",
        "FwdTotalBytes_Emulated",
        "ColorDiscardTotalBytes_Emulated",
        "DiscardGreenBytes_Emulated",
        "DiscardYellowBytes_Emulated",
        "DiscardRedBytes_Emulated",
        "twampReportCurrentDelayAverage",
        "twampReportCurrentIpdvMax",
    ]
    for col in numeric_cols:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")
        else:
            working[col] = 0.0

    working = working.dropna(subset=["DateTimeUTC"]).copy()
    if working.empty:
        return {
            "summary": {
                "samples": 0,
                "avg_discard_rate_pct": 0.0,
                "p95_discard_rate_pct": 0.0,
                "avg_yellow_traffic_pct": 0.0,
                "total_forward_packets": 0,
                "total_discard_packets": 0,
            },
            "top_discard_intervals": [],
            "time_trend": [],
            "alerts": [],
        }

    alert_threshold = float(prefs.get("alert_discard_rate_pct", 0.15))
    top_n = int(prefs.get("top_n_intervals", 5))

    trend = []
    alerts = []
    for _, row in working.iterrows():
        point = {
            "time_utc": str(row["DateTimeUTC"]),
            "discard_rate_pct": round(float(row["DiscardRatePct_Emulated"]), 4),
            "yellow_traffic_pct": round(float(row["YellowTrafficPct_Emulated"]), 4),
            "forward_packets": int(row["FwdTotalPackets_Emulated"] or 0),
            "discard_packets": int(row["ColorDiscardTotalPackets_Emulated"] or 0),
            "discard_green_packets": int(row["DiscardGreenPackets_Emulated"] or 0),
            "discard_yellow_packets": int(row["DiscardYellowPackets_Emulated"] or 0),
            "discard_red_packets": int(row["DiscardRedPackets_Emulated"] or 0),
            "discard_green_bytes": int(row["DiscardGreenBytes_Emulated"] or 0),
            "discard_yellow_bytes": int(row["DiscardYellowBytes_Emulated"] or 0),
            "discard_red_bytes": int(row["DiscardRedBytes_Emulated"] or 0),
            "delay_average": round(float(row["twampReportCurrentDelayAverage"]), 4),
            "ipdv_max": round(float(row["twampReportCurrentIpdvMax"]), 4),
        }
        trend.append(point)
        if point["discard_rate_pct"] >= alert_threshold:
            alerts.append(
                {
                    "severity": "high",
                    "message": (
                        f"{point['time_utc']}: discard rate {point['discard_rate_pct']:.4f}% "
                        f"exceeds {alert_threshold:.4f}%"
                    ),
                }
            )

    top_discard = sorted(trend, key=lambda x: x["discard_rate_pct"], reverse=True)[:top_n]
    for row in top_discard:
        row["likely_cause"], row["confidence"] = _infer_twamp_cause(
            row=row,
            alert_threshold=alert_threshold,
            avg_yellow_traffic_pct=float(working["YellowTrafficPct_Emulated"].mean()),
        )

    color_totals_packets = {
        "green": int(working["DiscardGreenPackets_Emulated"].sum()),
        "yellow": int(working["DiscardYellowPackets_Emulated"].sum()),
        "red": int(working["DiscardRedPackets_Emulated"].sum()),
    }
    color_totals_bytes = {
        "green": int(working["DiscardGreenBytes_Emulated"].sum()),
        "yellow": int(working["DiscardYellowBytes_Emulated"].sum()),
        "red": int(working["DiscardRedBytes_Emulated"].sum()),
    }
    total_color_packets = max(1, sum(color_totals_packets.values()))
    color_packet_share = {
        "green_pct": round(100 * color_totals_packets["green"] / total_color_packets, 4),
        "yellow_pct": round(100 * color_totals_packets["yellow"] / total_color_packets, 4),
        "red_pct": round(100 * color_totals_packets["red"] / total_color_packets, 4),
    }

    avg_discard_rate = float(working["DiscardRatePct_Emulated"].mean())
    p95_discard_rate = float(working["DiscardRatePct_Emulated"].quantile(0.95))
    avg_yellow_pct = float(working["YellowTrafficPct_Emulated"].mean())
    p95_ipdv = float(working["twampReportCurrentIpdvMax"].quantile(0.95))
    risk_score = _twamp_risk_score(
        avg_discard_rate_pct=avg_discard_rate,
        p95_discard_rate_pct=p95_discard_rate,
        green_discard_share_pct=color_packet_share["green_pct"],
        avg_yellow_traffic_pct=avg_yellow_pct,
        p95_ipdv_max=p95_ipdv,
    )
    risk_band = _risk_band(risk_score)
    hypotheses = _twamp_hypotheses(
        color_packet_share=color_packet_share,
        avg_discard_rate_pct=avg_discard_rate,
        p95_discard_rate_pct=p95_discard_rate,
        avg_yellow_traffic_pct=avg_yellow_pct,
    )

    return {
        "summary": {
            "samples": int(len(working)),
            "avg_discard_rate_pct": round(avg_discard_rate, 4),
            "p95_discard_rate_pct": round(p95_discard_rate, 4),
            "avg_yellow_traffic_pct": round(avg_yellow_pct, 4),
            "total_forward_packets": int(working["FwdTotalPackets_Emulated"].sum()),
            "total_discard_packets": int(working["ColorDiscardTotalPackets_Emulated"].sum()),
            "avg_delay": round(float(working["twampReportCurrentDelayAverage"].mean()), 4),
            "p95_ipdv_max": round(p95_ipdv, 4),
            "risk_score": risk_score,
            "risk_band": risk_band,
        },
        "color_discard_packets": color_totals_packets,
        "color_discard_bytes": color_totals_bytes,
        "color_packet_share_pct": color_packet_share,
        "l2_qos_insights": hypotheses,
        "top_discard_intervals": top_discard,
        "time_trend": trend,
        "alerts": alerts,
    }


def _infer_twamp_cause(
    row: dict[str, Any],
    alert_threshold: float,
    avg_yellow_traffic_pct: float,
) -> tuple[str, str]:
    discard_rate = float(row.get("discard_rate_pct", 0.0))
    yellow = float(row.get("yellow_traffic_pct", 0.0))
    green_discards = int(row.get("discard_green_packets", 0))
    red_discards = int(row.get("discard_red_packets", 0))
    if green_discards > 0 and discard_rate >= alert_threshold:
        return ("Possible severe contention/misclassification impacting high-priority traffic.", "high")
    if yellow >= avg_yellow_traffic_pct + 1.0 and discard_rate >= alert_threshold:
        return ("Likely QoS congestion on lower-priority traffic under load.", "medium")
    if red_discards > (green_discards + 1):
        return ("Color-aware policing/queue pressure affecting lower-priority classes.", "medium")
    if discard_rate >= alert_threshold:
        return ("Intermittent congestion likely at egress queue or policy boundary.", "medium")
    return ("No strong L2/QoS anomaly signal in this interval.", "low")


def _twamp_risk_score(
    avg_discard_rate_pct: float,
    p95_discard_rate_pct: float,
    green_discard_share_pct: float,
    avg_yellow_traffic_pct: float,
    p95_ipdv_max: float,
) -> int:
    s1 = min(1.0, avg_discard_rate_pct / 0.2) * 35
    s2 = min(1.0, p95_discard_rate_pct / 0.35) * 25
    s3 = min(1.0, green_discard_share_pct / 20.0) * 20
    s4 = min(1.0, avg_yellow_traffic_pct / 15.0) * 10
    s5 = min(1.0, p95_ipdv_max / 10.0) * 10
    return int(round(s1 + s2 + s3 + s4 + s5))


def _risk_band(score: int) -> str:
    if score >= 75:
        return "critical"
    if score >= 50:
        return "degraded"
    if score >= 25:
        return "watch"
    return "healthy"


def _twamp_hypotheses(
    color_packet_share: dict[str, float],
    avg_discard_rate_pct: float,
    p95_discard_rate_pct: float,
    avg_yellow_traffic_pct: float,
) -> list[dict[str, str]]:
    insights: list[dict[str, str]] = []
    green_share = color_packet_share.get("green_pct", 0.0)
    yellow_share = color_packet_share.get("yellow_pct", 0.0)
    red_share = color_packet_share.get("red_pct", 0.0)
    if green_share > 5:
        insights.append(
            {
                "hypothesis": "Non-trivial green discard share suggests high-priority impact under contention.",
                "confidence": "high",
            }
        )
    if yellow_share + red_share > 80 and avg_yellow_traffic_pct > 1.0:
        insights.append(
            {
                "hypothesis": "Lower-priority traffic likely dominates discard behavior due to QoS policing/queue pressure.",
                "confidence": "medium",
            }
        )
    if p95_discard_rate_pct >= max(0.2, avg_discard_rate_pct * 1.5):
        insights.append(
            {
                "hypothesis": "Discard spikes appear burst-driven rather than steady-state, indicating interval congestion events.",
                "confidence": "medium",
            }
        )
    if not insights:
        insights.append(
            {
                "hypothesis": "Discard profile appears stable; continue monitoring for emerging congestion patterns.",
                "confidence": "low",
            }
        )
    return insights


def _infer_network_hotspot_cause(row: dict[str, Any], threshold: float) -> tuple[str, str]:
    ratio = float(row.get("drop_ratio_bytes", 0.0))
    dom_share = float(row.get("dominant_queue_share_pct", 0.0))
    if ratio >= threshold and dom_share >= 80:
        return ("Single-queue congestion dominance likely (queue scheduling/policer bottleneck).", "high")
    if ratio >= threshold and dom_share >= 55:
        return ("Primary queue likely driving burst loss with secondary contention.", "medium")
    if ratio >= threshold:
        return ("Broad multi-queue contention likely at egress.", "medium")
    return ("No severe congestion signal for this interval.", "low")


def _network_risk_score(
    avg_drop_ratio: float,
    p95_drop_ratio: float,
    affected_intervals_pct: float,
    dominant_queue_share_pct: float,
) -> int:
    s1 = min(1.0, avg_drop_ratio / 0.2) * 35
    s2 = min(1.0, p95_drop_ratio / 0.4) * 25
    s3 = min(1.0, affected_intervals_pct / 60.0) * 25
    s4 = min(1.0, dominant_queue_share_pct / 85.0) * 15
    return int(round(s1 + s2 + s3 + s4))


def _network_l2_qos_insights(
    dominant_queue_share_pct: float,
    affected_intervals_pct: float,
    avg_drop_ratio: float,
    threshold: float,
) -> list[dict[str, str]]:
    insights: list[dict[str, str]] = []
    if dominant_queue_share_pct >= 75:
        insights.append(
            {
                "hypothesis": "A single queue dominates dropped bytes, suggesting localized queue/policy bottleneck.",
                "confidence": "high",
            }
        )
    if affected_intervals_pct >= 40:
        insights.append(
            {
                "hypothesis": "Loss spans many intervals, indicating sustained congestion rather than isolated bursts.",
                "confidence": "medium",
            }
        )
    if avg_drop_ratio >= threshold:
        insights.append(
            {
                "hypothesis": "Average drop ratio exceeds alert threshold, likely causing user-visible degradation.",
                "confidence": "high",
            }
        )
    if not insights:
        insights.append(
            {
                "hypothesis": "Drop profile is relatively stable; continue monitoring for trend shifts.",
                "confidence": "low",
            }
        )
    return insights


def _compute_pm_export_health(df: pd.DataFrame, prefs: dict[str, Any]) -> dict[str, Any]:
    records = _parse_pm_export_records(df)
    if not records:
        return {
            "summary": {"records": 0, "risk_score": 0, "risk_band": "healthy"},
            "table_counts": {},
            "top_interface_discards": [],
            "alerts": [],
            "l2_system_hypotheses": [],
        }

    discard_alert_delta = float(prefs.get("discard_alert_delta", 1000))
    crc_alert_delta = float(prefs.get("crc_alert_delta", 100))
    cpu_alert_pct = float(prefs.get("cpu_alert_pct", 80))
    memory_alert_pct = float(prefs.get("memory_alert_pct", 85))
    disk_alert_pct = float(prefs.get("disk_alert_pct", 85))

    table_counts: dict[str, int] = {}
    for row in records:
        table_counts[row["table_name"]] = table_counts.get(row["table_name"], 0) + 1

    alerts: list[dict[str, str]] = []

    if_rows = [r for r in records if r["table_name"] == "ifTable"]
    if_df = pd.DataFrame(if_rows)
    top_interface_discards = []
    total_discard_delta = 0.0
    total_error_delta = 0.0
    total_traffic_delta = 0.0
    total_discard_abs = 0.0
    total_error_abs = 0.0
    total_traffic_abs = 0.0
    interface_signal_basis = "delta"
    interface_error_signal = "error_delta"
    interface_error_signal_label = "Error Delta"
    if not if_df.empty and "ifIndex" in if_df.columns:
        for col in ["ifInDiscards", "ifOutDiscards", "ifInErrors", "ifOutErrors", "ifInOctets", "ifOutOctets"]:
            if col in if_df.columns:
                if_df[col] = pd.to_numeric(if_df[col], errors="coerce")
        if_df["time_utc"] = pd.to_datetime(if_df["time_utc"], errors="coerce")
        if_df = if_df.sort_values(["ifIndex", "time_utc"])
        for col in ["ifInDiscards", "ifOutDiscards", "ifInErrors", "ifOutErrors", "ifInOctets", "ifOutOctets"]:
            if col in if_df.columns:
                if_df[f"d_{col}"] = if_df.groupby("ifIndex")[col].diff()
        if_df = if_df.fillna(0)
        if_df = if_df[(if_df.get("d_ifInDiscards", 0) >= 0) & (if_df.get("d_ifOutDiscards", 0) >= 0)]
        if_df["discard_delta"] = if_df.get("d_ifInDiscards", 0) + if_df.get("d_ifOutDiscards", 0)
        if_df["error_delta"] = if_df.get("d_ifInErrors", 0) + if_df.get("d_ifOutErrors", 0)
        if_df["traffic_delta"] = if_df.get("d_ifInOctets", 0) + if_df.get("d_ifOutOctets", 0)
        if_df["discard_abs"] = if_df.get("ifInDiscards", 0) + if_df.get("ifOutDiscards", 0)
        if_df["error_abs"] = if_df.get("ifInErrors", 0) + if_df.get("ifOutErrors", 0)
        if_df["traffic_abs"] = if_df.get("ifInOctets", 0) + if_df.get("ifOutOctets", 0)
        total_discard_delta = float(if_df["discard_delta"].sum())
        total_error_delta = float(if_df["error_delta"].sum())
        total_traffic_delta = float(if_df["traffic_delta"].sum())
        total_discard_abs = float(if_df["discard_abs"].sum())
        total_error_abs = float(if_df["error_abs"].sum())
        total_traffic_abs = float(if_df["traffic_abs"].sum())
        grouped_if = (
            if_df.groupby("ifIndex")[["discard_delta", "error_delta", "traffic_delta", "discard_abs", "error_abs", "traffic_abs"]]
            .sum()
            .reset_index()
            .sort_values("discard_delta", ascending=False)
        )

        # With single-snapshot exports, deltas can be all zero; use absolute counters as fallback.
        use_abs_for_discard = total_discard_delta <= 0 and total_discard_abs > 0
        if use_abs_for_discard:
            interface_signal_basis = "absolute_counter"
            grouped_if = grouped_if.sort_values("discard_abs", ascending=False)
        else:
            grouped_if = grouped_if.sort_values("discard_delta", ascending=False)

        use_error_delta = total_error_delta > 0
        use_error_abs = (not use_error_delta) and total_error_abs > 0
        if use_error_delta:
            interface_error_signal = "error_delta"
            interface_error_signal_label = "Error Delta"
        elif use_error_abs:
            interface_error_signal = "error_abs"
            interface_error_signal_label = "Error Counter"
        else:
            interface_error_signal = "traffic_abs" if use_abs_for_discard else "traffic_delta"
            interface_error_signal_label = "Traffic Bytes Counter" if use_abs_for_discard else "Traffic Bytes Delta"

        for _, row in grouped_if.head(8).iterrows():
            discard_value = row["discard_abs"] if use_abs_for_discard else row["discard_delta"]
            top_interface_discards.append(
                {
                    "ifIndex": int(float(row["ifIndex"])),
                    "discard_delta": int(discard_value),
                    "error_delta": int(row["error_delta"]),
                    "error_chart_value": int(row[interface_error_signal]),
                }
            )
        if total_discard_delta >= discard_alert_delta:
            alerts.append(
                {
                    "severity": "high",
                    "message": (
                        f"Interface discard delta {total_discard_delta:.0f} exceeds {discard_alert_delta:.0f}."
                    ),
                }
            )

    ether_rows = [r for r in records if r["table_name"] == "etherStatsTable"]
    total_crc_delta = 0.0
    if ether_rows:
        eth_df = pd.DataFrame(ether_rows)
        if "etherStatsCRCAlignErrors" in eth_df.columns and "etherStatsIndex" in eth_df.columns:
            eth_df["etherStatsCRCAlignErrors"] = pd.to_numeric(eth_df["etherStatsCRCAlignErrors"], errors="coerce")
            eth_df["etherStatsIndex"] = pd.to_numeric(eth_df["etherStatsIndex"], errors="coerce")
            eth_df["time_utc"] = pd.to_datetime(eth_df["time_utc"], errors="coerce")
            eth_df = eth_df.sort_values(["etherStatsIndex", "time_utc"])
            eth_df["crc_delta"] = eth_df.groupby("etherStatsIndex")["etherStatsCRCAlignErrors"].diff().fillna(0)
            eth_df = eth_df[eth_df["crc_delta"] >= 0]
            total_crc_delta = float(eth_df["crc_delta"].sum())
            if total_crc_delta >= crc_alert_delta:
                alerts.append(
                    {
                        "severity": "high",
                        "message": f"CRC/align error delta {total_crc_delta:.0f} exceeds {crc_alert_delta:.0f}.",
                    }
                )

    cpu_rows = [r for r in records if r["table_name"] == "agnCpuUtilizationTable"]
    cpu_avg = 0.0
    cpu_max = 0.0
    if cpu_rows:
        cpu_df = pd.DataFrame(cpu_rows)
        for col in ["agnCpuUtilizationAverage", "agnCpuUtilizationMax"]:
            if col in cpu_df.columns:
                cpu_df[col] = pd.to_numeric(cpu_df[col], errors="coerce")
        cpu_avg = float(cpu_df.get("agnCpuUtilizationAverage", pd.Series([0])).mean())
        cpu_max = float(cpu_df.get("agnCpuUtilizationMax", pd.Series([0])).max())
        if cpu_max >= cpu_alert_pct:
            alerts.append(
                {
                    "severity": "medium",
                    "message": f"CPU max utilization {cpu_max:.1f}% exceeds {cpu_alert_pct:.1f}%.",
                }
            )

    memory_rows = [r for r in records if r["table_name"] == "memoryUsageTable"]
    memory_used_pct_avg = 0.0
    memory_used_pct_max = 0.0
    if memory_rows:
        mem_df = pd.DataFrame(memory_rows)
        for col in ["memoryUsageTotal", "memoryUsageFree"]:
            if col in mem_df.columns:
                mem_df[col] = pd.to_numeric(mem_df[col], errors="coerce")
        mem_df = mem_df[(mem_df.get("memoryUsageTotal", 0) > 0)]
        if not mem_df.empty:
            mem_df["used_pct"] = (1 - (mem_df["memoryUsageFree"] / mem_df["memoryUsageTotal"])) * 100
            memory_used_pct_avg = float(mem_df["used_pct"].mean())
            memory_used_pct_max = float(mem_df["used_pct"].max())
            if memory_used_pct_max >= memory_alert_pct:
                alerts.append(
                    {
                        "severity": "medium",
                        "message": (
                            f"Memory utilization peak {memory_used_pct_max:.1f}% exceeds {memory_alert_pct:.1f}%."
                        ),
                    }
                )

    disk_rows = [r for r in records if r["table_name"] == "agnDiskResourcesTable"]
    disk_used_pct_avg = 0.0
    disk_used_pct_max = 0.0
    if disk_rows:
        disk_df = pd.DataFrame(disk_rows)
        for col in ["agnDiskTotalSpace", "agnDiskAvailableSpace"]:
            if col in disk_df.columns:
                disk_df[col] = pd.to_numeric(disk_df[col], errors="coerce")
        disk_df = disk_df[(disk_df.get("agnDiskTotalSpace", 0) > 0)]
        if not disk_df.empty:
            disk_df["used_pct"] = (1 - (disk_df["agnDiskAvailableSpace"] / disk_df["agnDiskTotalSpace"])) * 100
            disk_used_pct_avg = float(disk_df["used_pct"].mean())
            disk_used_pct_max = float(disk_df["used_pct"].max())
            if disk_used_pct_max >= disk_alert_pct:
                alerts.append(
                    {
                        "severity": "medium",
                        "message": f"Disk utilization peak {disk_used_pct_max:.1f}% exceeds {disk_alert_pct:.1f}%.",
                    }
                )

    risk_score = _pm_risk_score(
        discard_delta=total_discard_delta,
        crc_delta=total_crc_delta,
        cpu_max=cpu_max,
        memory_max=memory_used_pct_max,
        disk_max=disk_used_pct_max,
    )
    risk_band = _risk_band(risk_score)
    hypotheses = _pm_hypotheses(
        discard_delta=total_discard_delta,
        crc_delta=total_crc_delta,
        cpu_max=cpu_max,
        memory_max=memory_used_pct_max,
        disk_max=disk_used_pct_max,
    )

    return {
        "summary": {
            "records": int(len(records)),
            "table_types": int(len(table_counts)),
            "discard_delta_total": round(total_discard_delta, 2),
            "error_delta_total": round(total_error_delta, 2),
            "traffic_delta_total": round(total_traffic_delta, 2),
            "discard_abs_total": round(total_discard_abs, 2),
            "error_abs_total": round(total_error_abs, 2),
            "traffic_abs_total": round(total_traffic_abs, 2),
            "interface_signal_basis": interface_signal_basis,
            "interface_error_signal": interface_error_signal,
            "interface_error_signal_label": interface_error_signal_label,
            "crc_delta_total": round(total_crc_delta, 2),
            "cpu_avg_pct": round(cpu_avg, 2),
            "cpu_max_pct": round(cpu_max, 2),
            "memory_avg_used_pct": round(memory_used_pct_avg, 2),
            "memory_max_used_pct": round(memory_used_pct_max, 2),
            "disk_avg_used_pct": round(disk_used_pct_avg, 2),
            "disk_max_used_pct": round(disk_used_pct_max, 2),
            "risk_score": risk_score,
            "risk_band": risk_band,
        },
        "table_counts": table_counts,
        "top_interface_discards": top_interface_discards,
        "l2_system_hypotheses": hypotheses,
        "alerts": alerts,
    }


def _parse_pm_export_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = df.fillna("").astype(str).values.tolist()
    known_header_starts = {
        "twampControllerId",
        "ifIndex",
        "etherStatsIndex",
        "agnCpuUtilizationCpuIdx",
        "memoryUsageEntityId",
        "agnDiskIdx",
        "snmpEngineTime",
    }

    records: list[dict[str, Any]] = []
    n = len(rows)
    i = 0
    while i < n:
        row = [c.strip() for c in rows[i]]
        if row and row[0] == "Table Name" and i + 1 < n:
            meta = [c.strip() for c in rows[i + 1]]
            table_name = meta[0] if meta else ""
            time_utc = meta[3] if len(meta) > 3 else ""
            j = i + 2
            while j < n:
                rr = [c.strip() for c in rows[j]]
                if rr and rr[0] == "Table Name":
                    break
                if rr and rr[0] in known_header_starts:
                    header_positions = [idx for idx, name in enumerate(rr) if name]
                    headers = [rr[idx] for idx in header_positions]
                    k = j + 1
                    while k < n:
                        data_row = [c.strip() for c in rows[k]]
                        if not data_row or not data_row[0] or data_row[0] == "Table Name":
                            break
                        if data_row[0] in known_header_starts:
                            break
                        rec = {"table_name": table_name, "time_utc": time_utc}
                        for pos, h in zip(header_positions, headers):
                            if pos < len(data_row):
                                rec[h] = data_row[pos]
                        records.append(rec)
                        k += 1
                    j = k
                    continue
                j += 1
            i = j
            continue
        i += 1
    return records


def _pm_risk_score(
    discard_delta: float,
    crc_delta: float,
    cpu_max: float,
    memory_max: float,
    disk_max: float,
) -> int:
    s1 = min(1.0, discard_delta / 5000.0) * 35
    s2 = min(1.0, crc_delta / 500.0) * 25
    s3 = min(1.0, cpu_max / 100.0) * 15
    s4 = min(1.0, memory_max / 100.0) * 15
    s5 = min(1.0, disk_max / 100.0) * 10
    return int(round(s1 + s2 + s3 + s4 + s5))


def _pm_hypotheses(
    discard_delta: float,
    crc_delta: float,
    cpu_max: float,
    memory_max: float,
    disk_max: float,
) -> list[dict[str, str]]:
    hypotheses: list[dict[str, str]] = []
    if crc_delta > 0:
        hypotheses.append(
            {
                "hypothesis": "CRC/align error growth suggests potential physical link quality issues.",
                "confidence": "medium",
            }
        )
    if discard_delta > 0 and crc_delta == 0:
        hypotheses.append(
            {
                "hypothesis": "Packet discard growth with clean CRC hints queue/policy congestion more than layer-1 faults.",
                "confidence": "high",
            }
        )
    if cpu_max >= 80:
        hypotheses.append(
            {
                "hypothesis": "High CPU peaks may contribute to control-plane pressure during busy intervals.",
                "confidence": "medium",
            }
        )
    if memory_max >= 85 or disk_max >= 85:
        hypotheses.append(
            {
                "hypothesis": "Resource pressure (memory/disk) may reduce system headroom and increase instability risk.",
                "confidence": "medium",
            }
        )
    if not hypotheses:
        hypotheses.append(
            {
                "hypothesis": "No strong degradation signal across L2 counters and system resources in this window.",
                "confidence": "low",
            }
        )
    return hypotheses
