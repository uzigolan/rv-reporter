from __future__ import annotations

from typing import Any

import pandas as pd

from rv_reporter.services.metrics import compute_legacy_metrics


def _parse_msg_type(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text, 0)
    except Exception:  # noqa: BLE001
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except Exception:  # noqa: BLE001
        return None
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except Exception:  # noqa: BLE001
        return None
    try:
        return int(float(value))
    except Exception:  # noqa: BLE001
        return None


def _percentile(sorted_values: list[float], p: float) -> float | None:
    if not sorted_values:
        return None
    idx = int((len(sorted_values) - 1) * p)
    return float(sorted_values[idx])


def _epoch_to_utc_str(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    ts = pd.to_datetime(epoch, unit="s", errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return str(ts)


def _sequence_continuity(rows: pd.DataFrame, msg_type: int, label: str) -> dict[str, Any]:
    selected = rows[rows["ptp_message_type_num"] == msg_type].copy()
    selected = selected.dropna(subset=["frame_time_epoch_num", "ptp_sequence_id_num"])
    selected = selected.sort_values("frame_time_epoch_num")
    seq_values = [int(v) for v in selected["ptp_sequence_id_num"].tolist()]
    if len(seq_values) <= 1:
        return {
            "message_type": label,
            "packets": len(seq_values),
            "gaps": 0,
            "estimated_missing_sequences": 0,
            "out_of_order": 0,
            "wrap_events": 0,
        }

    gaps = 0
    missing = 0
    out_of_order = 0
    wraps = 0
    prev = seq_values[0]
    for cur in seq_values[1:]:
        delta = cur - prev
        if delta == 1:
            prev = cur
            continue
        if delta > 1:
            gaps += 1
            missing += delta - 1
            prev = cur
            continue
        if prev > 65000 and cur < 1000:
            wraps += 1
            wrapped_delta = (65536 - prev) + cur
            if wrapped_delta > 1:
                gaps += 1
                missing += wrapped_delta - 1
            prev = cur
            continue
        out_of_order += 1
        prev = cur

    return {
        "message_type": label,
        "packets": len(seq_values),
        "gaps": int(gaps),
        "estimated_missing_sequences": int(missing),
        "out_of_order": int(out_of_order),
        "wrap_events": int(wraps),
    }


def _compute_pairwise_delay_metrics(ptp_rows: pd.DataFrame, prefs: dict[str, Any]) -> dict[str, Any]:
    outlier_ns = float(prefs.get("t4_t3_outlier_ns", 60_000.0))
    cluster_gap_s = float(prefs.get("outlier_cluster_gap_s", 1.0))
    max_samples = int(prefs.get("max_outlier_samples", 12))

    ordered = ptp_rows.dropna(subset=["frame_time_epoch_num"]).sort_values("frame_time_epoch_num")
    pending: dict[int, list[dict[str, float | int | None]]] = {}
    pair_rows: list[dict[str, Any]] = []
    unmatched_delay_resp = 0

    for _, row in ordered.iterrows():
        msg = _safe_int(row.get("ptp_message_type_num"))
        seq = _safe_int(row.get("ptp_sequence_id_num"))
        if msg is None or seq is None:
            continue
        if msg == 1:  # Delay_Req
            t3s = _safe_float(row.get("ptp_origin_ts_seconds_num"))
            t3ns = _safe_float(row.get("ptp_origin_ts_nanoseconds_num"))
            if t3s is None or t3ns is None:
                continue
            req = {
                "t3_ns": (t3s * 1_000_000_000.0) + t3ns,
                "frame_time_epoch": _safe_float(row.get("frame_time_epoch_num")),
            }
            pending.setdefault(seq, []).append(req)
        elif msg == 9:  # Delay_Resp
            recv_s = _safe_float(row.get("ptp_dr_receive_ts_seconds_num"))
            recv_ns = _safe_float(row.get("ptp_dr_receive_ts_nanoseconds_num"))
            if recv_s is None or recv_ns is None:
                continue
            queue = pending.get(seq) or []
            if not queue:
                unmatched_delay_resp += 1
                continue
            req = queue.pop(0)
            if not queue:
                pending.pop(seq, None)
            else:
                pending[seq] = queue
            t4 = (recv_s * 1_000_000_000.0) + recv_ns
            t4_t3 = t4 - float(req["t3_ns"])
            pair_rows.append(
                {
                    "seq": seq,
                    "time_epoch": _safe_float(row.get("frame_time_epoch_num")),
                    "t4_t3_ns": float(t4_t3),
                }
            )

    unmatched_delay_req = sum(len(v) for v in pending.values())
    values = sorted(float(r["t4_t3_ns"]) for r in pair_rows)
    stats: dict[str, Any] = {
        "pairs": len(pair_rows),
        "unmatched_delay_req": int(unmatched_delay_req),
        "unmatched_delay_resp": int(unmatched_delay_resp),
        "threshold_outlier_ns": outlier_ns,
        "min_ns": None,
        "p50_ns": None,
        "p90_ns": None,
        "p99_ns": None,
        "p99_9_ns": None,
        "max_ns": None,
        "mean_ns": None,
    }
    if values:
        stats.update(
            {
                "min_ns": float(values[0]),
                "p50_ns": _percentile(values, 0.50),
                "p90_ns": _percentile(values, 0.90),
                "p99_ns": _percentile(values, 0.99),
                "p99_9_ns": _percentile(values, 0.999),
                "max_ns": float(values[-1]),
                "mean_ns": round(float(sum(values) / len(values)), 2),
            }
        )

    outliers = [r for r in pair_rows if float(r["t4_t3_ns"]) > outlier_ns and r["time_epoch"] is not None]
    outliers = sorted(outliers, key=lambda r: float(r["time_epoch"]))
    clusters: list[dict[str, Any]] = []
    if outliers:
        current: list[dict[str, Any]] = [outliers[0]]
        for row in outliers[1:]:
            prev_t = float(current[-1]["time_epoch"])
            cur_t = float(row["time_epoch"])
            if (cur_t - prev_t) <= cluster_gap_s:
                current.append(row)
            else:
                c_values = sorted(float(x["t4_t3_ns"]) for x in current)
                start_t = float(current[0]["time_epoch"])
                end_t = float(current[-1]["time_epoch"])
                clusters.append(
                    {
                        "start_utc": _epoch_to_utc_str(start_t),
                        "end_utc": _epoch_to_utc_str(end_t),
                        "duration_s": round(end_t - start_t, 3),
                        "samples": len(current),
                        "max_t4_t3_ns": float(c_values[-1]),
                        "p99_t4_t3_ns": _percentile(c_values, 0.99),
                    }
                )
                current = [row]
        c_values = sorted(float(x["t4_t3_ns"]) for x in current)
        start_t = float(current[0]["time_epoch"])
        end_t = float(current[-1]["time_epoch"])
        clusters.append(
            {
                "start_utc": _epoch_to_utc_str(start_t),
                "end_utc": _epoch_to_utc_str(end_t),
                "duration_s": round(end_t - start_t, 3),
                "samples": len(current),
                "max_t4_t3_ns": float(c_values[-1]),
                "p99_t4_t3_ns": _percentile(c_values, 0.99),
            }
        )
    clusters = sorted(clusters, key=lambda r: int(r.get("samples") or 0), reverse=True)

    top_outliers = sorted(outliers, key=lambda r: float(r["t4_t3_ns"]), reverse=True)[:max_samples]
    outlier_samples = [
        {
            "time_utc": _epoch_to_utc_str(_safe_float(r.get("time_epoch"))),
            "sequence_id": int(r["seq"]),
            "t4_t3_ns": float(r["t4_t3_ns"]),
        }
        for r in top_outliers
    ]

    per_hour: list[dict[str, Any]] = []
    if pair_rows:
        with_time = [r for r in pair_rows if r["time_epoch"] is not None]
        if with_time:
            start = min(float(r["time_epoch"]) for r in with_time)
            buckets: dict[int, list[float]] = {}
            for r in with_time:
                hour = int((float(r["time_epoch"]) - start) / 3600.0)
                buckets.setdefault(hour, []).append(float(r["t4_t3_ns"]))
            for hour, series in sorted(buckets.items(), key=lambda x: x[0]):
                s = sorted(series)
                per_hour.append(
                    {
                        "hour_from_start": int(hour),
                        "samples": len(s),
                        "p99_ns": _percentile(s, 0.99),
                        "p99_9_ns": _percentile(s, 0.999),
                        "max_ns": float(s[-1]),
                        "gt_threshold_count": int(sum(1 for v in s if v > outlier_ns)),
                    }
                )

    return {
        "t4_t3_stats": stats,
        "t4_t3_spike_windows": clusters[:max_samples],
        "t4_t3_outlier_samples": outlier_samples,
        "t4_t3_hourly": per_hour,
    }


def _compute_correction_anomalies(ptp_rows: pd.DataFrame, prefs: dict[str, Any]) -> dict[str, Any]:
    threshold = float(prefs.get("correction_anomaly_ns", 1_000_000.0))
    max_samples = int(prefs.get("max_outlier_samples", 12))
    delay_resp = ptp_rows[ptp_rows["ptp_message_type_num"] == 9].copy()
    correction = pd.to_numeric(delay_resp.get("ptp_correction_ns_num"), errors="coerce")
    valid = delay_resp[correction.notna()].copy()
    valid["ptp_correction_ns_num"] = correction[correction.notna()]
    anomalous = valid[valid["ptp_correction_ns_num"] > threshold].copy()
    anomalous = anomalous.sort_values("ptp_correction_ns_num", ascending=False)
    values = sorted(float(v) for v in anomalous["ptp_correction_ns_num"].tolist())
    return {
        "threshold_ns": threshold,
        "count": int(len(anomalous)),
        "max_ns": float(values[-1]) if values else None,
        "p99_ns": _percentile(values, 0.99),
        "samples": [
            {
                "time_utc": _epoch_to_utc_str(_safe_float(r.get("frame_time_epoch_num"))),
                "sequence_id": _safe_int(r.get("ptp_sequence_id_num")),
                "correction_ns": _safe_float(r.get("ptp_correction_ns_num")),
            }
            for _, r in anomalous.head(max_samples).iterrows()
        ],
    }


def _build_ptp1588_local(df: pd.DataFrame, prefs: dict[str, Any]) -> dict[str, Any]:
    working = df.copy()
    for col in [
        "frame_time_epoch",
        "ptp_message_type",
        "ptp_sequence_id",
        "ptp_correction_ns",
        "ptp_origin_ts_seconds",
        "ptp_origin_ts_nanoseconds",
        "ptp_dr_receive_ts_seconds",
        "ptp_dr_receive_ts_nanoseconds",
    ]:
        if col not in working.columns:
            working[col] = pd.NA
    working["frame_time_epoch_num"] = pd.to_numeric(working["frame_time_epoch"], errors="coerce")
    working["ptp_sequence_id_num"] = pd.to_numeric(working["ptp_sequence_id"], errors="coerce")
    working["ptp_correction_ns_num"] = pd.to_numeric(working["ptp_correction_ns"], errors="coerce")
    working["ptp_origin_ts_seconds_num"] = pd.to_numeric(working["ptp_origin_ts_seconds"], errors="coerce")
    working["ptp_origin_ts_nanoseconds_num"] = pd.to_numeric(working["ptp_origin_ts_nanoseconds"], errors="coerce")
    working["ptp_dr_receive_ts_seconds_num"] = pd.to_numeric(working["ptp_dr_receive_ts_seconds"], errors="coerce")
    working["ptp_dr_receive_ts_nanoseconds_num"] = pd.to_numeric(working["ptp_dr_receive_ts_nanoseconds"], errors="coerce")
    working["ptp_message_type_num"] = working["ptp_message_type"].apply(_parse_msg_type)

    if "transport" not in working.columns:
        working["transport"] = ""
    if "frame_protocols" not in working.columns:
        working["frame_protocols"] = ""
    ptp_mask = (
        working["transport"].fillna("").astype(str).str.upper().eq("PTP")
        | working["frame_protocols"].fillna("").astype(str).str.contains(
            r"(?:^|[:;,\s])ptp(?:$|[:;,\s])",
            case=False,
            regex=True,
        )
        | pd.to_numeric(working.get("src_port"), errors="coerce").fillna(-1).isin([319, 320])
        | pd.to_numeric(working.get("dst_port"), errors="coerce").fillna(-1).isin([319, 320])
    )
    ptp_rows = working[ptp_mask].copy()

    sequence = [
        _sequence_continuity(ptp_rows, 0, "Sync"),
        _sequence_continuity(ptp_rows, 1, "Delay_Req"),
        _sequence_continuity(ptp_rows, 9, "Delay_Resp"),
        _sequence_continuity(ptp_rows, 11, "Announce"),
    ]
    pair_metrics = _compute_pairwise_delay_metrics(ptp_rows, prefs)
    correction_metrics = _compute_correction_anomalies(ptp_rows, prefs)

    sequence_gaps = int(sum(int(item["gaps"]) for item in sequence))
    severe_corr = int(correction_metrics.get("count") or 0)
    spike_windows = pair_metrics.get("t4_t3_spike_windows", [])
    alerts: list[dict[str, str]] = []
    if severe_corr > 0:
        alerts.append(
            {
                "severity": "high",
                "message": (
                    f"Delay_Resp correction-field anomalies detected ({severe_corr} above "
                    f"{int(float(correction_metrics['threshold_ns']))} ns)."
                ),
            }
        )
    if spike_windows:
        alerts.append(
            {
                "severity": "medium",
                "message": (
                    f"T4 (t4-t3) outlier windows detected ({len(spike_windows)} clusters above "
                    f"{int(float(pair_metrics['t4_t3_stats']['threshold_outlier_ns']))} ns)."
                ),
            }
        )
    if sequence_gaps > 0:
        alerts.append(
            {
                "severity": "high",
                "message": f"PTP sequence gaps detected across message classes (total gap events: {sequence_gaps}).",
            }
        )

    return {
        "ptp1588_local_summary": {
            "ptp_packets": int(len(ptp_rows)),
            "sequence_gap_events": sequence_gaps,
            "t4_t3_pairs": int(pair_metrics["t4_t3_stats"]["pairs"]),
            "correction_anomaly_events": severe_corr,
        },
        "ptp1588_sequence_continuity": sequence,
        "ptp1588_t4_t3_analysis": pair_metrics,
        "ptp1588_correction_anomalies": correction_metrics,
        "ptp1588_alerts": alerts,
    }


def get_spec() -> dict[str, Any]:
    return {
        "metrics_profile": "ptp1588",
        "api_version": 1,
        "title": "IEEE 1588 PTP G.8275.1 Analysis",
        "description": "IEEE 1588 PTP G.8275.1 Analysis plugin.",
    }


def build(df: pd.DataFrame, prefs: dict[str, Any], ctx: Any) -> dict[str, Any]:
    base = compute_legacy_metrics("wireshark_capture_health", df, prefs)
    local = _build_ptp1588_local(df, prefs)
    merged = dict(base)
    merged.update(local)
    merged_alerts = list(base.get("alerts", []))
    merged_alerts.extend(local.get("ptp1588_alerts", []))
    merged["alerts"] = merged_alerts
    return merged
