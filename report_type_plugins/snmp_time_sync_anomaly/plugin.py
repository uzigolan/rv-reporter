import pandas as pd
import numpy as np

def get_spec():
    return {
        "id": "snmp_time_sync_anomaly",
        "metrics_profile": "snmp_time_sync_anomaly",
        "api_version": 1,
        "title": "SNMP Time Sync & Performance — Anomaly Detection",
        "family": "time_series",
        "domain": "networking",
        "mode": "anomaly_detection",
        "required_columns": [
            "Entry OID",
            "Date And Time (Local)",
            "Date And Time (UTC)",
            "System Uptime (Seconds)",
            "Device ID",
            "Interval Length (Seconds)"
        ]
    }


def _find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # try case-insensitive match
    cols_lower = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def _to_seconds(x):
    try:
        return float(x)
    except Exception:
        # strip non-digits
        try:
            s = str(x)
            s = ''.join(ch for ch in s if (ch.isdigit() or ch == '.' or ch == '-'))
            return float(s) if s not in ("", "-") else np.nan
        except Exception:
            return np.nan


def build(df, prefs=None, ctx=None):
    """
    df: pandas.DataFrame with the SNMP export.
    prefs: dict of preferences (see default_prefs in report definition)
    ctx: unused context placeholder

    Returns a dict with deterministic metrics and top anomaly samples.
    """
    if prefs is None:
        prefs = {}
    # merge defaults
    defaults = {"interval_seconds": 60, "gap_multiplier": 1.5, "drift_threshold_seconds": 5, "top_n_samples": 10, "min_rows_for_device": 3}
    for k, v in defaults.items():
        prefs.setdefault(k, v)

    # Work on a copy
    df = df.copy()

    # Find likely columns
    local_col = _find_column(df, ["Date And Time (Local)", "Date And Time (Local Time)", "Date And Time (Local)", "Date And Time", "Date And Time (Local) "])
    utc_col = _find_column(df, ["Date And Time (UTC)", "Date And Time (UTC)", "Date And Time (UTC)", "Date And Time (UTC) "])
    uptime_col = _find_column(df, ["System Uptime (Seconds)", "System Uptime", "System Uptime (Seconds) "])
    device_col = _find_column(df, ["Device ID", "Device ID: 00-20-D2-61-5C-2A", "Device", "Agent"])
    entry_oid_col = _find_column(df, ["Entry OID", "EntryOID", "OID"])
    interval_col = _find_column(df, ["Interval Length (Seconds)", "Interval Length", "Interval Length (Seconds)"])

    # Parse timestamps
    df["__ts_local"] = pd.to_datetime(df[local_col], errors="coerce") if local_col is not None else pd.NaT
    df["__ts_utc"] = pd.to_datetime(df[utc_col], errors="coerce") if utc_col is not None else pd.NaT

    # Parse uptime
    if uptime_col is not None:
        df["__uptime_s"] = pd.to_numeric(df[uptime_col].apply(_to_seconds), errors="coerce")
    else:
        df["__uptime_s"] = np.nan

    # Device id fallback
    if device_col is None:
        # try common columns
        device_col = _find_column(df, ["Agent ID", "Host", "Hostname"]) or df.columns[0]
    df["__device"] = df[device_col].astype(str)

    # interval detection
    interval_seconds = prefs.get("interval_seconds")
    # If there is a column or a row like 'Interval Length (Seconds): 60', try to parse
    if interval_col is not None:
        # attempt to coerce first non-null cell
        val = df[interval_col].dropna().astype(str)
        if not val.empty:
            # try find digits
            import re
            m = re.search(r"(\d+)", val.iloc[0])
            if m:
                try:
                    interval_seconds = int(m.group(1))
                except Exception:
                    pass

    # Prepare result containers
    overall = {
        "row_count": int(len(df)),
        "device_count": int(df["__device"].nunique()),
        "interval_seconds_used": int(interval_seconds)
    }

    device_summaries = {}
    total_gaps = 0
    total_resets = 0
    total_drift = 0
    total_duplicates = 0

    anomaly_samples = []

    # Work group-by-device for deterministic ordering: sorted device names
    for dev in sorted(df["__device"].unique()):
        ddf = df[df["__device"] == dev].copy()
        # Sort by local timestamp then utc for determinism
        ddf = ddf.sort_values(["__ts_local", "__ts_utc"], na_position="last")
        row_count = int(len(ddf))
        if row_count == 0:
            continue

        # compute time range
        ts_min = ddf["__ts_local"].min()
        ts_max = ddf["__ts_local"].max()

        # compute diffs between consecutive non-null timestamps
        ts = ddf["__ts_local"].astype('datetime64[ns]')
        diffs = ts.diff().dt.total_seconds().fillna(0)

        # missing/gap where diff > gap_multiplier * interval_seconds
        gap_thresh = prefs.get("gap_multiplier") * interval_seconds
        gaps_mask = diffs > gap_thresh
        gap_count = int(gaps_mask.sum())

        # duplicates where diff == 0 (and timestamp not null)
        dup_count = int((diffs == 0).sum())

        # uptime resets: where uptime decreases compared to previous
        uptime = ddf["__uptime_s"].fillna(method='ffill')
        # If all NaN, then resets_count = 0
        if uptime.isna().all():
            resets_count = 0
        else:
            uptime_diff = uptime.diff()
            resets_count = int((uptime_diff < 0).sum())

        # clock drift: abs(local - utc)
        if ddf["__ts_local"].isna().all() or ddf["__ts_utc"].isna().all():
            drift_count = 0
            drift_median = None
        else:
            drift_seconds = (ddf["__ts_local"].dt.tz_localize(None) - ddf["__ts_utc"].dt.tz_localize(None)).abs().dt.total_seconds()
            drift_threshold = prefs.get("drift_threshold_seconds")
            drift_count = int((drift_seconds > drift_threshold).sum())
            drift_median = float(drift_seconds.median()) if not drift_seconds.dropna().empty else None

        # Uptime stats
        uptime_vals = ddf["__uptime_s"].dropna()
        uptime_stats = None
        if not uptime_vals.empty:
            uptime_stats = {
                "min": float(uptime_vals.min()),
                "max": float(uptime_vals.max()),
                "mean": float(uptime_vals.mean()),
                "median": float(uptime_vals.median())
            }

        device_summary = {
            "device": dev,
            "rows": int(row_count),
            "time_range": {"start": str(ts_min) if pd.notnull(ts_min) else None, "end": str(ts_max) if pd.notnull(ts_max) else None},
            "gap_count": gap_count,
            "duplicate_timestamps": dup_count,
            "uptime_resets": resets_count,
            "drift_count": drift_count,
            "drift_median_seconds": drift_median,
            "uptime_stats": uptime_stats
        }

        device_summaries[dev] = device_summary

        total_gaps += gap_count
        total_resets += resets_count
        total_drift += drift_count
        total_duplicates += dup_count

        # collect top anomaly samples for this device deterministically
        # Prioritize: largest gap diffs, largest drift_seconds, reset events
        # Prepare candidate samples
        # gaps: get rows where gaps_mask True, include previous and current rows to show context
        gap_idx = np.where(gaps_mask)[0]
        for gi in gap_idx[: prefs.get("top_n_samples")]:
            prev_idx = max(gi - 1, 0)
            sample_rows = []
            for idx in [prev_idx, gi]:
                if idx < len(ddf):
                    row = ddf.iloc[idx].to_dict()
                    # stringify timestamps for JSON-safe output
                    row["__ts_local"] = str(row.get("__ts_local"))
                    row["__ts_utc"] = str(row.get("__ts_utc"))
                    sample_rows.append(row)
            anomaly_samples.append({"type": "gap", "device": dev, "gap_seconds": float(diffs.iloc[gi]), "context": sample_rows})

        # drifts: compute per-row drift and add top ones
        if not (ddf["__ts_local"].isna().all() or ddf["__ts_utc"].isna().all()):
            drift_seconds = (ddf["__ts_local"].dt.tz_localize(None) - ddf["__ts_utc"].dt.tz_localize(None)).abs().dt.total_seconds()
            drift_df = ddf.assign(_drift_seconds=drift_seconds)
            drift_df = drift_df.sort_values("_drift_seconds", ascending=False)
            for _, r in drift_df.head(prefs.get("top_n_samples")).iterrows():
                if pd.isna(r.get('_drift_seconds')):
                    continue
                anomaly_samples.append({
                    "type": "drift",
                    "device": dev,
                    "drift_seconds": float(r.get('_drift_seconds')),
                    "row": {
                        "__ts_local": str(r.get('__ts_local')),
                        "__ts_utc": str(r.get('__ts_utc')),
                        "__uptime_s": (float(r.get('__uptime_s')) if not pd.isna(r.get('__uptime_s')) else None),
                        "entry_oid": r.get(entry_oid_col) if entry_oid_col in ddf.columns else None
                    }
                })

        # resets: include the rows where uptime decreased
        if not uptime_vals.empty:
            uid = ddf["__uptime_s"].fillna(method='ffill')
            dif = uid.diff()
            reset_idxs = np.where(dif < 0)[0]
            for ri in reset_idxs[: prefs.get("top_n_samples")]:
                r = ddf.iloc[ri]
                anomaly_samples.append({
                    "type": "uptime_reset",
                    "device": dev,
                    "row_index": int(r.name) if hasattr(r, 'name') else None,
                    "row": {
                        "__ts_local": str(r.get('__ts_local')),
                        "__uptime_s": (float(r.get('__uptime_s')) if not pd.isna(r.get('__uptime_s')) else None),
                        "entry_oid": r.get(entry_oid_col) if entry_oid_col in ddf.columns else None
                    }
                })

    overall.update({
        "total_gaps": int(total_gaps),
        "total_uptime_resets": int(total_resets),
        "total_large_drifts": int(total_drift),
        "total_duplicate_timestamps": int(total_duplicates)
    })

    # Deterministic ordering of anomaly samples: sort by type then device then numeric field descending
    def _sample_sort_key(s):
        t = s.get('type', '')
        dev = s.get('device', '')
        # numeric priority
        priority = 0
        if t == 'gap':
            priority = -float(s.get('gap_seconds', 0))
        elif t == 'drift':
            priority = -float(s.get('drift_seconds', 0))
        else:
            priority = 0
        return (t, dev, priority)

    anomaly_samples = sorted(anomaly_samples, key=_sample_sort_key)
    anomaly_samples = anomaly_samples[: prefs.get('top_n_samples')]

    result = {
        "spec": get_spec(),
        "overall": overall,
        "devices": device_summaries,
        "anomaly_samples": anomaly_samples
    }
    return result
