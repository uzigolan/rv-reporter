"""Shared build logic for assisted-living SLA plugins.

Canonical 12-section analysis: call volume, response times, risk scoring,
staffing optimisation, regulatory mapping, etc.

Expected CSV columns: timestamp, device_name, property
Property values: "Call Button Pressed", "Call Accepted", "Resolution Confirmed"
device_name prefixes: "Rm …" = resident, "Caregiver …" = caregiver
"""
from __future__ import annotations

from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_incidents(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Group raw rows into 4-step incident dicts (call→accept→res_resident→res_caregiver)."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)

    incidents: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None

    for _, row in df.iterrows():
        prop = str(row["property"]).strip()
        dev = str(row["device_name"]).strip()
        ts = row["timestamp"]

        if prop == "Call Button Pressed" and dev.startswith("Rm"):
            if pending is not None:
                pending["complete"] = False
                incidents.append(pending)
            pending = {"resident": dev, "call_pressed": ts, "caregiver": None,
                       "call_accepted": None, "res_resident": None,
                       "res_caregiver": None, "complete": False}
        elif prop == "Call Accepted" and dev.startswith("Caregiver") and pending is not None:
            pending["caregiver"] = dev
            pending["call_accepted"] = ts
        elif prop == "Resolution Confirmed" and pending is not None:
            if dev.startswith("Rm"):
                pending["res_resident"] = ts
            elif dev.startswith("Caregiver"):
                pending["res_caregiver"] = ts
                pending["complete"] = all(pending[k] is not None for k in
                                          ("call_pressed", "call_accepted", "res_resident", "res_caregiver"))
                incidents.append(pending)
                pending = None

    if pending is not None:
        incidents.append(pending)

    return incidents


def _minutes(a, b) -> float | None:
    if a is None or b is None:
        return None
    return round((b - a).total_seconds() / 60, 2)


def _risk_score(avg_response: float | None, call_count: int, sla_pct: float) -> int:
    score = 0
    if avg_response is not None:
        if avg_response > 10:
            score += 40
        elif avg_response > 7:
            score += 25
        elif avg_response > 5:
            score += 10
    if call_count > 15:
        score += 30
    elif call_count > 10:
        score += 20
    elif call_count > 5:
        score += 10
    if sla_pct < 60:
        score += 30
    elif sla_pct < 80:
        score += 15
    return min(score, 100)


def _risk_label(score: int) -> str:
    if score >= 60:
        return "High"
    if score >= 30:
        return "Moderate"
    return "Low"


def _hour_label(h: int) -> str:
    if 6 <= h < 12:
        return "Morning"
    if 12 <= h < 18:
        return "Afternoon"
    if 18 <= h < 22:
        return "Evening"
    return "Night"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_assisted_living_metrics(df: pd.DataFrame, prefs: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical assisted-living metrics dict."""
    sla_response_min = float(prefs.get("sla_response_threshold_min", 10))
    sla_resolution_min = float(prefs.get("sla_resolution_threshold_min", 30))

    incidents = _parse_incidents(df)
    total = len(incidents)
    complete = [i for i in incidents if i["complete"]]
    incomplete = [i for i in incidents if not i["complete"]]

    # Per-incident derived values
    for inc in incidents:
        inc["response_min"] = _minutes(inc["call_pressed"], inc["call_accepted"])
        inc["resolution_min"] = _minutes(inc["call_pressed"], inc["res_caregiver"] or inc["res_resident"])

    complete_count = len(complete)
    sla_pct = round(100 * complete_count / total, 1) if total else 0.0

    response_times = [i["response_min"] for i in incidents if i["response_min"] is not None]
    resolution_times = [i["resolution_min"] for i in incidents if i["resolution_min"] is not None]
    avg_response = round(sum(response_times) / len(response_times), 2) if response_times else None
    avg_resolution = round(sum(resolution_times) / len(resolution_times), 2) if resolution_times else None

    overall_risk = _risk_score(avg_response, total, sla_pct)
    risk_level = _risk_label(overall_risk)

    # --- Section 2: Call Volume per Resident ---
    resident_calls: dict[str, int] = {}
    for inc in incidents:
        r = inc["resident"]
        resident_calls[r] = resident_calls.get(r, 0) + 1
    call_volume_per_resident = [{"resident": r, "call_count": c} for r, c in sorted(resident_calls.items(), key=lambda x: -x[1])]

    # --- Section 3: Response Times per Caregiver ---
    cg_resp: dict[str, list[float]] = {}
    for inc in incidents:
        cg = inc.get("caregiver")
        rt = inc["response_min"]
        if cg and rt is not None:
            cg_resp.setdefault(cg, []).append(rt)
    response_times_per_caregiver = [
        {"caregiver": cg, "avg_response_min": round(sum(ts) / len(ts), 2),
         "min_response_min": round(min(ts), 2), "max_response_min": round(max(ts), 2),
         "incidents": len(ts)}
        for cg, ts in sorted(cg_resp.items())
    ]

    # --- Section 4: Resolution Time per Resident ---
    res_resol: dict[str, list[float]] = {}
    for inc in incidents:
        r = inc["resident"]
        rt = inc["resolution_min"]
        if rt is not None:
            res_resol.setdefault(r, []).append(rt)
    resolution_time_per_resident = [
        {"resident": r, "avg_resolution_min": round(sum(ts) / len(ts), 2),
         "min_resolution_min": round(min(ts), 2), "max_resolution_min": round(max(ts), 2),
         "incidents": len(ts)}
        for r, ts in sorted(res_resol.items())
    ]

    # --- Section 5: Peak Demand ---
    hour_counts: dict[str, int] = {}
    for inc in incidents:
        cp = inc["call_pressed"]
        if cp is not None:
            label = _hour_label(cp.hour)
            hour_counts[label] = hour_counts.get(label, 0) + 1
    peak_demand = [{"period": p, "call_count": c} for p, c in sorted(hour_counts.items(), key=lambda x: -x[1])]

    # --- Section 6: Caregiver Performance ---
    cg_perf_data: dict[str, dict] = {}
    for inc in incidents:
        cg = inc.get("caregiver")
        if not cg:
            continue
        d = cg_perf_data.setdefault(cg, {"handled": 0, "complete": 0, "resp": []})
        d["handled"] += 1
        if inc["complete"]:
            d["complete"] += 1
        if inc["response_min"] is not None:
            d["resp"].append(inc["response_min"])
    caregiver_performance = [
        {"caregiver": cg, "incidents_handled": d["handled"],
         "complete_closures": d["complete"],
         "closure_rate_pct": round(100 * d["complete"] / d["handled"], 1) if d["handled"] else 0,
         "avg_response_min": round(sum(d["resp"]) / len(d["resp"]), 2) if d["resp"] else None}
        for cg, d in sorted(cg_perf_data.items())
    ]

    # --- Section 7: Compliance Checks ---
    compliance_checks = [
        {"check": "4-step closure", "passed": complete_count, "failed": len(incomplete),
         "rate_pct": sla_pct},
        {"check": f"Response ≤ {sla_response_min} min",
         "passed": sum(1 for t in response_times if t <= sla_response_min),
         "failed": sum(1 for t in response_times if t > sla_response_min),
         "rate_pct": round(100 * sum(1 for t in response_times if t <= sla_response_min) / len(response_times), 1) if response_times else 0},
        {"check": f"Resolution ≤ {sla_resolution_min} min",
         "passed": sum(1 for t in resolution_times if t <= sla_resolution_min),
         "failed": sum(1 for t in resolution_times if t > sla_resolution_min),
         "rate_pct": round(100 * sum(1 for t in resolution_times if t <= sla_resolution_min) / len(resolution_times), 1) if resolution_times else 0},
    ]

    # --- Section 9: Risk Scoring ---
    resident_risks = []
    for r, count in resident_calls.items():
        r_resp = [i["response_min"] for i in incidents if i["resident"] == r and i["response_min"] is not None]
        r_complete = sum(1 for i in incidents if i["resident"] == r and i["complete"])
        r_sla = round(100 * r_complete / count, 1) if count else 0
        r_avg = round(sum(r_resp) / len(r_resp), 2) if r_resp else None
        rs = _risk_score(r_avg, count, r_sla)
        resident_risks.append({"resident": r, "risk_score": rs, "risk_level": _risk_label(rs),
                                "call_count": count, "sla_pct": r_sla})

    caregiver_risks = []
    for cg, d in cg_perf_data.items():
        cg_sla = round(100 * d["complete"] / d["handled"], 1) if d["handled"] else 0
        cg_avg = round(sum(d["resp"]) / len(d["resp"]), 2) if d["resp"] else None
        cs = _risk_score(cg_avg, d["handled"], cg_sla)
        caregiver_risks.append({"caregiver": cg, "risk_score": cs, "risk_level": _risk_label(cs),
                                 "incidents": d["handled"], "closure_rate_pct": cg_sla})

    # --- Section 10: Staffing Optimization ---
    staffing_optimization = []
    for p, c in sorted(hour_counts.items(), key=lambda x: -x[1]):
        rec_staff = max(1, round(c / max(1, total) * 4))
        staffing_optimization.append({"period": p, "call_count": c, "recommended_staff": rec_staff})

    # --- Section 11: Regulatory Mapping ---
    regulatory_mapping = [
        {"regulation": "Title 22 §87611 — Emergency Call System",
         "requirement": "Response within 10 minutes", "status": "Pass" if (avg_response or 0) <= 10 else "Fail"},
        {"regulation": "Title 22 §87464 — Incident Documentation",
         "requirement": "Four-step closure for all incidents",
         "status": "Pass" if sla_pct >= 95 else "Needs Improvement"},
        {"regulation": "Title 22 §87411 — Staffing Ratios",
         "requirement": "Adequate staffing during peak periods",
         "status": "Review Recommended"},
    ]

    # --- Alerts ---
    alerts: list[dict[str, str]] = []
    if sla_pct < 80:
        alerts.append({"severity": "critical", "message": f"SLA compliance at {sla_pct}% — below 80% threshold."})
    elif sla_pct < 95:
        alerts.append({"severity": "warning", "message": f"SLA compliance at {sla_pct}% — below 95% target."})
    if avg_response and avg_response > sla_response_min:
        alerts.append({"severity": "critical",
                        "message": f"Average response time {avg_response} min exceeds {sla_response_min} min SLA."})
    if len(incomplete) > 0:
        alerts.append({"severity": "warning",
                        "message": f"{len(incomplete)} incident(s) missing complete 4-step closure."})
    for rr in resident_risks:
        if rr["risk_level"] == "High":
            alerts.append({"severity": "warning",
                            "message": f"Resident {rr['resident']} has high risk score ({rr['risk_score']})."})

    # --- Charts ---
    charts: list[dict[str, Any]] = [
        {"chart_type": "bar", "title": "Call Volume per Resident",
         "x_label": "Resident", "y_label": "Calls",
         "data": {r["resident"]: r["call_count"] for r in call_volume_per_resident}},
        {"chart_type": "bar", "title": "Avg Response Time per Caregiver (min)",
         "x_label": "Caregiver", "y_label": "Minutes",
         "data": {c["caregiver"]: c["avg_response_min"] for c in response_times_per_caregiver}},
        {"chart_type": "bar", "title": "Avg Resolution Time per Resident (min)",
         "x_label": "Resident", "y_label": "Minutes",
         "data": {r["resident"]: r["avg_resolution_min"] for r in resolution_time_per_resident}},
        {"chart_type": "bar", "title": "Peak Demand by Period",
         "x_label": "Period", "y_label": "Calls",
         "data": {p["period"]: p["call_count"] for p in peak_demand}},
        {"chart_type": "bar", "title": "Risk Score by Resident",
         "x_label": "Resident", "y_label": "Score (0-100)",
         "data": {r["resident"]: r["risk_score"] for r in resident_risks}},
        {"chart_type": "bar", "title": "Caregiver Closure Rate (%)",
         "x_label": "Caregiver", "y_label": "%",
         "data": {c["caregiver"]: c["closure_rate_pct"] for c in caregiver_performance}},
    ]

    # --- Sections ---
    sections: list[dict[str, str]] = [
        {"title": "Overall Assessment",
         "body": (f"Analyzed {total} incidents across {len(resident_calls)} residents and "
                  f"{len(cg_perf_data)} caregivers. SLA compliance: {sla_pct}%. "
                  f"Average response: {avg_response} min. Average resolution: {avg_resolution} min. "
                  f"Overall risk: {risk_level} ({overall_risk}/100).")},
        {"title": "Call Volume per Resident",
         "body": "Breakdown of total calls initiated by each resident."},
        {"title": "Response Times per Caregiver",
         "body": "Caregiver acceptance speed from call button press."},
        {"title": "Resolution Time per Resident",
         "body": "End-to-end time from call to final resolution per resident."},
        {"title": "Peak Demand",
         "body": "Call distribution across time-of-day periods."},
        {"title": "Caregiver Performance",
         "body": "Per-caregiver closure rates and response statistics."},
        {"title": "Proof of Compliance",
         "body": "SLA pass/fail checks including 4-step closure and time thresholds."},
        {"title": "Two-Step Closure",
         "body": "Dual-confirmation closure: resident and caregiver both confirm resolution."},
        {"title": "Risk Scoring",
         "body": "Composite risk scores (0-100) for residents and caregivers."},
        {"title": "Staffing Optimization",
         "body": "Recommended staff levels per demand period."},
        {"title": "Regulatory Mapping",
         "body": "Alignment with Title 22 California assisted-living regulations."},
        {"title": "AI Recommendations",
         "body": "Data-driven recommendations based on the analysis above."},
    ]

    return {
        "summary": {
            "total_incidents": total,
            "complete_incidents": complete_count,
            "incomplete_incidents": len(incomplete),
            "sla_compliance_pct": sla_pct,
            "average_response_time": avg_response,
            "average_resolution_time": avg_resolution,
            "overall_risk_level": risk_level,
        },
        "call_volume_per_resident": call_volume_per_resident,
        "response_times_per_caregiver": response_times_per_caregiver,
        "resolution_time_per_resident": resolution_time_per_resident,
        "peak_demand": peak_demand,
        "caregiver_performance": caregiver_performance,
        "compliance_checks": compliance_checks,
        "resident_risks": resident_risks,
        "caregiver_risks": caregiver_risks,
        "staffing_optimization": staffing_optimization,
        "regulatory_mapping": regulatory_mapping,
        "alerts": alerts,
        "charts": charts,
        "sections": sections,
    }
