from __future__ import annotations

import io
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

_PCAP_COLUMN_MAP: list[tuple[str, str]] = [
    ("frame_number", "frame.number"),
    ("frame_time_epoch", "frame.time_epoch"),
    ("frame_len", "frame.len"),
    ("eth_src", "eth.src"),
    ("eth_dst", "eth.dst"),
    ("src_ip", "ip.src"),
    ("ipv6_src", "ipv6.src"),
    ("dst_ip", "ip.dst"),
    ("ipv6_dst", "ipv6.dst"),
    ("ip_proto", "ip.proto"),
    ("transport", "_ws.col.protocol"),
    ("tcp_srcport", "tcp.srcport"),
    ("tcp_dstport", "tcp.dstport"),
    ("udp_srcport", "udp.srcport"),
    ("udp_dstport", "udp.dstport"),
    ("tcp_flags_syn", "tcp.flags.syn"),
    ("tcp_flags_ack", "tcp.flags.ack"),
    ("tcp_flags_fin", "tcp.flags.fin"),
    ("tcp_flags_reset", "tcp.flags.reset"),
    ("dns_query", "dns.qry.name"),
    ("http_host", "http.host"),
    ("http_uri", "http.request.uri"),
    ("tls_sni", "tls.handshake.extensions_server_name"),
    ("icmp_type", "icmp.type"),
    ("arp_opcode", "arp.opcode"),
    ("frame_protocols", "frame.protocols"),
]

_PCAP_OPTIONAL_COLUMN_MAP: list[tuple[str, tuple[str, ...]]] = [
    ("ptp_message_type", ("ptp.v2.messagetype", "ptp.messagetype")),
    ("ptp_domain_number", ("ptp.v2.domainnumber",)),
    ("ptp_sequence_id", ("ptp.v2.sequenceid", "ptp.sequenceid")),
    ("ptp_source_port_identity", ("ptp.v2.sourceportid", "ptp.sourceportid")),
    ("ptp_correction_ns", ("ptp.v2.correction.ns",)),
    ("ptp_origin_ts_seconds", ("ptp.v2.sdr.origintimestamp.seconds", "ptp.sdr.origintimestamp_seconds")),
    ("ptp_origin_ts_nanoseconds", ("ptp.v2.sdr.origintimestamp.nanoseconds", "ptp.sdr.origintimestamp_nanoseconds")),
    ("ptp_dr_receive_ts_seconds", ("ptp.v2.dr.receivetimestamp.seconds",)),
    ("ptp_dr_receive_ts_nanoseconds", ("ptp.v2.dr.receivetimestamp.nanoseconds",)),
    ("ptp_dr_requesting_source_port_identity", ("ptp.v2.dr.requestingsourceportidentity",)),
    ("ptp_dr_requesting_source_port_id", ("ptp.v2.dr.requestingsourceportid",)),
    ("ptp_two_step", ("ptp.v2.flags.twostep",)),
]

_PCAP_COLUMNS = [
    "frame_number",
    "frame_time_epoch",
    "frame_len",
    "eth_src",
    "eth_dst",
    "src_ip",
    "dst_ip",
    "ip_proto",
    "transport",
    "src_port",
    "dst_port",
    "tcp_flags_syn",
    "tcp_flags_ack",
    "tcp_flags_fin",
    "tcp_flags_reset",
    "dns_query",
    "http_host",
    "http_uri",
    "tls_sni",
    "icmp_type",
    "arp_opcode",
    "frame_protocols",
    "ptp_message_type",
    "ptp_domain_number",
    "ptp_sequence_id",
    "ptp_source_port_identity",
    "ptp_correction_ns",
    "ptp_origin_ts_seconds",
    "ptp_origin_ts_nanoseconds",
    "ptp_dr_receive_ts_seconds",
    "ptp_dr_receive_ts_nanoseconds",
    "ptp_dr_requesting_source_port_identity",
    "ptp_dr_requesting_source_port_id",
    "ptp_two_step",
]


def describe_tabular_source(path: str | Path, sheet_name: str | None = None) -> dict[str, object]:
    data_path = Path(path)
    suffix = data_path.suffix.lower()
    row_count: int | None = _source_row_count(data_path, sheet_name=sheet_name)
    if suffix == ".csv":
        header = _read_delimited_auto(data_path, nrows=0)
        header = _normalize_wireshark_export_frame(header)
        return {
            "file_type": "csv",
            "sheets": [],
            "selected_sheet": "",
            "columns": [str(c) for c in header.columns],
            "row_count": row_count,
        }
    if suffix in {".xlsx", ".xls"}:
        workbook = pd.ExcelFile(data_path)
        sheets = list(workbook.sheet_names)
        selected_sheet = (sheet_name or "").strip()
        if not selected_sheet and len(sheets) == 1:
            selected_sheet = sheets[0]
        columns: list[str] = []
        if selected_sheet:
            header = pd.read_excel(data_path, sheet_name=selected_sheet, nrows=0)
            columns = [str(c) for c in header.columns]
        return {
            "file_type": "excel",
            "sheets": sheets,
            "selected_sheet": selected_sheet,
            "columns": columns,
            "row_count": row_count,
        }
    if suffix in {".pcap", ".pcapng"}:
        return {
            "file_type": "pcap",
            "sheets": [],
            "selected_sheet": "",
            "columns": list(_PCAP_COLUMNS),
            "row_count": row_count,
        }
    return {
        "file_type": "unknown",
        "sheets": [],
        "selected_sheet": "",
        "columns": [],
        "row_count": None,
    }


def _source_row_count(path: Path, sheet_name: str | None = None) -> int | None:
    suffix = path.suffix.lower()
    if suffix in {".pcap", ".pcapng"}:
        fast_count = _pcap_packet_count(path)
        if fast_count is not None:
            return fast_count
    try:
        frame = load_csv_with_limit(path, row_limit=None, sheet_name=sheet_name)
        return int(len(frame))
    except Exception:  # noqa: BLE001
        return None


def _pcap_packet_count(path: Path) -> int | None:
    tshark_exe = _resolve_tshark_executable()
    tshark_path = Path(tshark_exe.strip('"'))
    candidates: list[str] = ["capinfos"]
    if tshark_path.name:
        sibling = tshark_path.with_name("capinfos.exe" if os.name == "nt" else "capinfos")
        candidates.insert(0, str(sibling))

    for exe in candidates:
        try:
            completed = subprocess.run(
                [exe, "-c", str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:  # noqa: BLE001
            continue
        if completed.returncode != 0:
            continue
        text = (completed.stdout or "") + "\n" + (completed.stderr or "")
        match = re.search(r"Number of packets:\s*([0-9,]+)", text, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            return int(match.group(1).replace(",", ""))
        except Exception:  # noqa: BLE001
            continue
    return None


def list_excel_sheets(path: str | Path) -> list[str]:
    data_path = Path(path)
    if data_path.suffix.lower() not in {".xlsx", ".xls"}:
        return []
    workbook = pd.ExcelFile(data_path)
    return list(workbook.sheet_names)


def load_csv(path: str | Path, sheet_name: str | None = None) -> pd.DataFrame:
    return load_csv_with_limit(path, row_limit=None, sheet_name=sheet_name)


def load_csv_with_limit(
    path: str | Path,
    row_limit: int | None = None,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Input file not found: {data_path}")
    if row_limit is None:
        nrows = None
    else:
        if row_limit <= 0:
            raise ValueError("row_limit must be > 0.")
        nrows = row_limit

    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        frame = _read_delimited_auto(data_path, nrows=nrows)
        return _normalize_wireshark_export_frame(frame)
    if suffix in {".xlsx", ".xls"}:
        frame = _read_excel(data_path, sheet_name=sheet_name, nrows=nrows)
        return _normalize_wireshark_export_frame(frame)
    if suffix in {".pcap", ".pcapng"}:
        return _read_pcap(data_path, nrows=nrows)

    raise ValueError("Unsupported file type. Supported: .csv, .xlsx, .xls, .pcap, .pcapng")


def preflight_tabular_source(
    path: str | Path,
    *,
    sheet_name: str | None = None,
    required_columns: list[str] | None = None,
    row_limit: int | None = 250,
) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    metadata = describe_tabular_source(path, sheet_name=sheet_name)
    try:
        frame = load_csv_with_limit(path, row_limit=row_limit, sheet_name=sheet_name)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "issues": [str(exc)],
            "warnings": [],
            "metadata": metadata,
            "stats": {},
        }

    if frame.empty:
        issues.append("Source parsed successfully but produced no data rows.")

    required = [str(item).strip() for item in (required_columns or []) if str(item).strip()]
    if required:
        missing = [col for col in required if col not in frame.columns]
        if missing:
            issues.append(f"Missing required columns: {', '.join(missing)}.")

    repeated_header_rows = _count_duplicate_header_rows(frame)
    if repeated_header_rows:
        issues.append(
            f"Parsed data still contains {repeated_header_rows} repeated header row(s), which usually means the source has multiple embedded tables."
        )

    metadata_rows = _count_metadata_rows(frame)
    if metadata_rows:
        issues.append(
            f"Parsed data still contains {metadata_rows} metadata row(s), which means the source was not cleanly isolated to one table."
        )

    suspicious_numeric = _find_suspicious_numeric_columns(frame)
    if suspicious_numeric:
        formatted = ", ".join(
            f"{name} ({ratio}% numeric)" for name, ratio in suspicious_numeric[:5]
        )
        warnings.append(
            "Some numeric-looking columns contain mostly non-numeric values after parsing: "
            f"{formatted}."
        )

    unnamed_columns = [str(col) for col in frame.columns if str(col).startswith("Unnamed:")]
    if unnamed_columns and len(unnamed_columns) >= max(3, len(frame.columns) // 2):
        warnings.append(
            "Parsed table contains many unnamed columns. This often indicates the wrong header row was selected."
        )

    return {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "metadata": metadata,
        "stats": {
            "row_count": int(len(frame.index)),
            "column_count": int(len(frame.columns)),
            "repeated_header_rows": repeated_header_rows,
            "metadata_rows": metadata_rows,
        },
    }


def _read_excel(path: Path, sheet_name: str | None, nrows: int | None) -> pd.DataFrame:
    normalized_sheet = (sheet_name or "").strip()
    if normalized_sheet:
        return pd.read_excel(path, sheet_name=normalized_sheet, nrows=nrows)

    workbook = pd.ExcelFile(path)
    sheets = workbook.sheet_names
    if len(sheets) <= 1:
        return pd.read_excel(path, sheet_name=sheets[0] if sheets else 0, nrows=nrows)

    raise ValueError(
        "Excel file has multiple sheets. Please provide 'Sheet Name'. "
        f"Available sheets: {sheets}"
    )


def _read_delimited_auto(path: Path, nrows: int | None) -> pd.DataFrame:
    frame = _read_pm_export_format(path, nrows=nrows)
    if frame is not None:
        return frame

    return pd.read_csv(path, nrows=nrows, sep=None, engine="python")


def _read_pm_export_format(path: Path, nrows: int | None) -> pd.DataFrame | None:
    try:
        raw_frame = pd.read_csv(path, dtype=str, na_filter=False, header=None, sep=None, engine="python")
        if raw_frame.empty or len(raw_frame.columns) == 0:
            return None
        if not _has_structured_table_markers(raw_frame):
            return None
        section = _extract_structured_table_section(raw_frame)
        if section is None or section.empty:
            return None
        if nrows is not None and len(section) > nrows:
            section = section.iloc[:nrows].reset_index(drop=True)
        # Return result with correct columns even if empty (for nrows=0 case)
        return section
    except Exception:
        return None


def _extract_structured_table_section(raw_frame: pd.DataFrame) -> pd.DataFrame | None:
    candidates: list[tuple[int, pd.DataFrame]] = []
    row_count = len(raw_frame.index)
    if row_count < 2:
        return None

    for idx in range(row_count - 1):
        header_idx: int | None = None
        row_values = _row_values(raw_frame.iloc[idx])
        next_values = _row_values(raw_frame.iloc[idx + 1])
        if _is_oid_row(row_values) and _looks_like_table_header(next_values):
            header_idx = idx + 1
        elif _looks_like_table_header(row_values) and _looks_like_data_row(next_values, row_values):
            header_idx = idx
        if header_idx is None:
            continue

        header_values = _sanitize_header_values(raw_frame.iloc[header_idx])
        if len(header_values) < 2:
            continue

        data_rows: list[list[str]] = []
        for data_idx in range(header_idx + 1, row_count):
            values = _row_values(raw_frame.iloc[data_idx])
            if _is_blank_row(values):
                if data_rows:
                    break
                continue
            if _is_metadata_like_row(values):
                if data_rows:
                    break
                continue
            if _is_oid_row(values):
                if data_rows:
                    break
                continue
            if _looks_like_table_header(values):
                if data_rows:
                    break
                continue
            data_rows.append(values[: len(header_values)] + [""] * max(0, len(header_values) - len(values)))

        if not data_rows:
            continue

        section = pd.DataFrame(data_rows, columns=header_values)
        section = _remove_empty_rows(section)
        section = _remove_duplicate_header_rows(section)
        section = _remove_metadata_rows(section)
        if section.empty:
            continue
        score = (len(header_values) * 1000) + len(section.index)
        candidates.append((score, section.reset_index(drop=True)))

    if not candidates:
        return None
    # Merge candidates that share the same column headers (same table
    # repeated across measurement intervals in PM CSV-ES exports).
    merged: dict[tuple[str, ...], list[pd.DataFrame]] = {}
    for _score, section in candidates:
        key = tuple(section.columns)
        merged.setdefault(key, []).append(section)
    combined: list[tuple[int, pd.DataFrame]] = []
    for cols_key, frames in merged.items():
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        score = (len(cols_key) * 1000) + len(df.index)
        combined.append((score, df))
    combined.sort(key=lambda item: item[0], reverse=True)
    return combined[0][1]


def _row_values(row: pd.Series) -> list[str]:
    return [str(value).strip() if pd.notna(value) else "" for value in row.tolist()]


def _is_blank_row(values: list[str]) -> bool:
    return not any(values)


def _is_metadata_like_row(values: list[str]) -> bool:
    first_value = next((value for value in values if value), "")
    if not first_value:
        return False
    return any(first_value.startswith(pattern) for pattern in _METADATA_PREFIXES)


def _is_oid_token(value: str) -> bool:
    if not value or "." not in value or not value[0].isdigit():
        return False
    parts = [part for part in value.split(".") if part]
    return bool(parts) and all(part.isdigit() for part in parts)


def _is_oid_row(values: list[str]) -> bool:
    nonempty = [value for value in values if value]
    if len(nonempty) < 2:
        return False
    dotted_oid_count = sum(1 for value in nonempty if _is_oid_token(value))
    if dotted_oid_count >= max(1, len(nonempty) // 4):
        return True
    integer_tokens = [int(value) for value in nonempty if re.fullmatch(r"\d+", value)]
    if len(integer_tokens) != len(nonempty):
        return False
    return integer_tokens == sorted(integer_tokens) and max(integer_tokens, default=0) <= max(10, len(integer_tokens) * 3)


def _looks_like_table_header(values: list[str]) -> bool:
    nonempty = [value for value in values if value]
    if len(nonempty) < 2 or _is_metadata_like_row(values) or _is_oid_row(values):
        return False
    identifier_like = 0
    for value in nonempty:
        if _is_oid_token(value) or re.fullmatch(r"[+-]?\d+(\.\d+)?([Ee][+-]?\d+)?", value):
            continue
        if any(char.isalpha() for char in value):
            identifier_like += 1
    return identifier_like >= max(2, min(6, max(2, len(nonempty) // 2)))


def _looks_like_data_row(values: list[str], header_values: list[str]) -> bool:
    nonempty = [value for value in values if value]
    if not nonempty or _is_metadata_like_row(values):
        return False
    normalized_header = {value.strip().lower() for value in header_values if value.strip()}
    overlap = sum(1 for value in nonempty if value.strip().lower() in normalized_header)
    return overlap < max(1, len(nonempty) // 2)


def _sanitize_header_values(row: pd.Series) -> list[str]:
    values = _row_values(row)
    last_nonempty = max((idx for idx, value in enumerate(values) if value), default=-1)
    if last_nonempty < 0:
        return []

    headers: list[str] = []
    seen: dict[str, int] = {}
    for idx, value in enumerate(values[: last_nonempty + 1]):
        base_name = value or f"Unnamed: {idx}"
        suffix = seen.get(base_name, 0)
        seen[base_name] = suffix + 1
        headers.append(base_name if suffix == 0 else f"{base_name}.{suffix}")
    return headers


def _has_structured_table_markers(raw_frame: pd.DataFrame) -> bool:
    sample_size = min(len(raw_frame.index), 40)
    saw_metadata = False
    saw_oid_row = False
    for idx in range(sample_size):
        values = _row_values(raw_frame.iloc[idx])
        if _is_metadata_like_row(values):
            saw_metadata = True
        if _is_oid_row(values):
            saw_oid_row = True
        if saw_metadata and saw_oid_row:
            return True
    return False


def _remove_empty_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    mask = frame.astype(str).apply(lambda row: not all(str(v).strip() == "" for v in row), axis=1)
    return frame[mask].reset_index(drop=True)


def _remove_metadata_rows_by_pattern(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or len(frame.columns) == 0:
        return frame

    mask = ~_metadata_row_mask(frame)
    return frame[mask].reset_index(drop=True)


def _read_pcap(path: Path, nrows: int | None) -> pd.DataFrame:
    tshark_exe = _resolve_tshark_executable()
    command = [
        tshark_exe,
        "-r",
        str(path),
        "-T",
        "fields",
        "-E",
        "header=y",
        "-E",
        "separator=/t",
        "-E",
        "quote=n",
        "-E",
        "occurrence=f",
    ]
    if nrows is not None:
        command.extend(["-c", str(nrows)])
    for _, tshark_field in _PCAP_COLUMN_MAP:
        command.extend(["-e", tshark_field])
    available_fields = _tshark_field_catalog(tshark_exe)
    selected_optional: list[tuple[str, str]] = []
    for internal_name, candidates in _PCAP_OPTIONAL_COLUMN_MAP:
        selected = next((field for field in candidates if field in available_fields), "")
        if selected:
            selected_optional.append((internal_name, selected))
    for _, tshark_field in selected_optional:
        command.extend(["-e", tshark_field])

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ValueError(
            "tshark was not found. Set TSHARK_PATH in .env.sandbox "
            "(for example: C:\\Program Files\\Wireshark\\tshark.exe) "
            "or install Wireshark/tshark and ensure it is in PATH."
        ) from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise ValueError(
            "Failed to parse capture with tshark. "
            + (f"Details: {stderr}" if stderr else "Check capture file integrity and tshark installation.")
        )

    raw = completed.stdout or ""
    if not raw.strip():
        return pd.DataFrame(columns=_PCAP_COLUMNS)
    frame = pd.read_csv(io.StringIO(raw), dtype=str, sep="\t", na_filter=False)
    # Fallback for unexpected tshark formatting variants.
    if len(frame.columns) <= 1:
        frame = pd.read_csv(io.StringIO(raw), dtype=str, sep=None, engine="python", na_filter=False)
    rename_pairs = _PCAP_COLUMN_MAP + selected_optional
    frame = frame.rename(columns={tshark_field: internal_name for internal_name, tshark_field in rename_pairs if tshark_field in frame.columns})

    if "src_ip" not in frame.columns:
        frame["src_ip"] = ""
    if "dst_ip" not in frame.columns:
        frame["dst_ip"] = ""
    frame["src_ip"] = frame["src_ip"].fillna("")
    frame["dst_ip"] = frame["dst_ip"].fillna("")
    if "ipv6_src" in frame.columns:
        frame["src_ip"] = frame["src_ip"].where(frame["src_ip"] != "", frame["ipv6_src"].fillna(""))
        frame = frame.drop(columns=["ipv6_src"])
    if "ipv6_dst" in frame.columns:
        frame["dst_ip"] = frame["dst_ip"].where(frame["dst_ip"] != "", frame["ipv6_dst"].fillna(""))
        frame = frame.drop(columns=["ipv6_dst"])

    if "src_port" not in frame.columns:
        frame["src_port"] = ""
    if "dst_port" not in frame.columns:
        frame["dst_port"] = ""
    frame["src_port"] = frame["src_port"].fillna("")
    frame["dst_port"] = frame["dst_port"].fillna("")
    if "tcp_srcport" in frame.columns:
        frame["src_port"] = frame["src_port"].where(frame["src_port"] != "", frame["tcp_srcport"].fillna(""))
        frame = frame.drop(columns=["tcp_srcport"])
    if "udp_srcport" in frame.columns:
        frame["src_port"] = frame["src_port"].where(frame["src_port"] != "", frame["udp_srcport"].fillna(""))
        frame = frame.drop(columns=["udp_srcport"])
    if "tcp_dstport" in frame.columns:
        frame["dst_port"] = frame["dst_port"].where(frame["dst_port"] != "", frame["tcp_dstport"].fillna(""))
        frame = frame.drop(columns=["tcp_dstport"])
    if "udp_dstport" in frame.columns:
        frame["dst_port"] = frame["dst_port"].where(frame["dst_port"] != "", frame["udp_dstport"].fillna(""))
        frame = frame.drop(columns=["udp_dstport"])

    for col in _PCAP_COLUMNS:
        if col not in frame.columns:
            frame[col] = ""
    return frame[_PCAP_COLUMNS]


def _resolve_tshark_executable() -> str:
    configured = (os.getenv("TSHARK_PATH", "") or os.getenv("RV_TSHARK_PATH", "")).strip()
    if not configured:
        return "tshark"
    candidate = Path(configured.strip('"'))
    if candidate.name.lower() == "wireshark.exe":
        candidate = candidate.with_name("tshark.exe")
    return str(candidate)


@lru_cache(maxsize=4)
def _tshark_field_catalog(tshark_exe: str) -> set[str]:
    try:
        completed = subprocess.run(
            [tshark_exe, "-G", "fields"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return set()
    if completed.returncode != 0:
        return set()
    fields: set[str] = set()
    for line in (completed.stdout or "").splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        if parts[0] != "F":
            continue
        name = parts[2].strip()
        if name:
            fields.add(name)
    return fields


def _remove_duplicate_header_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    mask = ~frame.apply(lambda row: _matches_header_row(row, frame.columns), axis=1)
    return frame[mask].reset_index(drop=True)


def _remove_metadata_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or len(frame.columns) == 0:
        return frame

    mask = ~_metadata_row_mask(frame)
    return frame[mask].reset_index(drop=True)


def _normalize_wireshark_export_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty and not list(frame.columns):
        return frame

    frame = _remove_duplicate_header_rows(frame)
    frame = _remove_metadata_rows(frame)

    cols = {str(c).strip().lower(): c for c in frame.columns}
    wireshark_alias_hits = sum(1 for name in ("time", "length", "source", "destination", "protocol", "info", "no.", "no") if name in cols)
    canonical_hits = sum(1 for name in ("frame_time_epoch", "frame_len", "src_ip", "dst_ip", "transport") if name in frame.columns)
    if wireshark_alias_hits < 3 and canonical_hits < 3:
        return frame

    alias_map = {
        "time": "frame_time_epoch",
        "length": "frame_len",
        "source": "src_ip",
        "destination": "dst_ip",
        "protocol": "transport",
        "info": "info",
        "no.": "frame_number",
        "no": "frame_number",
    }

    rename_map: dict[str, str] = {}
    for alias, canonical in alias_map.items():
        src = cols.get(alias)
        if src is not None and src != canonical and canonical not in frame.columns:
            rename_map[src] = canonical
    if rename_map:
        frame = frame.rename(columns=rename_map)

    expected_hit = sum(1 for name in ("frame_time_epoch", "frame_len", "src_ip", "dst_ip", "transport") if name in frame.columns)
    if expected_hit < 3:
        return frame

    for col in (
        "frame_time_epoch",
        "frame_len",
        "src_ip",
        "dst_ip",
        "transport",
        "src_port",
        "dst_port",
        "tcp_flags_syn",
        "tcp_flags_ack",
        "tcp_flags_fin",
        "tcp_flags_reset",
        "dns_query",
        "http_host",
        "http_uri",
        "tls_sni",
        "icmp_type",
        "arp_opcode",
        "frame_protocols",
    ):
        if col not in frame.columns:
            frame[col] = ""

    if "frame_protocols" in frame.columns:
        frame["frame_protocols"] = frame["frame_protocols"].fillna("").astype(str)
    frame["frame_protocols"] = frame["frame_protocols"].where(
        frame["frame_protocols"].str.strip() != "",
        frame["transport"].fillna("").astype(str).str.lower(),
    )

    # Parse ports and TCP flags from Wireshark "Info" text where available.
    if "info" in frame.columns:
        info = frame["info"].fillna("").astype(str)
        ports = info.str.extract(r"(?P<src>\d+)\s*[→>]\s*(?P<dst>\d+)", expand=True)
        frame["src_port"] = frame["src_port"].where(frame["src_port"].astype(str).str.strip() != "", ports["src"].fillna(""))
        frame["dst_port"] = frame["dst_port"].where(frame["dst_port"].astype(str).str.strip() != "", ports["dst"].fillna(""))

        upper = info.str.upper()
        frame["tcp_flags_syn"] = frame["tcp_flags_syn"].where(frame["tcp_flags_syn"].astype(str).str.strip() != "", upper.str.contains(r"\bSYN\b").astype(int).astype(str))
        frame["tcp_flags_ack"] = frame["tcp_flags_ack"].where(frame["tcp_flags_ack"].astype(str).str.strip() != "", upper.str.contains(r"\bACK\b").astype(int).astype(str))
        frame["tcp_flags_fin"] = frame["tcp_flags_fin"].where(frame["tcp_flags_fin"].astype(str).str.strip() != "", upper.str.contains(r"\bFIN\b").astype(int).astype(str))
        frame["tcp_flags_reset"] = frame["tcp_flags_reset"].where(frame["tcp_flags_reset"].astype(str).str.strip() != "", upper.str.contains(r"\bRST\b").astype(int).astype(str))

        dns_q = info.str.extract(r"(?i)standard\s+query\s+[0-9a-fx]+\s+(.+)$", expand=False).fillna("")
        frame["dns_query"] = frame["dns_query"].where(frame["dns_query"].astype(str).str.strip() != "", dns_q)

    return frame


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


_METADATA_PREFIXES = (
    "Table Name",
    "Interval State",
    "Version",
    "Device ID",
    "Device ",
    "Entry OID",
    "Date And Time",
    "System Uptime",
)

_NUMERIC_COLUMN_HINT = re.compile(
    r"(count|delay|loss|rate|packet|pkt|byte|octet|speed|latency|jitter|sum|avg|average|max|min|second|error|ratio|score|util|throughput|duration)",
    flags=re.IGNORECASE,
)


def _matches_header_row(row: pd.Series, columns: Any) -> bool:
    row_values = [str(value).strip() if pd.notna(value) else "" for value in row]
    header_values = [str(value).strip() for value in columns]
    return row_values == header_values


def _metadata_row_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty or len(frame.columns) == 0:
        return pd.Series(dtype=bool)
    first_col = frame.iloc[:, 0]
    return first_col.apply(
        lambda value: any(
            str(value).strip().startswith(prefix) for prefix in _METADATA_PREFIXES if str(value).strip()
        )
    )


def _count_duplicate_header_rows(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return int(frame.apply(lambda row: _matches_header_row(row, frame.columns), axis=1).sum())


def _count_metadata_rows(frame: pd.DataFrame) -> int:
    if frame.empty or len(frame.columns) == 0:
        return 0
    return int(_metadata_row_mask(frame).sum())


def _find_suspicious_numeric_columns(frame: pd.DataFrame) -> list[tuple[str, int]]:
    suspicious: list[tuple[str, int]] = []
    if frame.empty:
        return suspicious

    for column in frame.columns:
        column_name = str(column).strip()
        if not _NUMERIC_COLUMN_HINT.search(column_name):
            continue
        series = frame[column].fillna("").astype(str).str.strip()
        nonempty = series[series != ""]
        if len(nonempty.index) < 3:
            continue
        numeric = pd.to_numeric(nonempty, errors="coerce")
        numeric_ratio = int(round((numeric.notna().sum() / len(nonempty.index)) * 100))
        if numeric_ratio < 60:
            suspicious.append((column_name, numeric_ratio))
    suspicious.sort(key=lambda item: item[1])
    return suspicious
