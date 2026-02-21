from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path

import pandas as pd

_PCAP_COLUMN_MAP: list[tuple[str, str]] = [
    ("frame_time_epoch", "frame.time_epoch"),
    ("frame_len", "frame.len"),
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

_PCAP_COLUMNS = [
    "frame_time_epoch",
    "frame_len",
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
]


def describe_tabular_source(path: str | Path, sheet_name: str | None = None) -> dict[str, object]:
    data_path = Path(path)
    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        header = _read_delimited_auto(data_path, nrows=0)
        header = _normalize_wireshark_export_frame(header)
        return {
            "file_type": "csv",
            "sheets": [],
            "selected_sheet": "",
            "columns": [str(c) for c in header.columns],
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
        }
    if suffix in {".pcap", ".pcapng"}:
        return {
            "file_type": "pcap",
            "sheets": [],
            "selected_sheet": "",
            "columns": list(_PCAP_COLUMNS),
        }
    return {
        "file_type": "unknown",
        "sheets": [],
        "selected_sheet": "",
        "columns": [],
    }


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
    # Wireshark exports are commonly tab-delimited even with .csv extension.
    return pd.read_csv(path, nrows=nrows, sep=None, engine="python")


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
    frame = frame.rename(columns={tshark_field: internal_name for internal_name, tshark_field in _PCAP_COLUMN_MAP if tshark_field in frame.columns})

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


def _normalize_wireshark_export_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty and not list(frame.columns):
        return frame

    cols = {str(c).strip().lower(): c for c in frame.columns}
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

    # If this is not a Wireshark-export-like table, keep original frame unchanged.
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
