from __future__ import annotations

from pathlib import Path

import pandas as pd


def describe_tabular_source(path: str | Path, sheet_name: str | None = None) -> dict[str, object]:
    data_path = Path(path)
    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        header = pd.read_csv(data_path, nrows=0)
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
        return pd.read_csv(data_path, nrows=nrows)
    if suffix in {".xlsx", ".xls"}:
        return _read_excel(data_path, sheet_name=sheet_name, nrows=nrows)

    raise ValueError("Unsupported file type. Supported: .csv, .xlsx, .xls")


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


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
