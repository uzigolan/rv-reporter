from __future__ import annotations

from typing import Any

import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    columns = []
    numeric_columns = []
    datetime_like_columns = []
    missing_percent = {}

    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_percent[col] = round(float(df[col].isna().mean()) * 100, 2)
        column_meta = {"name": col, "dtype": dtype, "missing_percent": missing_percent[col]}
        columns.append(column_meta)
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)
        if "date" in col.lower() or "time" in col.lower():
            datetime_like_columns.append(col)

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": columns,
        "numeric_columns": numeric_columns,
        "datetime_like_columns": datetime_like_columns,
        "missing_percent": missing_percent,
    }
