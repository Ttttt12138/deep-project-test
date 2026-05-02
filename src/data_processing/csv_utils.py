"""
Shared CSV IO and feature schema helpers.

CSV is the canonical project data format. Files are written with UTF-8 BOM
for safer Excel/Windows handling, and stock codes are always read as strings
so leading zeros are preserved.
"""

from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd


CSV_ENCODING = "utf-8-sig"

META_COLUMNS = frozenset({
    "date",
    "code",
    "time",
    "label",
    "window_start_time",
    "window_end_time",
    "month",
    "Unnamed: 0",
})

NON_FEATURE_COLUMNS = META_COLUMNS | frozenset({
    "current",
    "limit_price",
})


def _csv_dtype(dtype: Optional[Mapping] = None, preserve_code: bool = True) -> dict:
    merged = dict(dtype or {})
    if preserve_code:
        merged["code"] = "string"
    return merged


def read_csv(path: str | Path, *, dtype: Optional[Mapping] = None,
             preserve_code: bool = True, **kwargs) -> pd.DataFrame:
    """Read a project CSV with stable encoding and stock-code typing."""
    kwargs.setdefault("encoding", CSV_ENCODING)
    kwargs["dtype"] = _csv_dtype(dtype, preserve_code=preserve_code)
    return pd.read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path, *, index: bool = False,
              **kwargs) -> None:
    """Write a project CSV with the canonical encoding."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("encoding", CSV_ENCODING)
    df.to_csv(output_path, index=index, **kwargs)


def ensure_code_string(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve stock-code leading zeros after in-memory transformations."""
    if "code" in df.columns:
        df = df.copy()
        df["code"] = df["code"].astype("string")
    return df


def get_feature_columns(df: pd.DataFrame,
                        extra_exclude: Optional[Iterable[str]] = None) -> list[str]:
    """Return model feature columns using the shared non-feature schema."""
    exclude = set(NON_FEATURE_COLUMNS)
    if extra_exclude:
        exclude.update(extra_exclude)
    return [col for col in df.columns if col not in exclude]


def month_dir_from_date(date_str: str) -> str:
    """Return the two-digit month directory for a YYYY-MM-DD style date."""
    parts = str(date_str).split("-")
    if len(parts) >= 2 and parts[1].isdigit():
        return f"{int(parts[1]):02d}"
    return "00"
