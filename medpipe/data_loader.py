"""
Data loading utilities.

Supports .sav (SPSS), .csv, .tsv, .xlsx/.xls. For .sav files, SPSS value labels
are read but the numeric codes are kept (the pipeline's preprocessing agent does
its own encoding). Returns a pandas DataFrame plus an optional metadata dict.
"""
from __future__ import annotations
import os


def load_any(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".sav":
        import pyreadstat
        df, meta = pyreadstat.read_sav(path)
        info = {"value_labels": getattr(meta, "variable_value_labels", {}),
                "column_labels": dict(zip(meta.column_names, meta.column_labels))
                if getattr(meta, "column_labels", None) else {}}
        return df, info
    if ext in (".csv", ".tsv"):
        import pandas as pd
        sep = "\t" if ext == ".tsv" else ","
        return pd.read_csv(path, sep=sep), {}
    if ext in (".xlsx", ".xls"):
        import pandas as pd
        return pd.read_excel(path), {}
    raise ValueError(f"Unsupported file extension: {ext}")


def binarize_target(series, positive_from: float = 1.0):
    """Collapse an ordinal severity target (0..k) to binary present/absent.

    In the HOPE data, symptom targets use 0 = none/no-data and 1..3 = present at
    increasing severity, so `>= 1` means the symptom is present.
    """
    return (series >= positive_from).astype(int)
