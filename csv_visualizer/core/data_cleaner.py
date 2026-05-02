import pandas as pd
import numpy as np
import re




_NUMERIC_REGEX = re.compile(r"^-?\d+(\.\d+)?$")


def _looks_numeric(series: pd.Series) -> bool:
    """Check if a column mostly contains numeric-like strings"""
    if series.dtype != object:
        return False

    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False

    matches = sample.apply(lambda x: bool(_NUMERIC_REGEX.match(x.strip())))
    return matches.mean() > 0.8


def _looks_datetime(series: pd.Series) -> bool:
    """Check if a column likely represents dates"""
    if series.dtype != object:
        return False

    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False

    parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
    return parsed.notna().mean() > 0.7



def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely normalize dataframe types without changing semantics.
    This prevents plotting & validation failures.
    """

    df = df.copy()

    
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    
    df.dropna(how="all", inplace=True)

    
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan}, inplace=True)

    
    for col in df.columns:
        if _looks_numeric(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    
    for col in df.columns:
        if _looks_datetime(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    
    for col in df.columns:
        if df[col].dtype == object:
            lower = df[col].str.lower()
            if lower.isin(["true", "false"]).mean() > 0.8:
                df[col] = lower.map({"true": True, "false": False})

    return df
