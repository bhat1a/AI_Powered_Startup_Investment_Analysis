import pandas as pd
import re




def _normalize(col: str) -> str:
    """
    Canonical column form used internally
    Example:
        'Order Date ' → 'order_date'
        'Ship-Date'   → 'ship_date'
    """
    col = col.strip().lower()
    col = re.sub(r"[^\w]+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_")


def _is_datetime_series(series: pd.Series) -> bool:
    if series.dtype == "datetime64[ns]":
        return True

    if series.dtype != "object":
        return False

    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False

    try:
        parsed = pd.to_datetime(sample, errors="raise", infer_datetime_format=True)

        
        if parsed.dt.year.nunique() <= 1 and len(sample.unique()) > 5:
            return False

        return True
    except Exception:
        return False




def profile_dataframe(df: pd.DataFrame):

    profile = {
        "row_count": len(df),
        "columns": [],

        
        "column_lookup": {},          
        "normalized_lookup": {},      

        
        "numeric_columns": [],
        "categorical_columns": [],
        "datetime_columns": []
    }

    for col in df.columns:

        actual_name = col
        normalized = _normalize(col)
        natural = col.lower().strip()

        
        profile["column_lookup"][natural] = actual_name
        profile["column_lookup"][normalized.replace("_", " ")] = actual_name
        profile["normalized_lookup"][normalized] = actual_name

        s = df[col]


        if pd.api.types.is_numeric_dtype(s):
            profile["numeric_columns"].append(actual_name)

            profile["columns"].append({
                "name": actual_name,
                "type": "numeric",
                "min": float(s.min()) if not s.empty else None,
                "max": float(s.max()) if not s.empty else None,
                "mean": float(s.mean()) if not s.empty else None,
                "unique": int(s.nunique())
            })
            continue

        
        if _is_datetime_series(s):
            df[col] = pd.to_datetime(df[col], errors="coerce")

            profile["datetime_columns"].append(actual_name)

            profile["columns"].append({
                "name": actual_name,
                "type": "datetime",
                "min": str(df[col].min()) if df[col].notna().any() else None,
                "max": str(df[col].max()) if df[col].notna().any() else None,
                "unique": int(df[col].nunique())
            })
            continue

        
        profile["categorical_columns"].append(actual_name)

        profile["columns"].append({
            "name": actual_name,
            "type": "categorical",
            "unique": int(s.nunique()),
            "examples": s.dropna().astype(str).unique()[:5].tolist()
        })

    return profile