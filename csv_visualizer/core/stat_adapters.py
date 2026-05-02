import pandas as pd
import numpy as np


_VALID_AGGS = {"mean", "sum", "count", "min", "max"}

def compute_stats(chart_type, df, mapping, aggregation=None, binding=None):
    """
    Returns DATA strictly according to binding contract.
    Binding controls output shape.
    """
    if binding is None:
        raise ValueError("Binding required for stat adapter")

    # Normalise aggregation — LLM sometimes returns "none" or None
    if not aggregation or str(aggregation).lower() not in _VALID_AGGS:
        aggregation = "sum"
    if chart_type == "boxplot":
        return _boxplot_grouped(df, mapping)
    # =========================================================
    # BINDING-DRIVEN ROUTING (Authoritative)
    # =========================================================

    if binding == "series_y":
        return _numeric_series(df, mapping)

    if binding == "category_series":
        return _category_series(df, mapping, aggregation)

    if binding == "named_points":
        base = _category_series(df, mapping, aggregation)
        return [
            {"name": c, "y": v}
            for c, v in zip(base["categories"], base["values"])
        ]

    if binding == "xy_pairs":
        return _xy_pairs(df, mapping)

    if binding == "xyz_points":
        return _xyz_points(df, mapping)

    if binding == "datetime_series":
        return _datetime_series(df, mapping, aggregation)

    if binding == "boxplot_array":
        return _boxplot_grouped(df, mapping)

    raise ValueError(f"Unsupported binding {binding}")


# =========================================================
# NUMERIC SERIES
# =========================================================

def _numeric_series(df, mapping):
    col = mapping.get("numeric_column") or mapping.get("y_column")
    return df[col].dropna().astype(float).tolist()


# =========================================================
# CATEGORY SERIES (Aggregated)
# =========================================================

def _category_series(df, mapping, aggregation):
    cat = mapping["category_column"]
    num = mapping.get("numeric_column") or mapping.get("y_column")

    grouped = df.groupby(cat)

    if aggregation == "count":
        result = grouped.size()
    else:
        agg = aggregation if aggregation in _VALID_AGGS else "sum"
        result = getattr(grouped[num], agg)()

    result = result.reset_index(name="value")

    return {
        "categories": result[cat].astype(str).tolist(),
        "values": result["value"].astype(float).tolist()
    }


# =========================================================
# XY PAIRS
# =========================================================

def _xy_pairs(df, mapping):
    x = mapping["x_column"]
    y = mapping["y_column"]

    data = df[[x, y]].dropna()
    return data.astype(float).values.tolist()


# =========================================================
# XYZ POINTS
# =========================================================

def _xyz_points(df, mapping):
    x = mapping["x_column"]
    y = mapping["y_column"]
    z = mapping["z_column"]

    data = df[[x, y, z]].dropna()
    return [
        {"x": float(a), "y": float(b), "z": float(c)}
        for a, b, c in data.values
    ]


# =========================================================
# DATETIME SERIES
# =========================================================

def _datetime_series(df, mapping, aggregation):
    dt = mapping.get("datetime_column") or mapping.get("x_column")
    num = mapping.get("numeric_column") or mapping.get("y_column")

    data = df[[dt, num]].dropna().copy()
    data[dt] = pd.to_datetime(data[dt], errors="coerce")
    data = data.dropna()

    if aggregation:
        grouped = data.groupby(dt)[num]

        if aggregation == "count":
            result = grouped.size()
        else:
            result = getattr(grouped, aggregation)()

        result = result.sort_index()

        return [
            [int(ts.timestamp() * 1000), float(v)]
            for ts, v in result.items()
        ]

    data = data.sort_values(dt)

    return [
        [int(a.timestamp() * 1000), float(b)]
        for a, b in data.values
    ]


# =========================================================
# GROUPED BOXPLOT
# =========================================================

def _boxplot_grouped(df, mapping):
    cat = mapping["category_column"]
    num = mapping["numeric_column"]

    grouped = df.groupby(cat)[num]

    categories = []
    values = []

    for name, group in grouped:
        s = group.dropna().astype(float)

        min_v = float(s.min())
        q1 = float(s.quantile(0.25))
        median = float(s.quantile(0.5))
        q3 = float(s.quantile(0.75))
        max_v = float(s.max())

        categories.append(str(name))
        values.append([min_v, q1, median, q3, max_v])

    return {
        "categories": categories,
        "values": values
    }