from typing import Dict, List
import pandas as pd


class SchemaValidationError(Exception):
    pass




def _require_columns(columns: List[str], df_columns: List[str]):
    missing = [c for c in columns if c not in df_columns]
    if missing:
        raise SchemaValidationError(f"Column(s) not found in dataset: {', '.join(missing)}")


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _is_numeric_convertible(series: pd.Series) -> bool:
    try:
        pd.to_numeric(series.dropna().iloc[:50])
        return True
    except Exception:
        return False


def _is_datetime_convertible(series: pd.Series) -> bool:
    try:
        pd.to_datetime(series.dropna().iloc[:50])
        return True
    except Exception:
        return False




def validate_schema(schema: Dict, df: pd.DataFrame):

    if not isinstance(schema, dict):
        raise SchemaValidationError("Planner did not return a valid schema object")

    required_fields = ["highcharts_type", "data_shape", "data_mapping"]
    for field in required_fields:
        if field not in schema:
            raise SchemaValidationError(f"Schema missing required field: {field}")

    data_shape = schema["data_shape"]
    mapping = schema["data_mapping"]
    aggregation = schema.get("aggregation")

    df_columns = list(df.columns)

    
    if data_shape == "one_dimensional_numeric":

        col = mapping.get("numeric_column")
        if not col:
            raise SchemaValidationError("numeric_column required")

        _require_columns([col], df_columns)

        if not (_is_numeric(df[col]) or _is_numeric_convertible(df[col])):
            raise SchemaValidationError(f"Column '{col}' must be numeric")

    
    elif data_shape == "categorical_numeric":

        cat = mapping.get("category_column")
        if not cat:
            raise SchemaValidationError("category_column required")

        _require_columns([cat], df_columns)


        if aggregation and aggregation != "count":
            num = mapping.get("numeric_column")
            if not num:
                raise SchemaValidationError("numeric_column required for aggregation")

            _require_columns([num], df_columns)

            if not (_is_numeric(df[num]) or _is_numeric_convertible(df[num])):
                raise SchemaValidationError(f"Column '{num}' must be numeric")

    
    elif data_shape == "two_dimensional_numeric":

        x = mapping.get("x_column")
        y = mapping.get("y_column")

        if not x or not y:
            raise SchemaValidationError("x_column and y_column required")

        _require_columns([x, y], df_columns)

        for c in [x, y]:
            if not (_is_numeric(df[c]) or _is_numeric_convertible(df[c])):
                raise SchemaValidationError(f"Column '{c}' must be numeric")

        
        group = mapping.get("group_by_column")
        if group:
            _require_columns([group], df_columns)

    
    
    elif data_shape == "three_dimensional_numeric":

        cols = [mapping.get("x_column"), mapping.get("y_column"), mapping.get("z_column")]

        if not all(cols):
            raise SchemaValidationError("x_column, y_column and z_column required")

        _require_columns(cols, df_columns)

        for c in cols:
            if not (_is_numeric(df[c]) or _is_numeric_convertible(df[c])):
                raise SchemaValidationError(f"Column '{c}' must be numeric")

    
    elif data_shape == "time_series":

        dt = mapping.get("datetime_column")
        num = mapping.get("numeric_column")

        if not dt or not num:
            raise SchemaValidationError("datetime_column and numeric_column required")

        _require_columns([dt, num], df_columns)

        if not _is_datetime_convertible(df[dt]):
            raise SchemaValidationError(f"Column '{dt}' is not a valid datetime column")

        if not (_is_numeric(df[num]) or _is_numeric_convertible(df[num])):
            raise SchemaValidationError(f"Column '{num}' must be numeric")


    elif data_shape == "hierarchical":

        parent = mapping.get("parent_column")
        child = mapping.get("child_column")
        value = mapping.get("numeric_column")

        if not all([parent, child, value]):
            raise SchemaValidationError("parent_column, child_column, numeric_column required")

        _require_columns([parent, child, value], df_columns)

        if not (_is_numeric(df[value]) or _is_numeric_convertible(df[value])):
            raise SchemaValidationError(f"Column '{value}' must be numeric")

    else:
        raise SchemaValidationError(f"Unsupported data_shape '{data_shape}'")

    
    
    if aggregation:
        allowed = {"mean", "sum", "count", "min", "max"}
        if aggregation.lower() not in allowed:
            raise SchemaValidationError(f"Unsupported aggregation '{aggregation}'")

    return True
