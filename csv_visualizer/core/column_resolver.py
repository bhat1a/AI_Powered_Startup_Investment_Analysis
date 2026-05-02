import re


def normalize(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())


def build_column_map(df_columns):
    """
    Maps normalized -> actual column
    """
    mapping = {}
    for col in df_columns:
        mapping[normalize(col)] = col
    return mapping


def resolve_mapping(schema_mapping: dict, df_columns: list) -> dict:

    col_map = build_column_map(df_columns)

    resolved = {}

    for key, col in schema_mapping.items():
        norm = normalize(col)

        if norm not in col_map:
            raise ValueError(f"Column '{col}' not found in dataset")

        resolved[key] = col_map[norm]

    return resolved