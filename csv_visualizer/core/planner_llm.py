import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.cerebras.ai/v1"
)
MODEL = os.getenv("INTENT_MODEL", "llama3.1-8b")
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROMPT_DIR = os.path.join(BASE_DIR, "prompts")



ALLOWED_SHAPES = {
    "one_dimensional_numeric",
    "categorical_numeric",
    "two_dimensional_numeric",
    "three_dimensional_numeric",
    "time_series",
    "hierarchical",
    "specialized"
}

ALLOWED_AGGREGATIONS = {"mean", "sum", "count", "min", "max"}



CHART_TYPE_ALIASES = {
    "pie": ["pie"],
    "bar": ["bar"],
    "column": ["column"],
    "line": ["line"],
    "area": ["area"],
    "scatter": ["scatter"],
    "histogram": ["histogram"],
    "heatmap": ["heatmap"],
    
}

CHART_SHAPE_MAP = {
    "pie": "categorical_numeric",
    "bar": "categorical_numeric",
    "column": "categorical_numeric",
    "line": "time_series",
    "area": "time_series",
    "scatter": "two_dimensional_numeric",
    "histogram": "one_dimensional_numeric",
    
    "treemap": "hierarchical",
    "heatmap": "specialized"
}


SAFE_CONVERSIONS = {
    ("categorical_numeric", "pie"),
    ("categorical_numeric", "bar"),
    ("categorical_numeric", "column"),
    ("categorical_numeric", "area"),
    ("categorical_numeric", "line"),
    ("time_series", "line"),
    ("time_series", "area"),
    ("one_dimensional_numeric", "histogram"),
    ("hierarchical", "treemap"),
    ("specialized", "heatmap"),
}


def _detect_chart_change(query: str):
    q = query.lower()
    for chart, aliases in CHART_TYPE_ALIASES.items():
        for word in aliases:
            if word in q:
                return chart
    return None


def _apply_structural_patch(query: str, previous_state: dict) -> dict | None:
    """
    Conversational structural edit:
    Change chart type ONLY if mapping compatible.
    Otherwise fallback to full LLM replanning.
    """

    new_chart = _detect_chart_change(query)
    if not new_chart:
        return None

    old_shape = previous_state["data_shape"]

    
    if (old_shape, new_chart) not in SAFE_CONVERSIONS:
        return None

    patched = dict(previous_state)
    patched["highcharts_type"] = new_chart
    #patched["data_shape"] = CHART_SHAPE_MAP.get(new_chart, old_shape)

    return patched


def _find_json_object(text: str) -> str:
    start = None
    depth = 0

    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]

    raise ValueError("No complete JSON object found")


def _extract_json(text: str) -> dict:
    text = text.replace("```json", "").replace("```", "")
    return json.loads(_find_json_object(text))



def _attach_binding(schema: dict) -> dict:
    chart = schema["highcharts_type"]
    shape = schema["data_shape"]

    if chart in {"pie", "funnel", "pyramid","treemap"}:
        schema["binding"] = "named_points"
        return schema
    # Heatmap
    if chart == "heatmap":
        schema["binding"] = "xy_pairs"
        return schema
    shape_binding_map = {
        "one_dimensional_numeric": "series_y",
        "categorical_numeric": "category_series",
        "two_dimensional_numeric": "xy_pairs",
        "three_dimensional_numeric": "xyz_points",
        "time_series": "datetime_series",
        "hierarchical": "named_points",
        "specialized": "xy_pairs"
    }

    schema["binding"] = shape_binding_map.get(shape, "series_y")
    return schema



def _complete_aggregation(schema: dict) -> dict:
    """shape = schema.get("data_shape")
    aggregation = schema.get("aggregation")

    if shape in {"categorical_numeric", "time_series"} and aggregation is None:
        schema["aggregation"] = "sum"""

    return schema




def _validate_shape_mapping(shape: str, mapping: dict, aggregation: str | None = None):
    rules = {
        "one_dimensional_numeric": {"numeric_column"},
        "categorical_numeric": {"category_column", "numeric_column"},
        "two_dimensional_numeric": {"x_column", "y_column"},
        "time_series": {"datetime_column", "numeric_column"},
         "hierarchical": {"category_column", "numeric_column"},
        "specialized": {"x_column", "y_column", "numeric_column"}
    }

    
    if shape == "categorical_numeric":
        if aggregation == "count":
            required = {"category_column"}
        else:
            required = {"category_column", "numeric_column"}
    else:
        required = rules.get(shape)

    if not required:
        return

    missing = required - set(mapping.keys())
    if missing:
        raise ValueError(f"{shape} requires {required}")


def _validate_columns_exist(mapping: dict, profile: dict):
    all_columns = set(
        profile.get("numeric_columns", [])
        + profile.get("categorical_columns", [])
        + profile.get("datetime_columns", [])
    )

    for col in mapping.values():
        if col not in all_columns:
            raise ValueError(f"Column '{col}' not found in dataset")


def _validate_schema(schema: dict, profile: dict):
    required = {"highcharts_type", "data_shape", "data_mapping", "requirements", "operation"}
    missing = required - schema.keys()
    if missing:
        raise ValueError(f"Schema missing keys: {missing}")

    if schema["data_shape"] not in ALLOWED_SHAPES:
        raise ValueError(f"Invalid data_shape: {schema['data_shape']}")

    mapping = schema["data_mapping"]
    if not isinstance(mapping, dict):
        raise ValueError("data_mapping must be object")

    #_validate_shape_mapping(schema["data_shape"], mapping)
    _validate_shape_mapping(
    schema["data_shape"],
    mapping,
    schema.get("aggregation")
    )
    _validate_columns_exist(mapping, profile)



def plan_visualization(user_query, data_profile, previous_state=None):

    
    if previous_state is not None:
        patched = _apply_structural_patch(user_query, previous_state)
        if patched:
            patched = _complete_aggregation(patched)
            patched = _attach_binding(patched)
            return {"status": "ok", "schema": patched}

    
    prompt_file = "planner_new.txt" if previous_state is None else "planner_update.txt"
    prompt_path = os.path.join(PROMPT_DIR, prompt_file)

    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    if previous_state is None:
        user_prompt = f"""
DATASET PROFILE:
{json.dumps(data_profile, indent=2, default=str)}

USER REQUEST:
{user_query}

Return JSON only.
"""
    else:
        user_prompt = f"""
CURRENT VISUAL STATE:
{json.dumps(previous_state, indent=2, default=str)}

DATASET PROFILE:
{json.dumps(data_profile, indent=2, default=str)}

USER MODIFICATION REQUEST:
{user_query}

Return JSON only.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=900,
        timeout=30,
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Planner LLM returned empty response")

    raw = content.strip()
    parsed = _extract_json(raw)

    if parsed.get("status") == "error":
        return parsed

    schema = parsed.get("schema")
    if not isinstance(schema, dict):
        raise ValueError("Missing schema object")

    schema = _complete_aggregation(schema)
    _validate_schema(schema, data_profile)
    schema = _attach_binding(schema)

    return {"status": "ok", "schema": schema}