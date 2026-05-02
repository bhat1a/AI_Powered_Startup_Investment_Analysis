import json
from langchain.tools import tool

from csv_visualizer.core.planner_llm import plan_visualization
from csv_visualizer.core.codegen_llm import generate_code
from csv_visualizer.core.patch_llm import generate_patch, apply_patch


# ─────────────────────────────────────
# PLANNER TOOL
# ─────────────────────────────────────

@tool
def planner_tool(query: str, profile: dict, previous_schema: dict | None = None):
    """
    Generates a visualization schema from a natural language query
    and dataset profile.

    Use this tool when the user asks to:
    - create a new chart
    - change chart type
    - change aggregation or columns
    """

    print("\n================ PLANNER TOOL ================")
    print(query)
    print("==============================================")

    plan = plan_visualization(
        query,
        profile,
        previous_state=previous_schema
    )

    if plan.get("status") != "ok":
        raise ValueError(plan.get("reason", "Planner failed"))

    schema = plan["schema"]

    return {
        "schema": schema
    }


# ─────────────────────────────────────
# CODEGEN TOOL
# ─────────────────────────────────────

@tool
def codegen_tool(schema: dict):
    """
    Generates Highcharts code files from a visualization schema.

    Use this tool after planner_tool generates a schema.
    """

    print("\n================ CODEGEN TOOL ================")
    print(json.dumps(schema, indent=2))
    print("==============================================")

    code = generate_code(json.dumps({
        "visual_state": schema,
        "chart_type": schema.get("highcharts_type"),
        "data_shape": schema.get("data_shape")
    }))

    return {
        "code": code
    }


# ─────────────────────────────────────
# PATCH TOOL
# ─────────────────────────────────────

@tool
def patch_tool(query: str, files: dict, schema: dict | None = None):
    """
    Applies edits to an existing chart using patch diffs.

    Use this tool when the user asks to:
    - change colors
    - move legend
    - modify labels
    - add reference lines
    - update styling
    """

    if not files:
        print("[PATCH TOOL] No files available")
        return {"files": files, "patch": "NO_CHANGES"}

    print("\n================ PATCH TOOL ==================")
    print(query)
    print("==============================================")

    patch = generate_patch(
        files=files,
        instruction=query,
        schema=schema
    )

    if patch != "NO_CHANGES":
        files = apply_patch(patch, files)

    return {
        "files": files,
        "patch": patch
    }