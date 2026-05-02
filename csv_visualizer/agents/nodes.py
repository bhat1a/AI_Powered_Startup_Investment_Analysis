import logging
import json

from csv_visualizer.core.router_llm import route_query
from csv_visualizer.core.planner_llm import plan_visualization
from csv_visualizer.core.codegen_llm import generate_code
from csv_visualizer.core.codegen_validator import validate_codegen_output
from csv_visualizer.core.parser import parse_llm_output
from csv_visualizer.core.patch_llm import generate_patch, apply_patch

logger = logging.getLogger(__name__)


def router_node(state):
    logger.debug(f"ROUTER — query: {state['query']}")
    try:
        result = route_query(state)
        state["tool"] = result["tool"]
    except Exception as e:
        logger.error(f"Router failed: {e}")
        state["stop"] = True
        state["tool"] = "router_error::Could not process your request. Please try again."
    return state


def planner_node(state):
    query = state["query"]
    profile = state["profile"]

    logger.debug(f"PLANNER — query: {query}")

    plan = plan_visualization(
        query,
        profile,
        previous_state=state.get("schema")
    )

    logger.debug(f"PLANNER output: {json.dumps(plan, indent=2)}")

    if plan.get("status") != "ok":
        reason = plan.get("reason", "Planner failed")
        logger.error(f"PLANNER ERROR: {reason}")
        state["error"] = reason
        state["stop"] = True
        state["tool"] = f"planner_error::{reason}"
        return state

    state["schema"] = plan["schema"]
    logger.debug(f"PLANNER schema: {json.dumps(state['schema'], indent=2)}")

    return state


def codegen_node(state):
    if state.get("stop"):
        logger.debug("CODEGEN skipping — stop flag set")
        return state

    schema = state["schema"]
    logger.debug(f"CODEGEN — schema: {json.dumps(schema, indent=2)}")

    codegen_input = json.dumps({
        "visual_state": schema,
        "chart_type": schema.get("highcharts_type"),
        "data_shape": schema.get("data_shape")
    })

    code = generate_code(codegen_input)
    logger.debug(f"CODEGEN output length: {len(code)} chars")

    validate_codegen_output(code)

    files = parse_llm_output(code)
    logger.debug(f"CODEGEN files: {list(files.keys())}")

    state["files"] = files
    return state


def patch_node(state):
    files = state.get("files")
    query = state["query"]
    df = state.get("df")
    schema = state.get("schema")

    logger.debug("PATCH NODE")

    if not files:
        logger.debug("PATCH — no files, skipping")
        return state

    logger.debug(f"PATCH — query: {query}")

    q = query.lower()
    stat_value = None
    stat_type = None

    if any(k in q for k in ["average", "mean", "avg"]):
        stat_type = "mean"
    elif any(k in q for k in ["median", "50th"]):
        stat_type = "median"
    elif any(k in q for k in ["max", "maximum", "highest"]):
        stat_type = "max"
    elif any(k in q for k in ["min", "minimum", "lowest"]):
        stat_type = "min"

    if stat_type and df is not None and schema:
        mapping = schema.get("data_mapping", {})
        num_col = mapping.get("numeric_column") or mapping.get("y_column")

        if num_col and num_col in df.columns:
            series = df[num_col].dropna().astype(float)

            if stat_type == "mean":
                stat_value = float(series.mean())
            elif stat_type == "median":
                stat_value = float(series.median())
            elif stat_type == "max":
                stat_value = float(series.max())
            elif stat_type == "min":
                stat_value = float(series.min())

            logger.debug(f"PATCH computed {stat_type}: {stat_value:.4f}")

            query = f"""
USER REQUEST:
{query}

Computed statistic:
{stat_type} = {stat_value:.4f}

Use this numeric value when generating the patch.
"""

    try:
        patch = generate_patch(
            files=files,
            instruction=query,
            schema=schema
        )
    except Exception as e:
        logger.error(f"PATCH LLM failed: {e}")
        patch = "NO_CHANGES"

    logger.debug(f"PATCH output: {patch}")

    if patch != "NO_CHANGES":
        try:
            logger.debug("PATCH applying...")
            files = apply_patch(patch, files)
        except Exception as e:
            logger.error(f"PATCH apply failed: {e}")
            patch = "NO_CHANGES"
    else:
        logger.debug("PATCH — no changes required")

    state["patch"] = patch
    state["files"] = files
    return state
