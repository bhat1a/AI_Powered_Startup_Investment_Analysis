import os
import json
import uuid
import traceback
import concurrent.futures
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from csv_visualizer.core.json_safe import to_json_safe
from csv_visualizer.core.data_cleaner import clean_dataframe
from csv_visualizer.core.profiler import profile_dataframe
from csv_visualizer.core.stat_adapters import compute_stats
from csv_visualizer.agents.graph import build_graph


BASE_DIR   = Path(__file__).parent
STATIC_DIR = BASE_DIR / "csv_visualizer" / "static" / "export"

HC_MODULES = [
    "highcharts.js","highcharts-more.js","highcharts-3d.js",
    "histogram-bellcurve.js","accessibility.js","solid-gauge.js",
    "treemap.js","heatmap.js","sankey.js","networkgraph.js",
    "venn.js","funnel.js","funnel3d.js","pyramid3d.js","gantt.js",
    "wordcloud.js","streamgraph.js","timeline.js","sunburst.js",
    "annotations.js","annotations-advanced.js","data.js","drilldown.js",
    "stock.js","map.js","datagrid.js","dashboards.js","layout.js",
    "exporting.js","export-data.js","offline-exporting.js",
]

_HC_THEME = {
    "colors":   ["#7C3AED", "#A855F7", "#10B981", "#F59E0B", "#3B82F6", "#C9A227"],
    "chart":    {"backgroundColor": "transparent",
                 "style": {"fontFamily": "Outfit, sans-serif"}},
    "title":    {"style": {"color": "#F0EEFF", "fontSize": "16px",
                           "fontWeight": "500", "fontFamily": "Outfit, sans-serif"}},
    "subtitle": {"style": {"color": "#A89BC2"}},
    "xAxis":    {"gridLineColor": "rgba(255,255,255,0.06)",
                 "labels": {"style": {"color": "#A89BC2", "fontSize": "12px"}},
                 "lineColor": "rgba(255,255,255,0.08)", "tickColor": "rgba(255,255,255,0.08)",
                 "title": {"style": {"color": "#A89BC2"}}},
    "yAxis":    {"gridLineColor": "rgba(255,255,255,0.06)",
                 "labels": {"style": {"color": "#A89BC2", "fontSize": "12px"}},
                 "lineColor": "rgba(255,255,255,0.08)", "tickColor": "rgba(255,255,255,0.08)",
                 "title": {"style": {"color": "#A89BC2"}}},
    "legend":   {"itemStyle": {"color": "#A89BC2", "fontWeight": "400"},
                 "itemHoverStyle": {"color": "#F0EEFF"}},
    "tooltip":  {"backgroundColor": "rgba(13,8,32,0.95)",
                 "borderColor": "rgba(124,58,237,0.40)",
                 "borderRadius": 8,
                 "style": {"color": "#F0EEFF", "fontSize": "13px"}},
    "credits":  {"enabled": False},
}


@st.cache_resource
def load_hc_bundle():
    parts, missing = [], []
    for fname in HC_MODULES:
        p = STATIC_DIR / fname
        if p.exists():
            parts.append(p.read_text(encoding="utf-8"))
        else:
            missing.append(fname)
    if missing:
        print("[HC] Missing modules:", missing)
    return "\n".join(parts)


@st.cache_resource
def get_graph():
    return build_graph()


def _resolve_binding(chart_type, data_shape):
    if chart_type in {"pie", "funnel", "pyramid"}:
        return "named_points"
    mapping = {
        "categorical_numeric":     "category_series",
        "time_series":             "datetime_series",
        "two_dimensional_numeric": "xy_pairs",
    }
    return mapping.get(data_shape, "series_y")


def _save_version(visual_id, version, schema, files):
    base = Path("generated") / visual_id / f"v{version}"
    base.mkdir(parents=True, exist_ok=True)
    (base / "visual_state.json").write_text(json.dumps(schema, indent=2))
    for name, content in files.items():
        (base / name).write_text(content, encoding="utf-8")
    return str(base)


def _render_chart(files):
    hc_bundle  = load_hc_bundle()
    css        = files.get("style.css", "")
    js         = files.get("script.js", "")
    theme_json = json.dumps(_HC_THEME)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
html, body {{ margin:0; padding:0; background:transparent; }}
#container {{
    width:100%; height:560px;
    background:rgba(13,8,32,0.60);
    backdrop-filter:blur(20px);
    -webkit-backdrop-filter:blur(20px);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:20px;
}}
{css}
</style>
</head>
<body>
<div id="container"></div>
<script>{hc_bundle}</script>
<script>Highcharts.setOptions({theme_json});</script>
<script>{js}</script>
</body>
</html>"""

    components.html(html, height=600, scrolling=False)


def _init_state():
    defaults = {
        "visual_state": None, "version": 0, "visual_id": None,
        "last_data": None, "binding": None, "last_files": None,
        "patch_llm_debug": None, "last_tool": None, "save_path": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_visualizer():
    _init_state()
    graph = get_graph()

    elevated   = "rgba(255,255,255,0.06)"
    border     = "rgba(255,255,255,0.10)"
    accent     = "#7C3AED"
    text_sec   = "#A89BC2"
    text_muted = "#5A4E72"

    col_upload, col_query = st.columns([1, 1])
    with col_upload:
        st.markdown(
            "<p style='font-family:Outfit,sans-serif;font-size:13px;"
            "color:#A89BC2;font-weight:400;margin:0 0 6px;'>CSV File</p>",
            unsafe_allow_html=True,
        )
        file = st.file_uploader("", type=["csv"], label_visibility="collapsed")
    with col_query:
        st.markdown(
            "<p style='font-family:Outfit,sans-serif;font-size:13px;"
            "color:#A89BC2;font-weight:400;margin:0 0 6px;'>Describe your chart</p>",
            unsafe_allow_html=True,
        )
        query = st.text_area(
            "",
            placeholder='e.g. Show me revenue by sector as a bar chart',
            height=100,
            label_visibility="collapsed",
        )

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    generate = st.button("Generate Chart", type="primary", use_container_width=True)
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    if generate and file and query:
        with st.spinner("Running pipeline..."):
            try:
                df      = clean_dataframe(pd.read_csv(file))
                profile = profile_dataframe(df)

                state = {
                    "query":   query,
                    "df":      df,
                    "profile": profile,
                    "schema":  st.session_state.visual_state,
                    "files":   st.session_state.last_files,
                }
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(graph.invoke, state)
                    try:
                        result = future.result(timeout=90)
                    except concurrent.futures.TimeoutError:
                        st.error("The chart generation timed out — the AI service may be under heavy load. Please try again in a moment.")
                        st.stop()

                if result.get("stop"):
                    tool_val  = result.get("tool", "")
                    error_msg = tool_val.split("::", 1)[1] if "::" in tool_val else "Unknown planner error"
                    if st.session_state.last_files:
                        _render_chart(st.session_state.last_files)
                    st.error(error_msg)
                    st.stop()

                schema = result.get("schema")
                files  = result.get("files")
                patch  = result.get("patch")
                tool   = result.get("tool")

                st.session_state.last_tool = tool
                if schema:
                    st.session_state.visual_state = schema

                if tool == "planner_tool" and schema:
                    binding = _resolve_binding(schema["highcharts_type"], schema["data_shape"])
                    st.session_state.binding = binding
                    DATA = compute_stats(
                        schema["highcharts_type"], df,
                        schema["data_mapping"], schema.get("aggregation"), binding,
                    )
                    st.session_state.last_data = DATA
                    if files and "script.js" in files and "const DATA" not in files["script.js"]:
                        files["script.js"] = (
                            "const DATA = " + json.dumps(to_json_safe(DATA)) + ";\n\n"
                            + files["script.js"]
                        )

                if tool == "patch_tool":
                    DATA = st.session_state.last_data
                    if DATA and files and "script.js" in files and "const DATA" not in files["script.js"]:
                        files["script.js"] = (
                            "const DATA = " + json.dumps(to_json_safe(DATA)) + ";\n\n"
                            + files["script.js"]
                        )

                if files:
                    st.session_state.last_files = files
                if patch:
                    st.session_state.patch_llm_debug = patch
                if st.session_state.visual_id is None:
                    st.session_state.visual_id = str(uuid.uuid4())

                st.session_state.version += 1
                save_path = _save_version(
                    st.session_state.visual_id,
                    st.session_state.version,
                    st.session_state.visual_state,
                    st.session_state.last_files,
                )
                st.session_state.save_path = save_path

            except Exception:
                st.error("Pipeline error")
                st.code(traceback.format_exc())
                st.stop()

    if st.session_state.last_files:
        _render_chart(st.session_state.last_files)

        pass

    elif generate and (not file or not query):
        st.info("Please upload a CSV file and enter a chart description.")
