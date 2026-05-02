import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
from csv_visualizer.core.codegen_validator import validate_codegen_output
from csv_visualizer.core.highcharts_api import get_chart_api_context

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.cerebras.ai/v1"
)

MODEL = os.getenv("INTENT_MODEL", "llama3.1-8b")


# ─────────────────────────────────────
# SANITIZER
# ─────────────────────────────────────

def _sanitize(text: str) -> str:

    if not text:
        return ""

    # Remove markdown code blocks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Remove thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    return text.strip()


# ─────────────────────────────────────
# CODE GENERATION
# ─────────────────────────────────────

_HC_WRAPPER = """function waitForHighcharts() {{
  if (typeof Highcharts !== 'undefined') {{ renderChart(); }}
  else {{ setTimeout(waitForHighcharts, 50); }}
}}
function renderChart() {{
  Highcharts.chart('container', {config});
}}
waitForHighcharts();"""

_BASE_CSS = """body { margin: 0; padding: 0; font-family: system-ui, sans-serif; }
#container { width: 100%; height: 560px; }"""


def _template_category_series(s):
    mapping  = s.get("data_mapping", {})
    x_label  = mapping.get("category_column", "")
    y_label  = mapping.get("numeric_column", mapping.get("y_column", ""))
    title    = s.get("title", "")
    ctype    = s.get("highcharts_type", "bar")
    config = f"""{{
  chart: {{ type: '{ctype}' }},
  title: {{ text: '{title}' }},
  xAxis: {{ categories: DATA.categories, title: {{ text: '{x_label}' }} }},
  yAxis: {{ title: {{ text: '{y_label}' }} }},
  legend: {{ enabled: false }},
  tooltip: {{ pointFormat: '{{series.name}}: <b>{{point.y:,.1f}}</b>' }},
  plotOptions: {{ {ctype}: {{ dataLabels: {{ enabled: true }} }} }},
  colors: ['#7C3AED'],
  series: [{{ name: '{title}', data: DATA.values }}]
}}"""
    return config


def _template_named_points(s):
    title = s.get("title", "")
    ctype = s.get("highcharts_type", "pie")
    config = f"""{{
  chart: {{ type: '{ctype}' }},
  title: {{ text: '{title}' }},
  xAxis: {{}},
  yAxis: {{}},
  legend: {{ enabled: true }},
  tooltip: {{ pointFormat: '{{series.name}}: <b>{{point.percentage:.1f}}%</b>' }},
  plotOptions: {{ {ctype}: {{ allowPointSelect: true, cursor: 'pointer', dataLabels: {{ enabled: true, format: '<b>{{point.name}}</b>: {{point.percentage:.1f}}%' }} }} }},
  colors: ['#7C3AED','#A855F7','#10B981','#F59E0B','#3B82F6','#EF4444','#C9A227','#06B6D4'],
  series: [{{ name: '{title}', data: DATA }}]
}}"""
    return config


def _template_datetime_series(s):
    mapping = s.get("data_mapping", {})
    y_label = mapping.get("numeric_column", mapping.get("y_column", ""))
    title   = s.get("title", "")
    ctype   = s.get("highcharts_type", "line")
    config = f"""{{
  chart: {{ type: '{ctype}' }},
  title: {{ text: '{title}' }},
  xAxis: {{ type: 'datetime', title: {{ text: '' }} }},
  yAxis: {{ title: {{ text: '{y_label}' }} }},
  legend: {{ enabled: false }},
  tooltip: {{ xDateFormat: '%Y-%m-%d', pointFormat: '{{series.name}}: <b>{{point.y:,.1f}}</b>' }},
  plotOptions: {{ {ctype}: {{ marker: {{ enabled: true }} }} }},
  colors: ['#7C3AED'],
  series: [{{ name: '{title}', data: DATA }}]
}}"""
    return config


def _template_xy_pairs(s):
    mapping = s.get("data_mapping", {})
    x_label = mapping.get("x_column", "")
    y_label = mapping.get("y_column", "")
    title   = s.get("title", "")
    config = f"""{{
  chart: {{ type: 'scatter' }},
  title: {{ text: '{title}' }},
  xAxis: {{ title: {{ text: '{x_label}' }} }},
  yAxis: {{ title: {{ text: '{y_label}' }} }},
  legend: {{ enabled: false }},
  tooltip: {{ pointFormat: '{x_label}: <b>{{point.x}}</b><br/>{y_label}: <b>{{point.y}}</b>' }},
  plotOptions: {{ scatter: {{ marker: {{ radius: 4 }} }} }},
  colors: ['#7C3AED'],
  series: [{{ name: '{title}', data: DATA }}]
}}"""
    return config


_BINDING_TEMPLATES = {
    "category_series": _template_category_series,
    "named_points":    _template_named_points,
    "datetime_series": _template_datetime_series,
    "xy_pairs":        _template_xy_pairs,
}


def _build_from_template(parsed_schema) -> str:
    binding = parsed_schema.get("binding", "")
    fn = _BINDING_TEMPLATES.get(binding)
    if fn is None:
        return ""
    config  = fn(parsed_schema)
    script  = _HC_WRAPPER.format(config=config)
    return f"--- style.css ---\n{_BASE_CSS}\n\n--- script.js ---\n{script}"


def generate_code(schema: str) -> str:

    try:
        parsed_schema = json.loads(schema)
    except Exception:
        parsed_schema = {}

    # codegen_node wraps schema under "visual_state" — unwrap it
    if "visual_state" in parsed_schema and isinstance(parsed_schema["visual_state"], dict):
        parsed_schema = parsed_schema["visual_state"]

    # ── Use deterministic Python template for all common bindings
    template_output = _build_from_template(parsed_schema)
    if template_output:
        return template_output

    # ── Fall back to LLM only for exotic types (histogram, heatmap, etc.)
    chart_type = parsed_schema.get("chart_type", "")
    base_prompt = open("csv_visualizer/prompts/codegen.txt", encoding="utf-8").read()
    api_context = get_chart_api_context(chart_type) if chart_type else ""
    system_prompt = base_prompt + (f"\n\nHighcharts API for '{chart_type}':\n{api_context}" if api_context else "")

    user_prompt = f"""Compile visualization from schema. Use DATA variable for all data. Return files only.

Schema:
{schema}
"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=6000,
        timeout=30,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    raw = response.choices[0].message.content
    cleaned = _sanitize(raw)

    # ── Token usage log
    usage = response.usage
    if usage:
        print(
            "\n[CODEGEN] Token usage:"
            f"\n  prompt_tokens     : {usage.prompt_tokens}"
            f"\n  completion_tokens : {usage.completion_tokens}"
            f"\n  total_tokens      : {usage.total_tokens}\n"
        )

    validate_codegen_output(cleaned)

    return cleaned