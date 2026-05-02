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

def generate_code(schema: str) -> str:

    chart_type = ""

    try:
        parsed = json.loads(schema)
        chart_type = parsed.get("chart_type", "")
    except Exception:
        pass

    # ── Load base system prompt
    base_prompt = open("csv_visualizer/prompts/codegen.txt", encoding="utf-8").read()

    # ── Inject Highcharts API context
    api_context = get_chart_api_context(chart_type) if chart_type else ""

    if api_context:

        system_prompt = base_prompt + f"""

────────────────────────────────────────
HIGHCHARTS API REFERENCE (AUTHORITATIVE)
────────────────────────────────────────

The following is the OFFICIAL Highcharts API for '{chart_type}'.

Use ONLY these property names.
Do NOT invent property names not listed here.
Do NOT use properties from other chart types.

{api_context}

ENFORCEMENT:
If a property you want to use is NOT in this reference:
→ Do NOT use it.
→ Using invented properties = INVALID OUTPUT.
"""

    else:

        system_prompt = base_prompt
        print(f"[WARN] No API context available for chart type: '{chart_type}'")

    # ── Inject binding-specific template so llama doesn't hallucinate data access
    try:
        parsed_schema = json.loads(schema)
        binding = parsed_schema.get("binding", "")
        chart = parsed_schema.get("highcharts_type", "")
    except Exception:
        binding = ""
        chart = ""

    if binding == "category_series":
        binding_hint = """
CRITICAL — binding is category_series. You MUST use EXACTLY this pattern, no exceptions:

xAxis: { categories: DATA.categories }
series: [{ name: '<title>', data: DATA.values }]

DO NOT use numeric indices. DO NOT use DATA[0], DATA[1]. DO NOT hardcode values.
xAxis.categories MUST be DATA.categories. series[0].data MUST be DATA.values.
"""
    elif binding == "named_points":
        binding_hint = """
CRITICAL — binding is named_points. You MUST use EXACTLY this pattern:

series: [{ name: '<title>', data: DATA }]

DATA is already an array of {name, y} objects. Do not transform it.
"""
    elif binding == "datetime_series":
        binding_hint = """
CRITICAL — binding is datetime_series. You MUST use EXACTLY this pattern:

xAxis: { type: 'datetime' }
series: [{ name: '<title>', data: DATA }]

DATA is already [[timestamp_ms, value], ...]. Do not transform it.
"""
    else:
        binding_hint = ""

    # ── User prompt
    user_prompt = f"""
Compile visualization from schema.
{binding_hint}
TITLE RULE: If schema contains a "title" field, use that exact string as the Highcharts title.text value.

RETURN FILES ONLY.

Schema:
{schema}
"""

    # ── Single LLM call (LangGraph handles retry)
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