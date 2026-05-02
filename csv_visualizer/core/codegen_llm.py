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

    # ── User prompt
    user_prompt = f"""
Compile visualization from schema.

IMPORTANT DATA BINDING RULES:

If chart type is:
- pie, funnel, pyramid → series.data = DATA
- bar, column, line, area → categories = DATA.categories AND data = DATA.values
- scatter/bubble → series.data = DATA
- histogram/bellcurve → use derived baseSeries

Never access DATA.categories or DATA.values unless categorical_numeric.

TITLE RULE: If schema contains a "title" field, use that exact string as the Highcharts
title.text value. Never use the schema field names or data shape as the chart title.

RETURN FILES ONLY.

Schema:
{schema}
"""

    # ── Single LLM call (LangGraph handles retry)
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.1,
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