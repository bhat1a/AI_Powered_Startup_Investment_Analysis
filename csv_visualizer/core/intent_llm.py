import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# OpenAI-compatible client
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.cerebras.ai/v1"
)

MODEL = os.getenv("INTENT_MODEL", "llama3.1-8b")

VALID_LABELS = [
    "NEW_VISUAL",
    "STRUCTURAL_UPDATE",
    "STYLE_UPDATE",
    "DATA_UPDATE"
]


# ══════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
You are an expert conversational visualization intent classifier.

You are given:
1) The CURRENT chart state.
2) A USER QUERY.

Your task is to classify the USER QUERY into EXACTLY ONE of:

NEW_VISUAL
STRUCTURAL_UPDATE
STYLE_UPDATE
DATA_UPDATE

Return ONLY the label.
No explanation.
No punctuation.
No extra text.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEFINITION OF EACH LABEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NEW_VISUAL
The user is requesting a brand new visualization specification.
They are describing what to build.
They are NOT referring to the existing chart.

STRUCTURAL_UPDATE
The user is modifying the existing chart’s type or axis mapping.
They are referring to the current chart and changing its structure.

STYLE_UPDATE
The user is modifying only visual appearance.
No chart type change. No data change.

DATA_UPDATE
The user is modifying filters, subsets, or aggregation.
Chart type stays the same.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL DISTINCTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If the query describes what to build without referencing
the existing chart → NEW_VISUAL.

If the query refers to the existing chart using words like:
“it”, “this”, “the chart”, “the graph”
→ It is NOT NEW_VISUAL.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTRAST EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USER QUERY: create a boxplot of petal length grouped by species
LABEL: NEW_VISUAL

USER QUERY: build a scatter plot of sepal length vs petal length
LABEL: NEW_VISUAL

USER QUERY: generate a histogram of sepal width
LABEL: NEW_VISUAL

USER QUERY: make a pie chart of species count
LABEL: NEW_VISUAL

USER QUERY: change it to a boxplot
LABEL: STRUCTURAL_UPDATE

USER QUERY: convert this to a scatter chart
LABEL: STRUCTURAL_UPDATE

USER QUERY: switch the chart to a histogram
LABEL: STRUCTURAL_UPDATE

USER QUERY: make it a pie chart
LABEL: STRUCTURAL_UPDATE

USER QUERY: use petal width on the x-axis instead
LABEL: STRUCTURAL_UPDATE

USER QUERY: change the bar color to orange
LABEL: STYLE_UPDATE

USER QUERY: enable data labels
LABEL: STYLE_UPDATE

USER QUERY: rotate the x-axis labels by 45 degrees
LABEL: STYLE_UPDATE

USER QUERY: move the legend to the bottom
LABEL: STYLE_UPDATE

USER QUERY: only show setosa
LABEL: DATA_UPDATE

USER QUERY: filter where sepal length > 5
LABEL: DATA_UPDATE

USER QUERY: show median instead of average
LABEL: DATA_UPDATE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Now classify the next query.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# ══════════════════════════════════════════════════════════════
# PROFILE HELPERS
# ══════════════════════════════════════════════════════════════

def _extract_columns(profile):

    raw = profile.get("columns", [])

    if isinstance(raw, dict):
        return list(raw.keys())

    if isinstance(raw, list):

        cols = []

        for item in raw:

            if isinstance(item, str):
                cols.append(item)

            elif isinstance(item, dict):

                name = (
                    item.get("name")
                    or item.get("column")
                    or item.get("col")
                    or str(next(iter(item.values()), "unknown"))
                )

                cols.append(str(name))

        return cols

    return []


# ══════════════════════════════════════════════════════════════
# USER PROMPT
# ══════════════════════════════════════════════════════════════

def _build_user_prompt(query, current_schema, profile):

    chart_type = current_schema.get("highcharts_type", "unknown")
    mapping = current_schema.get("data_mapping", {})

    columns = _extract_columns(profile)

    x_col = mapping.get("category_column", "unknown")
    y_col = mapping.get("numeric_column", "unknown")

    return f"""
CURRENT CHART
Type: {chart_type}
X-axis: {x_col}
Y-axis: {y_col}

DATASET COLUMNS
{", ".join(columns) if columns else "unknown"}

USER QUERY
{query}

LABEL:
"""


# ══════════════════════════════════════════════════════════════
# LLM CLASSIFICATION
# ══════════════════════════════════════════════════════════════

def _llm_classify(query, current_schema, profile):

    prompt = _build_user_prompt(query, current_schema, profile)

    retries = 3

    for attempt in range(retries):

        try:

            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                max_tokens=3,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )

            msg = response.choices[0].message.content

            if not msg:
                print("[INTENT] Empty response from LLM")
                continue

            raw = msg.strip().upper()

            print(f"[INTENT] LLM raw output: {repr(raw)}")

            if raw in VALID_LABELS:
                return raw

            print("[INTENT] Invalid label returned")
            continue

        except Exception as e:

            print(f"[INTENT] LLM error: {e}")

            # exponential backoff
            time.sleep(2 ** attempt)

    print("[INTENT] Failed after retries → fallback STYLE_UPDATE")

    # fallback if LLM repeatedly fails
    return "STYLE_UPDATE"


# ══════════════════════════════════════════════════════════════
# MAIN CLASSIFIER
# ══════════════════════════════════════════════════════════════

def classify_intent(
    query,
    has_active_visual,
    profile,
    current_schema=None
):

    if not has_active_visual:
        print("[INTENT] No active visual → NEW_VISUAL")
        return "NEW_VISUAL"

    if current_schema is None:
        current_schema = {}

    label = _llm_classify(query, current_schema, profile)

    print(f"[INTENT] Final: {label}")

    return label