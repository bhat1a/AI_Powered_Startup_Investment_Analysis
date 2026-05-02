import re

KEYWORDS = {
    "color": ["color", "colors", "series", "data", "Highcharts.chart"],
    "title": ["title", "text:"],
    "chart_type": ["type:"],
    "axis": ["xAxis", "yAxis"],
    "tooltip": ["tooltip"],
    "legend": ["legend"],
    "height": ["height"],
}


def detect_intent_area(query: str):
    q = query.lower()

    if "color" in q:
        return "color"
    if "title" in q:
        return "title"
    if "line" in q or "bar" in q or "column" in q or "area" in q:
        return "chart_type"
    if "tooltip" in q:
        return "tooltip"
    if "height" in q or "width" in q:
        return "height"

    return "general"


def extract_relevant_snippet(files, query, context=40):

    area = detect_intent_area(query)
    keywords = KEYWORDS.get(area, [])

    snippets = {}

    for name, content in files.items():
        lines = content.split("\n")

        hits = []
        for i, line in enumerate(lines):
            if any(k in line for k in keywords):
                start = max(0, i - context)
                end = min(len(lines), i + context)
                hits.extend(lines[start:end])

        if hits:
            snippets[name] = "\n".join(hits)

    return snippets
