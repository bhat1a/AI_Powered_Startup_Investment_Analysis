import re

BLOCKS = {
    "series": r"series\s*:\s*\[(.*?)\]",
    "title": r"title\s*:\s*\{(.*?)\}",
    "xAxis": r"xAxis\s*:\s*\{(.*?)\}",
    "yAxis": r"yAxis\s*:\s*\{(.*?)\}",
    "tooltip": r"tooltip\s*:\s*\{(.*?)\}",
    "plotOptions": r"plotOptions\s*:\s*\{(.*?)\}",
    "chart": r"chart\s*:\s*\{(.*?)\}"
}

KEYWORD_MAP = {
    "color": "series",
    "line": "chart",
    "bar": "chart",
    "column": "chart",
    "area": "chart",
    "title": "title",
    "tooltip": "tooltip",
    "axis": "xAxis",
    "grid": "xAxis",
    "marker": "plotOptions",
    "stack": "plotOptions"
}

def detect_target_block(query):
    q = query.lower()
    for key, block in KEYWORD_MAP.items():
        if key in q:
            return block
    return "series"

def extract_block(js, block_name):
    pattern = BLOCKS.get(block_name)
    if not pattern:
        return None
    match = re.search(pattern, js, re.DOTALL)
    return match.group(0) if match else None
