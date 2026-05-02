import os
import re
import json
import requests

CACHE_DIR = "api_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# bar→column, pyramid→funnel, spline→line (no own properties in tree)
CHART_API_SECTIONS = {
    "boxplot":   ["plotOptions.boxplot"],
    "line":      ["plotOptions.line"],
    "column":    ["plotOptions.column"],
    "bar":       ["plotOptions.column"],
    "pie":       ["plotOptions.pie"],
    "funnel":    ["plotOptions.funnel"],
    "pyramid":   ["plotOptions.funnel"],
    "scatter":   ["plotOptions.scatter"],
    "area":      ["plotOptions.area"],
    "heatmap":   ["plotOptions.heatmap"],
    "treemap":   ["plotOptions.treemap"],
    "histogram": ["plotOptions.histogram"],
    "bellcurve": ["plotOptions.bellcurve"],
    "bubble":    ["plotOptions.bubble"],
    "spline":    ["plotOptions.line"],
}

CHART_INHERITANCE = {
    "bar":     "bar shares all plotOptions with column — displayed horizontally.",
    "pyramid": "pyramid shares all plotOptions with funnel — displayed inverted.",
    "spline":  "spline shares all plotOptions with line — rendered with curved lines.",
}

CACHE_FALLBACK = {
    "bar":     "column",
    "pyramid": "funnel",
    "spline":  "line",
}

_api_tree = None


def _load_tree() -> dict:
    global _api_tree
    if _api_tree is not None:
        return _api_tree
    try:
        response = requests.get(
            "https://api.highcharts.com/highcharts/tree.json",
            timeout=15
        )
        response.raise_for_status()
        _api_tree = response.json()
        print("[API] Highcharts tree.json loaded successfully")
        return _api_tree
    except Exception as e:
        print(f"[WARN] Failed to fetch Highcharts tree.json: {e}")
        _api_tree = {}
        return {}


def _navigate(api: dict, path: str) -> dict:
    """
    Navigate tree by dot-separated path.
    Tries direct key access and children-wrapped access at each level.
    """
    parts = path.split(".")
    node  = api

    for part in parts:
        found = False

        if isinstance(node, dict) and part in node:
            node  = node[part]
            found = True
        elif isinstance(node, dict) and "children" in node:
            children = node["children"]
            if isinstance(children, dict) and part in children:
                node  = children[part]
                found = True

        if not found:
            return {}

    return node


def _build_js_skeleton(node: dict, indent: int = 0, max_depth: int = 6) -> str:
    """
    Recursively builds a JavaScript object skeleton from an API node.

    This produces output like a real Highcharts fiddle:

        dataLabels: {
            align (string) [default: center]  // Alignment of the data label
            enabled (boolean) [default: false]  // Enable or disable the data labels
            format (string) [default: {y}]  // A format string for the data label
        }

    Every property at every depth level is included.
    Properties with sub-children get a nested block.
    Leaf properties get a single commented line.
    """
    if indent > max_depth:
        return ""

    lines    = []
    pad      = "    " * indent

    # Resolve children
    if "children" in node and isinstance(node["children"], dict):
        children = node["children"]
    elif isinstance(node, dict):
        meta_keys = {
            "description", "returnType", "defaults", "default",
            "extends", "inherits", "since", "deprecated",
            "products", "type", "title", "isParent",
            "children", "demo", "seeAlso", "samples"
        }
        children = {
            k: v for k, v in node.items()
            if k not in meta_keys and isinstance(v, dict)
        }
    else:
        return ""

    if not children:
        return ""

    for prop, data in sorted(children.items()):
        if not isinstance(data, dict):
            continue

        prop_type   = data.get("returnType", "")
        default     = data.get("defaults", data.get("default", ""))
        description = data.get("description", "")
        description = re.sub(r"<[^>]+>", "", str(description)).strip()
        description = description[:100] if description else ""

        has_children = bool(
            data.get("children") and
            isinstance(data.get("children"), dict) and
            len(data["children"]) > 0
        )

        if has_children:
            # Nested object property — recurse
            nested = _build_js_skeleton(data, indent + 1, max_depth)
            if nested:
                meta = ""
                if prop_type:
                    meta += f" ({prop_type})"
                if default not in ("", None, "undefined", "null"):
                    meta += f" [default: {default}]"
                if description:
                    meta += f"  // {description}"

                lines.append(f"{pad}{prop}:{meta} {{")
                lines.append(nested)
                lines.append(f"{pad}}}")
            else:
                # Has children key but they're empty — treat as leaf
                meta = _format_meta(prop_type, default, description)
                lines.append(f"{pad}{prop}{meta}")
        else:
            # Leaf property
            meta = _format_meta(prop_type, default, description)
            lines.append(f"{pad}{prop}{meta}")

    return "\n".join(lines)


def _format_meta(prop_type: str, default: str, description: str) -> str:
    """Format the type/default/description annotation for a property line."""
    parts = []
    if prop_type:
        parts.append(f"({prop_type})")
    if default not in ("", None, "undefined", "null"):
        parts.append(f"[default: {default}]")
    if description:
        parts.append(f"// {description}")

    if parts:
        return " " + " ".join(parts)
    return ""


def _build_api_context(chart_type: str) -> str:
    """
    Builds the full API context string for a chart type.
    Output is a deeply nested JS skeleton showing every available property.
    """
    api      = _load_tree()
    sections = CHART_API_SECTIONS.get(chart_type, [])

    if not sections or not api:
        return ""

    output = [
        f"HIGHCHARTS API REFERENCE — {chart_type.upper()}",
        "=" * 60,
        "",
        "The following lists EVERY available property for this chart type.",
        "Nested blocks show object properties and their sub-properties.",
        "You MUST explicitly define every property shown below.",
        "",
    ]

    if chart_type in CHART_INHERITANCE:
        output.append(f"NOTE: {CHART_INHERITANCE[chart_type]}")
        output.append("")

    found_any = False

    for section_path in sections:
        node = _navigate(api, section_path)

        if not node:
            print(f"[WARN] API path not found: {section_path}")
            continue

        display_path = f"plotOptions.{chart_type}"
        if section_path != display_path:
            output.append(f"{display_path} (inherited from {section_path}) {{")
        else:
            output.append(f"{section_path} {{")

        skeleton = _build_js_skeleton(node, indent=1, max_depth=6)

        if skeleton:
            output.append(skeleton)
            output.append("}")
            found_any = True
        else:
            output.pop()  # remove the opening line
            print(f"[WARN] No skeleton generated for: {section_path}")

    return "\n".join(output) if found_any else ""


def get_chart_api_context(chart_type: str) -> str:
    """
    Returns full Highcharts API context for a chart type.
    Fetches once, caches to disk, returns from cache on subsequent calls.
    Falls back to parent type for inherited charts (bar, pyramid, spline).
    """
    if not chart_type:
        return ""

    cache_file = os.path.join(CACHE_DIR, f"{chart_type}.txt")

    # Return disk cache if valid
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = f.read()
        if cached.strip():
            print(f"[API] Loaded cached context for '{chart_type}'")
            return cached
        else:
            os.remove(cache_file)

    # Build fresh
    print(f"[API] Building API context for '{chart_type}'...")
    context = _build_api_context(chart_type)

    if context:
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(context)
        print(f"[API] Cached '{chart_type}' → {cache_file} ({len(context)} chars)")
        return context

    # Fallback: copy from parent cache for inherited types
    parent = CACHE_FALLBACK.get(chart_type)
    if parent:
        print(f"[API] Falling back to parent '{parent}' for '{chart_type}'...")

        # Ensure parent is cached
        parent_context = get_chart_api_context(parent)

        if parent_context:
            note      = CHART_INHERITANCE.get(chart_type, "")
            # Replace header
            lines     = parent_context.splitlines()
            new_header = [
                f"HIGHCHARTS API REFERENCE — {chart_type.upper()}",
                "=" * 60,
                "",
                "The following lists EVERY available property for this chart type.",
                "Nested blocks show object properties and their sub-properties.",
                "You MUST explicitly define every property shown below.",
                "",
                f"NOTE: {note}",
                "",
            ]
            # Skip old header lines (up to first blank line after =====)
            skip = 0
            for i, line in enumerate(lines):
                if line.startswith("plotOptions") or line.startswith("NOTE"):
                    skip = i
                    break

            child_context = "\n".join(new_header + lines[skip:])

            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(child_context)

            print(f"[API] Cached '{chart_type}' from parent '{parent}' ({len(child_context)} chars)")
            return child_context

    print(f"[WARN] No API context retrieved for '{chart_type}'")
    return ""


def debug_tree_structure(max_depth: int = 3):
    """Print the raw tree structure for debugging."""
    api = _load_tree()
    if not api:
        print("[DEBUG] Tree is empty or failed to load")
        return

    def _print_tree(node, depth=0, name="root"):
        if depth > max_depth:
            return
        indent = "  " * depth
        if isinstance(node, dict):
            print(f"{indent}{name}/  ({len(node)} keys)")
            for k, v in list(node.items())[:8]:
                _print_tree(v, depth + 1, k)
        elif isinstance(node, list):
            print(f"{indent}{name}[]  ({len(node)} items)")
        else:
            print(f"{indent}{name}: {str(node)[:60]}")

    print("\n[DEBUG] Highcharts tree.json structure:")
    print("=" * 60)
    _print_tree(api, max_depth=max_depth)
    print("=" * 60 + "\n")