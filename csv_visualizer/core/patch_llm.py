import os
import re
from openai import OpenAI
from dotenv import load_dotenv
import json 
from csv_visualizer.core.logger import get_logger
logger = get_logger("patch_llm")
load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.cerebras.ai/v1"
)

MODEL = os.getenv("INTENT_MODEL", "llama3.1-8b")
SYSTEM_PROMPT = SYSTEM_PROMPT = """
You are a senior Highcharts developer. Edit script.js using V4A patches.

OUTPUT: Return only NO_CHANGES or:
*** Begin Patch
*** Update File: script.js
@@
 <parent context>
-<exact line from file>
+<replacement>
*** End Patch

CONDITIONAL COLORING — replace data: DATA.values with a map:
@@
         {
             name: 'Series Name',
-            data: DATA.values
+            data: DATA.values.map(function(v) {
+                return { y: v, color: v > 6 ? 'red' : '#7cb5ec' };
+            })
         }
Default color must be copied EXACTLY from the colors array in the file. Never use undefined or null.

ADDING PLOTLINES — anchor on yAxis title text AND its closing brace:
@@
     yAxis: {
         title: {
-            text: 'Y Axis Title'
-        }
+            text: 'Y Axis Title'
+        },
+        plotLines: [{ value: X, color: 'red', dashStyle: 'Solid', width: 2 }]
     },

PATCH RULES:
1. '-' line must be copied VERBATIM from the file (exact spaces, no line numbers).
2. Never anchor on }, ], }) — anchor on property lines only.
3. Include parent context lines when the same property exists in multiple places.
4. Re-add '-' line as '+' if it must stay, then append new lines.
5. Ensure valid JS — commas between properties, balanced braces.
6. Never modify: const DATA, Highcharts.chart('container',{, final });
7. Patch ONLY what was explicitly requested — never add changes the user did not ask for.
8. Line numbers [N] are shown for reference only — NEVER include them in '-' or '+' lines.
9. Reference lines → yAxis.plotLines for horizontal, xAxis.plotLines for vertical. Category/bar/column charts always use yAxis.plotLines.
10. All color values must be explicit strings copied from the file or specified by the user — never use undefined, null, or placeholder values.
11.line number are only included for the reffrence never add them in the patch 
TOOLTIP FORMATTER — detect from file before writing:
- xAxis has categories  → use this.point.category  (this.x returns index, wrong)
- chart type pie/funnel/pyramid → use this.point.name
- xAxis type datetime   → use Highcharts.dateFormat('%b %e', this.x)
- otherwise             → use this.x
formatter is always a JS function, never a string, never inside legend.

If Highcharts cannot do what is asked → NO_CHANGES.
Return patch immediately.
Do not include reasoning.
Do not include <think> blocks.
Keep response under 200 tokens.
"""
def clean_llm_output(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("```", "").strip()

    if re.search(r"\bNO_CHANGES\b", text):
        return "NO_CHANGES"

    if "*** Begin Patch" not in text:
        return ""
    text = text[text.index("*** Begin Patch"):]
    if "*** End Patch" in text:
        text = text[:text.index("*** End Patch") + len("*** End Patch")]
        return text.strip()

    raise ValueError(
        "Patch truncated: LLM hit token limit mid-output. "
        "Retrying may help if the reasoning is shorter next time."
    )


# ══════════════════════════════════════════════════════════════
# VALIDATE
# ══════════════════════════════════════════════════════════════

def validate_diff(patch_text: str) -> str:
    patch_text = patch_text.strip()

    if patch_text == "NO_CHANGES":
        return patch_text

    if not patch_text.startswith("*** Begin Patch"):
        raise ValueError("Invalid patch: must start with *** Begin Patch")
    if not patch_text.endswith("*** End Patch"):
        raise ValueError("Invalid patch: must end with *** End Patch")
    if "*** Update File:" not in patch_text:
        raise ValueError("Invalid patch: missing *** Update File block")
    if "@@" not in patch_text:
        raise ValueError("Invalid patch: missing @@ hunk marker")

    lines       = patch_text.splitlines()
    minus_lines = [l[1:] for l in lines if l.startswith("-") and not l.startswith("---")]
    plus_lines  = [l[1:] for l in lines if l.startswith("+") and not l.startswith("+++")]

    if not minus_lines and not plus_lines:
        raise ValueError("Invalid patch: no changes detected")
    if not plus_lines:
        raise ValueError("Invalid patch: no '+' lines found")
    if not minus_lines:
        raise ValueError(
            "Hunk has no '-' lines. "
            "Every hunk MUST have at least one '-' line. "
            "Find a nearby existing line, copy it EXACTLY as '-', "
            "re-add it as '+', then add your new lines as '+'."
        )

    
    if len(minus_lines) == len(plus_lines):
        if all(m.strip() == p.strip() for m, p in zip(minus_lines, plus_lines)):
            raise ValueError(
                "No-op patch: every '-' line is identical to its '+' line. "
                "The patch changes nothing. Modify the actual value requested."
            )

    return patch_text




def _parse_hunks(hunk_section: str) -> list:
    raw_hunks = re.split(r"@@[ \t]*\n", hunk_section)
    hunks     = []

    for raw in raw_hunks:
        if not raw.strip():
            continue

        context_before = []
        minus_lines    = []
        plus_lines     = []
        context_after  = []
        in_change      = False

        for line in raw.splitlines():
            if line.startswith("-"):
                minus_lines.append(line[1:])
                in_change = True
            elif line.startswith("+"):
                plus_lines.append(line[1:])
                in_change = True
            else:
                if in_change:
                    context_after.append(line)
                else:
                    context_before.append(line)

        if not minus_lines and not plus_lines:
            continue

        hunks.append({
            "context_before": context_before,
            "minus_lines":    minus_lines,
            "plus_lines":     plus_lines,
            "context_after":  context_after,
        })

    return hunks


# ══════════════════════════════════════════════════════════════
# EXTRACT DATA BLOCK
# ══════════════════════════════════════════════════════════════

def _extract_data_block(script: str):
    """
    Extracts full `const DATA = ...;` block (array or object format).
    Returns (data_block, stripped_script_with_placeholder).
    """
    match = re.search(r"(const\s+DATA\s*=\s*\[.*?\];)", script, re.DOTALL)
    if not match:
        match = re.search(r"(const\s+DATA\s*=\s*\{.*?\};)", script, re.DOTALL)
    if not match:
        return None, script

    data_block = match.group(1)
    stripped   = script.replace(data_block, "const DATA = __DATA__;", 1)
    return data_block, stripped
def _infer_data_shape_from_script(script: str) -> str:
    """
    Detects data structure from DATA declaration.
    """

    if "DATA.categories" in script and "DATA.values" in script:
        return "categorical_numeric"

    if '"name"' in script and '"y"' in script:
        return "named_points"

    if "[[" in script and "]," in script:
        return "xy_pairs"

    return "unknown"
def _build_chart_semantics(schema: dict, script: str) -> str:

    chart_type = schema.get("highcharts_type", "unknown")
    binding = schema.get("binding", "")
    mapping = schema.get("data_mapping", {})

    category = mapping.get("category_column")
    numeric = mapping.get("numeric_column")

    data_shape = _infer_data_shape_from_script(script)

    # categorical charts
    if data_shape == "categorical_numeric" or binding == "category_series":

        return f"""
CHART SEMANTICS
---------------
Chart type: {chart_type}

DATA FORMAT
categories → x-axis labels
values → numeric values

Axis mapping:
xAxis → categorical
yAxis → numeric

Rules:
• categories belong in xAxis.categories
• numeric values belong in series.data
• numeric thresholds must go in yAxis.plotLines
• NEVER add plotLines to xAxis for categorical charts
"""

    # scatter / bubble
    if data_shape == "xy_pairs" or binding == "xy_pairs":

        return f"""
CHART SEMANTICS
---------------
Chart type: {chart_type}

DATA FORMAT
[x, y] numeric pairs

Axis mapping:
xAxis → numeric
yAxis → numeric

Rules:
• horizontal reference lines → yAxis.plotLines
• vertical reference lines → xAxis.plotLines
"""

    # pie / funnel
    if data_shape == "named_points" or binding == "named_points":

        return f"""
CHART SEMANTICS
---------------
Chart type: {chart_type}

DATA FORMAT
{{name, y}}

Rules:
• this chart has NO axes
• axis modifications are invalid
• reference lines are not supported
"""

    return f"""
CHART SEMANTICS
---------------
Chart type: {chart_type}

Use Highcharts documentation knowledge to determine axis placement.
"""
# ══════════════════════════════════════════════════════════════
# FIND MATCH
# ══════════════════════════════════════════════════════════════

def _find_match(file_lines: list, hunk: dict) -> int:
    """
    Finds the index in file_lines where minus_lines start.

    Three-tier strategy:
        1. Exact  — context + minus lines verbatim
        2. Normalized — strip whitespace for comparison
        3. Fallback — single minus line with context disambiguation
    """
    ctx   = hunk["context_before"]
    minus = hunk["minus_lines"]
    n_m   = len(minus)
    n_c   = len(ctx)

    def matches(file_slice, pattern, normalize=False):
        if len(file_slice) != len(pattern):
            return False
        for f, p in zip(file_slice, pattern):
            fa = f.strip() if normalize else f
            pa = p.strip() if normalize else p
            if fa != pa:
                return False
        return True

    # Strategy 1: exact
    for i in range(len(file_lines) - n_m + 1):
        if n_c:
            s = i - n_c
            if s < 0 or not matches(file_lines[s:i], ctx):
                continue
        if matches(file_lines[i:i + n_m], minus):
            #print(f"[PATCH] Exact match at line {i}")
            logger.debug(f"Exact match at line {i}")
            return i

    # Strategy 2: normalized
    for i in range(len(file_lines) - n_m + 1):
        if n_c:
            s = i - n_c
            if s < 0 or not matches(file_lines[s:i], ctx, normalize=True):
                continue
        if matches(file_lines[i:i + n_m], minus, normalize=True):
            #print(f"[PATCH] Normalized match at line {i}")
            logger.debug(f"Normalized match at line {i}")
            return i

    # Strategy 3: single anchor with context disambiguation
    if n_m == 1:
        anchor = minus[0].strip()
        hits   = [i for i, l in enumerate(file_lines) if l.strip() == anchor]

        if len(hits) == 1:
            #print(f"[PATCH] Single-anchor match at line {hits[0]}")
            logger.debug(f"Single-anchor match at line {hits[0]}")
            return hits[0]

        if len(hits) > 1 and n_c > 0:
            for idx in hits:
                s = idx - n_c
                if s >= 0 and matches(file_lines[s:idx], ctx, normalize=True):
                    #print(f"[PATCH] Disambiguated anchor at line {idx}")
                    logger.debug(f"Disambiguated anchor at line {idx}")
                    return idx

    raise ValueError(
        f"No match found for hunk.\n"
        f"  First minus: {repr(minus[0] if minus else 'N/A')}\n"
        f"  Context:     {ctx}"
    )


# ══════════════════════════════════════════════════════════════
# INDENT CORRECTION
# ══════════════════════════════════════════════════════════════

def _fix_indent(plus_lines: list, minus_lines: list, matched_file_line: str) -> list:
    """
    Corrects indentation of plus_lines based on the actual
    indentation of the matched line in the file.
    Preserves relative indentation within the block.
    """
    if not minus_lines or not plus_lines:
        return plus_lines

    actual = len(matched_file_line) - len(matched_file_line.lstrip())
    llm    = len(minus_lines[0]) - len(minus_lines[0].lstrip())
    delta  = actual - llm

    result = []
    for line in plus_lines:
        if not line.strip():
            result.append("")
            continue
        li    = len(line) - len(line.lstrip())
        new_i = max(0, li + delta)
        result.append(" " * new_i + line.lstrip())

    return result


# ══════════════════════════════════════════════════════════════
# JS VALIDATORS
# ══════════════════════════════════════════════════════════════

def _check_js_syntax(js_code: str):
    """
    Validates JS syntax using node if available,
    falls back to Python heuristic checks otherwise.
    """
    import subprocess, tempfile, shutil

    node_bin = shutil.which("node") or shutil.which("node.exe")

    if not node_bin:
        #print("[PATCH] node not found — using heuristic syntax check")
        logger.warning("node not found — using heuristic syntax check")
        _check_js_syntax_heuristic(js_code)
        return

    wrapped = "(function(){\n" + js_code + "\n});"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".js", delete=False, encoding="utf-8"
    ) as f:
        f.write(wrapped)
        tmp = f.name

    try:
        result = subprocess.run(
            [node_bin, "--check", tmp],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            err = result.stderr.replace(tmp, "script.js")
            raise ValueError(f"JS syntax error:\n{err}")
        logger.info("JS syntax OK (node)")
        #print("[PATCH] JS syntax OK (node)")
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _check_js_syntax_heuristic(js_code: str):
    """
    Python-based JS syntax checks when node is unavailable.
    Catches unbalanced braces/brackets and missing commas.
    """
    pairs   = {"{": "}", "[": "]", "(": ")"}
    closing = set(pairs.values())
    stack   = []
    in_str  = False
    str_ch  = None

    for ch in js_code:
        if in_str:
            if ch == str_ch:
                in_str = False
        elif ch in ('"', "'", "`"):
            in_str = True
            str_ch = ch
        elif ch in pairs:
            stack.append(pairs[ch])
        elif ch in closing:
            if not stack or stack[-1] != ch:
                raise ValueError(f"JS syntax: unmatched closing '{ch}'")
            stack.pop()

    if stack:
        raise ValueError(f"JS syntax: unclosed '{stack[-1]}'")

    lines = js_code.splitlines()
    for i in range(len(lines) - 1):
        cur = lines[i].rstrip()
        nxt = lines[i + 1]
        if re.match(r".*[}\]]\s*$", cur) and not cur.rstrip().endswith(","):
            if re.match(r"\s{2,}[a-zA-Z_$]", nxt) and not nxt.strip().startswith("//"):
                if "Highcharts.chart" not in cur and "function" not in cur:
                    raise ValueError(
                        f"JS syntax: likely missing comma at line {i+1}: "
                        f"{cur.strip()} / {nxt.strip()}"
                    )
    logger.info("JS syntax OK (heuristic)")
    #print("[PATCH] JS syntax OK (heuristic)")


def _check_js_structure(js_code: str):
    """
    Validates structural correctness — e.g. plotLines must be
    inside xAxis/yAxis, not at chart root level.
    """
    lines   = js_code.splitlines()
    depth   = 0
    in_axis = False

    for i, line in enumerate(lines):
        depth += line.count("{") - line.count("}")
        depth += line.count("[") - line.count("]")

        if re.match(r"\s*(y|x)Axis\s*:", line):
            in_axis = True
        if depth <= 1:
            in_axis = False

        if re.match(r"\s*plot(Lines|Bands)\s*:", line):
            if not in_axis:
                prop = "plotLines" if "plotLines" in line else "plotBands"
                raise ValueError(
                    f"Line {i+1}: '{prop}' must be inside yAxis or xAxis, "
                    f"not at chart root level."
                )

    #print("[PATCH] JS structure OK")
    logger.info("JS structure OK")

# ══════════════════════════════════════════════════════════════
# APPLY PATCH
# ══════════════════════════════════════════════════════════════

def apply_patch(patch_text: str, files: dict, data_block=None) -> dict:
    if patch_text == "NO_CHANGES":
        #print("[PATCH] NO_CHANGES — skipping")
        logger.info("NO_CHANGES — skipping")
        return files

    updated = dict(files)

    body = re.sub(r"^\*\*\* Begin Patch\s*\n?", "", patch_text)
    body = re.sub(r"\n?\*\*\* End Patch\s*$", "", body)

    file_blocks = re.split(r"(?=\*\*\* Update File:)", body.strip())

    for block in file_blocks:
        block = block.strip()
        if not block:
            continue

        m = re.match(r"\*\*\* Update File:\s*([^\n]+)\n(.*)", block, re.DOTALL)
        if not m:
            continue

        fname        = m.group(1).strip()
        hunk_section = m.group(2)

        if fname not in updated:
            raise ValueError(f"File '{fname}' not in project")

        #print(f"\n[PATCH] File: {fname}")
        logger.info(f"File: {fname}")
        file_lines = updated[fname].splitlines()
        hunks      = _parse_hunks(hunk_section)

        #print(f"[PATCH] {len(hunks)} hunk(s)")
        logger.info(f"{len(hunks)} hunk(s)")
        for idx, hunk in enumerate(hunks):
            minus = hunk["minus_lines"]
            plus  = hunk["plus_lines"]

            if not minus:
                raise ValueError(
                    f"Hunk {idx+1} has no '-' lines. "
                    f"Use minus-anchor technique."
                )

            match_i    = _find_match(file_lines, hunk)
            fixed_plus = _fix_indent(plus, minus, file_lines[match_i])

            del file_lines[match_i: match_i + len(minus)]
            for offset, line in enumerate(fixed_plus):
                file_lines.insert(match_i + offset, line)

            #print(f"[PATCH] Hunk {idx+1} applied at line {match_i}")
            logger.info(f"Hunk {idx+1} applied at line {match_i}")
        updated[fname] = "\n".join(file_lines) + "\n"

        # Restore DATA block
        if fname == "script.js" and data_block:
            if "__DATA__" in updated[fname]:
                updated[fname] = updated[fname].replace(
                    "const DATA = __DATA__;",
                    data_block
                )

        # Integrity checks
        if fname == "script.js":
            c = updated[fname]

            if c.count("{") != c.count("}"):
                raise ValueError("Brace mismatch after patch")
            if c.count("[") != c.count("]"):
                raise ValueError("Bracket mismatch after patch")
            if "Highcharts.chart" not in c:
                raise ValueError("Highcharts.chart removed by patch")
            if "const DATA" not in c:
                raise ValueError("DATA declaration removed by patch")

            _check_js_structure(c)
            _check_js_syntax(c)

    #print("[PATCH] Done")
    logger.info("Done")
    return updated


# ══════════════════════════════════════════════════════════════
# SEMANTIC PATCH VALIDATOR
# ══════════════════════════════════════════════════════════════

def _warn_if_patch_unrelated(patch_text: str, instruction: str) -> None:
    """
    Warns when the patch's plus lines don't contain values
    related to the instruction — catches reasoning-to-patch
    disconnect without blocking valid patches.
    """
    if patch_text in ("NO_CHANGES", ""):
        return

    plus_lines = [
        l[1:].strip()
        for l in patch_text.splitlines()
        if l.startswith("+") and not l.startswith("+++")
    ]
    plus_text         = " ".join(plus_lines).lower()
    instruction_lower = instruction.lower()

    COLOR_WORDS = ["orange", "red", "blue", "green", "yellow", "purple",
                   "white", "black", "pink", "teal", "#", "map", "color:"]
    SIZE_WORDS  = ["px", "pt", "em", "rem", "bold", "italic"]

    if any(w in instruction_lower for w in ["color", "colour"]):
        for w in COLOR_WORDS:
            if w in instruction_lower and w in plus_text:
                return
        print(
            f"[PATCH WARN] Color instruction but no color in plus lines.\n"
            f"  Instruction: {instruction}\n"
            f"  Plus lines:  {plus_lines}"
        )

    if any(w in instruction_lower for w in ["font size", "fontsize", "font-size"]):
        if any(w in plus_text for w in SIZE_WORDS):
            return
        print(
            f"[PATCH WARN] Font size instruction but no size value in plus lines.\n"
            f"  Plus lines: {plus_lines}"
        )


# ══════════════════════════════════════════════════════════════
# GENERATE PATCH
# ══════════════════════════════════════════════════════════════

def generate_patch(files: dict, instruction: str, previous_error: str = None,schema=None) -> str:
    

    script_content = files["script.js"]

    data_block, stripped_script = _extract_data_block(script_content)

    if stripped_script:
        script_content = stripped_script
    
    
    script_lines   = script_content.splitlines()
    numbered       = "\n".join(
        f"{i+1:3}: {l}" for i, l in enumerate(script_lines)
    )


    error_block = ""
    if previous_error:
        error_block = (
            f"PREVIOUS ATTEMPT FAILED:\n"
            f"  Error: {previous_error}\n\n"
            f"Look at the numbered file carefully.\n"
            f"Find the correct line and copy it EXACTLY as '-'.\n\n"
        )
    schema_block = ""

    if schema:

        semantics = _build_chart_semantics(schema, script_content)

        schema_block = f"""
        CURRENT CHART SCHEMA
         --------------------
        {json.dumps(schema, indent=2)}

        {semantics}
        """
    user_prompt = (
       # f"{schema_block}\n\n"
        f"IMPORTANT: Follow the CHART SEMANTICS rules strictly.\n\n"
        f"SCRIPT.JS :\n{script_content}\n\n"
        f"{error_block}"
        f"USER REQUEST: {instruction}\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Find the exact line NUMBER containing the value to change.\n"
        f"2. Copy that line EXACTLY as your '-' line "
        f"(same text, same indentation).\n"
        f"3. Write replacement as '+' lines.\n"
        f"4. Every hunk MUST have at least one '-' line.\n"
        f"Return PATCH ONLY."
    )

    print(f"[PATCH] Prompt size: {len(user_prompt) + len(SYSTEM_PROMPT)} chars")

    response = client.chat.completions.create(

        model=MODEL,

        temperature=0,

        max_tokens=2000,

        stop=["Explanation:", "```"],

        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
    )
    usage = response.usage

    if usage:
        print("\n[PATCH] Token usage:")
        print(f"  prompt_tokens     : {usage.prompt_tokens}")
        print(f"  completion_tokens : {usage.completion_tokens}")
        print(f"  total_tokens      : {usage.total_tokens}\n")
    """ print("=================================================================================================")
        print(script_content)
        print("===========================================================================================================")
        raw     = response.choices[0].message.content
    
    print("=========================================================================================")
    print(raw)
    print("=========================================================================================================")
    cleaned = clean_llm_output(raw)
    print("===========cleaned==============================================================================")
    print(cleaned)
    print("=============================================================================================================")"""
    raw     = response.choices[0].message.content
    cleaned = clean_llm_output(raw)
    logger.debug(f"Script sent to LLM:\n{script_content}")
    logger.debug(f"LLM raw output:\n{raw}")
    logger.debug(f"Cleaned output:\n{cleaned}")
    if not cleaned:
        raise ValueError("LLM returned no valid patch output")

    validated = validate_diff(cleaned)
    _warn_if_patch_unrelated(validated, instruction)

    return validated




