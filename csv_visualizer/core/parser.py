import re
from typing import Dict

REQUIRED_FILES = { "style.css", "script.js"}



def _sanitize_llm_output(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)

    return text.strip()




def normalize_filename(name: str) -> str:
    name = name.lower().strip()
    name = name.replace(" ", "")

    mapping = {
        "main.js": "script.js",
        "app.js": "script.js",
        "chart.js": "script.js",
        "scriptjs": "script.js",
        "styles.css": "style.css",
        "stylecss": "style.css",
        "index.htm": "index.html",
    }

    return mapping.get(name, name)




def parse_delimiter_blocks(text: str) -> Dict[str, str]:
    files = {}
    current = None
    buffer = []

    for line in text.split("\n"):
        match = re.fullmatch(r"-{3}\s*(.+?)\s*-{3}", line.strip())
        if match:
            if current:
                files[current] = "\n".join(buffer).strip()
            buffer = []

            fname = normalize_filename(match.group(1))
            current = fname if fname in REQUIRED_FILES else None
            continue

        if current:
            buffer.append(line)

    if current:
        files[current] = "\n".join(buffer).strip()

    return files





def parse_markdown_blocks(text: str) -> Dict[str, str]:
    files = {}
    pattern = r"```(.*?)\n([\s\S]*?)```"

    for lang, content in re.findall(pattern, text):
        fname = normalize_filename(lang)
        if fname in REQUIRED_FILES:
            files[fname] = content.strip()

    return files


def parse_heuristic(text: str) -> Dict[str, str]:
    files = {}

    # find html
    html_match = re.search(r"<!DOCTYPE html>[\s\S]*?</html>", text, re.IGNORECASE)
    if html_match:
        files["index.html"] = html_match.group(0).strip()

    # find css
    css_match = re.search(r"(body\s*\{[\s\S]*?\})", text)
    if css_match:
        files["style.css"] = css_match.group(0).strip()

    # find js
    js_match = re.search(r"Highcharts\.chart\([\s\S]*?\);", text)
    if js_match:
        files["script.js"] = js_match.group(0).strip()

    return files




def parse_llm_output(text: str) -> Dict[str, str]:

    text = _sanitize_llm_output(text)

    # Try strategies in order
    files = parse_delimiter_blocks(text)

    if len(files) < 3:
        files.update(parse_markdown_blocks(text))

    if len(files) < 3:
        files.update(parse_heuristic(text))

    
    missing = REQUIRED_FILES - files.keys()
    if missing:
        raise ValueError(f"Codegen failed. Missing files: {missing}")

    return files