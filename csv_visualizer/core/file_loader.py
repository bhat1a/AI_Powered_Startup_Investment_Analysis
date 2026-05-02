import os

def load_last_files(visual_id, version):
    base = os.path.join("generated", visual_id, f"v{version}")
    files = {}

    for name in ["index.html", "style.css", "script.js"]:
        with open(os.path.join(base, name), "r", encoding="utf-8") as f:
            files[name] = f.read()

    return files
