REQUIRED_MARKERS = [
    "--- style.css ---",
    "--- script.js ---"
]

def validate_codegen_output(text: str):
    missing = [m for m in REQUIRED_MARKERS if m not in text]

    if missing:
        raise ValueError(
            "LLM violated file protocol. Missing markers: " + ", ".join(missing)
        )

    return text