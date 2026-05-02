"""
Run this once to pre-fetch and cache Highcharts API
for all supported chart types.

Usage:
    python scripts/warm_api_cache.py
"""
from core.highcharts_api import get_chart_api_context

CHART_TYPES = [
    "boxplot", "line", "column", "bar", "pie",
    "funnel", "pyramid", "scatter", "area",
    "heatmap", "treemap", "histogram", "bellcurve",
    "bubble", "spline"
]

if __name__ == "__main__":
    for chart_type in CHART_TYPES:
        print(f"\nWarming cache for: {chart_type}")
        context = get_chart_api_context(chart_type)
        if context:
            print(f"  ✓ {len(context)} chars cached")
        else:
            print(f"  ✗ Failed")