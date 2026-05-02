import re

AGG_WORDS = [
    "total","sum","average","avg","mean","count","min","max"
]

CHART_WORDS = [
    "histogram","scatter","line","bar","column","area",
    "pie","boxplot","heatmap","distribution"
]

RELATION_WORDS = [
    "by","vs","against","over","per"
]

def defines_visualization(query: str, profile: dict) -> bool:
    q = query.lower()

    columns = (
        profile.get("numeric_columns", [])
        + profile.get("categorical_columns", [])
        + profile.get("datetime_columns", [])
    )

    
    mentions_column = any(col.lower() in q for col in columns)

    mentions_chart = any(w in q for w in CHART_WORDS)
    mentions_relation = any(w in q for w in RELATION_WORDS)
    mentions_agg = any(w in q for w in AGG_WORDS)

    
    return mentions_column or mentions_chart or mentions_relation or mentions_agg