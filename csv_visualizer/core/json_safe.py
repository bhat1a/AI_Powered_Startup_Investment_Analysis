import pandas as pd
import numpy as np

def to_json_safe(obj):

    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [to_json_safe(v) for v in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()

    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()

    if isinstance(obj, (pd.Timedelta,)):
        return str(obj)

    if obj is pd.NaT:
        return None

    return obj
