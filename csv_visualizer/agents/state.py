from typing import TypedDict, Optional, Dict, Any
import pandas as pd


class AgentState(TypedDict, total=False):

    
    query: str

    
    df: pd.DataFrame


    profile: Dict[str, Any]


    schema: Dict[str, Any]


    files: Dict[str, str]

    
    patch: Optional[str]

    
    tool: Optional[str]
    stop: bool