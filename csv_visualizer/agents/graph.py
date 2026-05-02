from langgraph.graph import StateGraph, END
from langgraph.pregel._retry import RetryPolicy

from csv_visualizer.agents.state import AgentState

from csv_visualizer.agents.nodes import (
    router_node,
    planner_node,
    codegen_node,
    patch_node
)


def should_run_codegen(state):
    return "end" if state.get("stop") else "codegen"


def build_graph():
    graph = StateGraph(AgentState)

    retry_policy = RetryPolicy(
        max_attempts=2,
        backoff_factor=1.5,
    )

    # router handles its own errors and never raises — no RetryPolicy needed
    graph.add_node("router", router_node)
    graph.add_node("planner", planner_node, retry_policy=retry_policy)
    graph.add_node("codegen", codegen_node, retry_policy=retry_policy)
    graph.add_node("patch",   patch_node,   retry_policy=retry_policy)

    graph.set_entry_point("router")

    def route_decision(state: AgentState):
        if state.get("stop"):
            return "end_route"
        tool = state.get("tool")
        if tool == "planner_tool":
            return "planner"
        if tool == "patch_tool":
            return "patch"
        if tool == "codegen_tool":
            return "codegen"
        return "planner"

    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "planner":   "planner",
            "patch":     "patch",
            "codegen":   "codegen",
            "end_route": END,
        },
    )

    graph.add_conditional_edges("planner", should_run_codegen, {"end": END, "codegen": "codegen"})
    graph.add_edge("codegen", END)
    graph.add_edge("patch",   END)

    return graph.compile()
