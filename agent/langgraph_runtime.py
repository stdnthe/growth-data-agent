from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .agent_loop import BoundedLoopController, LoopObservation
from .models import AgentDecision


class AgentGraphState(TypedDict):
    decision: AgentDecision
    observation: LoopObservation | None


ActionExecutor = Callable[[AgentDecision], LoopObservation]
DecisionRecorder = Callable[[AgentDecision], None]
NextDecision = Callable[[LoopObservation], AgentDecision]


def run_bounded_agent_graph(
    *,
    initial_decision: AgentDecision,
    controller: BoundedLoopController,
    execute_action: ActionExecutor,
    record_decision: DecisionRecorder,
    decide_next: NextDecision | None = None,
) -> AgentGraphState:
    """Run the bounded analytics loop through explicit LangGraph conditional edges."""

    def action_node(state: AgentGraphState) -> dict[str, LoopObservation]:
        controller.mark_action_executed()
        return {"observation": execute_action(state["decision"])}

    def decide_node(state: AgentGraphState) -> dict[str, AgentDecision]:
        observation = state["observation"]
        if observation is None:
            decision = initial_decision
        elif decide_next is not None:
            decision = decide_next(observation)
        elif observation.stage == "schema":
            decision = controller.after_schema_inspection()
        else:
            decision = controller.after_observation(observation)
        record_decision(decision)
        return {"decision": decision}

    def terminal_node(_: AgentGraphState) -> dict[str, LoopObservation | None]:
        return {}

    def route_action(state: AgentGraphState) -> str:
        return state["decision"].action

    builder = StateGraph(AgentGraphState)
    for action in ("query_metric", "inspect_schema", "run_gmv_recipe", "run_fulfillment_recipe"):
        builder.add_node(action, action_node)
        builder.add_edge(action, "decide")
    builder.add_node("decide", decide_node)
    for action in ("finish", "clarify", "unsupported", "stop"):
        builder.add_node(action, terminal_node)
        builder.add_edge(action, END)

    destinations = {
        action: action
        for action in (
            "query_metric",
            "inspect_schema",
            "run_gmv_recipe",
            "run_fulfillment_recipe",
            "finish",
            "clarify",
            "unsupported",
            "stop",
        )
    }
    builder.add_conditional_edges(START, route_action, destinations)
    builder.add_conditional_edges("decide", route_action, destinations)
    graph = builder.compile(name="growth_copilot_bounded_loop")
    recursion_limit = max(10, controller.max_steps * 3 + 5)
    return graph.invoke(
        {"decision": initial_decision, "observation": None},
        {"recursion_limit": recursion_limit},
    )
