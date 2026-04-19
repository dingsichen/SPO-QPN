"""Termination guards for targeted SPO-QPN exploration.

This module deliberately uses a finite abstraction for the pre-check:
it tracks the control marking together with the most recent branch name
that produced each live control place / quantum register. That is enough
to evaluate predecessor guards while avoiding any accumulation of visible
history, so the guard stays finite on bounded models.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .model import AnalysisSpec, Branch, SPOQPN, Transition


@dataclass(frozen=True)
class AbstractState:
    marking: frozenset[str]
    control_sources: tuple[tuple[str, str], ...]
    quantum_sources: tuple[tuple[str, str], ...]


def ensure_terminating_search_space(model: SPOQPN, analysis: AnalysisSpec) -> None:
    if not analysis.enforce_termination_guard:
        return
    if analysis.max_total_events is not None:
        return
    tau_graph = _reachable_tau_graph(model, analysis)
    cycle = _find_cycle(tau_graph)
    if cycle is not None:
        formatted = " -> ".join(_format_state(state) for state in cycle)
        raise ValueError(
            "Targeted exploration may diverge because the allowed tau-subgraph contains a reachable cycle. "
            f"Cycle witness: {formatted}. Add a structural cut, forbid the looping transition, or "
            "provide an explicit event bound."
        )


def _reachable_tau_graph(
    model: SPOQPN,
    analysis: AnalysisSpec,
) -> dict[AbstractState, set[AbstractState]]:
    forbidden = set(analysis.forbidden_transitions)
    allowed = [transition for transition in model.transitions if transition.name not in forbidden]
    initial = _initial_state(model)
    reachable: set[AbstractState] = {initial}
    worklist = deque([initial])
    tau_graph: dict[AbstractState, set[AbstractState]] = {initial: set()}
    while worklist:
        state = worklist.popleft()
        tau_graph.setdefault(state, set())
        for transition in allowed:
            if not _enabled(state.marking, transition):
                continue
            for branch in transition.branches:
                if not _branch_guard_satisfied(state, branch):
                    continue
                successor = _fire_branch(state, transition, branch)
                if branch.label == analysis.tau_label:
                    tau_graph[state].add(successor)
                if successor not in reachable:
                    reachable.add(successor)
                    worklist.append(successor)
                tau_graph.setdefault(successor, set())
    return tau_graph


def _initial_state(model: SPOQPN) -> AbstractState:
    return AbstractState(
        marking=model.initial_marking,
        control_sources=tuple(sorted((place, "init") for place in model.initial_marking)),
        quantum_sources=tuple(sorted((register.name, "init") for register in model.quantum_registers)),
    )


def _branch_guard_satisfied(state: AbstractState, branch: Branch) -> bool:
    control_lookup = dict(state.control_sources)
    quantum_lookup = dict(state.quantum_sources)
    for source, allowed in branch.predecessor_guards:
        kind, name = source.split(":", 1)
        if kind == "place":
            producer = control_lookup.get(name)
        elif kind == "register":
            producer = quantum_lookup.get(name)
        else:
            raise ValueError(f"Unsupported predecessor guard kind: {kind}")
        if producer is None or producer not in allowed:
            return False
    return True


def _enabled(marking: frozenset[str], transition: Transition) -> bool:
    return transition.pre.issubset(marking) and not ((marking - transition.pre) & transition.post)


def _fire_branch(
    state: AbstractState,
    transition: Transition,
    branch: Branch,
) -> AbstractState:
    control_lookup = dict(state.control_sources)
    quantum_lookup = dict(state.quantum_sources)
    for place in transition.pre:
        control_lookup.pop(place, None)
    for place in transition.post:
        control_lookup[place] = branch.name
    for register in transition.access:
        quantum_lookup[register] = branch.name
    return AbstractState(
        marking=_fire(state.marking, transition),
        control_sources=tuple(sorted(control_lookup.items())),
        quantum_sources=tuple(sorted(quantum_lookup.items())),
    )


def _fire(marking: frozenset[str], transition: Transition) -> frozenset[str]:
    return frozenset((marking - transition.pre) | transition.post)


def _find_cycle(graph: dict[AbstractState, set[AbstractState]]) -> list[AbstractState] | None:
    color: dict[AbstractState, int] = {}
    stack: list[AbstractState] = []

    def dfs(node: AbstractState) -> list[AbstractState] | None:
        color[node] = 1
        stack.append(node)
        for successor in graph[node]:
            state = color.get(successor, 0)
            if state == 0:
                cycle = dfs(successor)
                if cycle is not None:
                    return cycle
            elif state == 1:
                index = stack.index(successor)
                return stack[index:] + [successor]
        stack.pop()
        color[node] = 2
        return None

    for node in graph:
        if color.get(node, 0) == 0:
            cycle = dfs(node)
            if cycle is not None:
                return cycle
    return None


def _format_state(state: AbstractState) -> str:
    marking = "{" + ",".join(sorted(state.marking)) + "}"
    sources = ",".join(f"{name}:{branch}" for name, branch in state.control_sources)
    return f"{marking}[{sources}]"
