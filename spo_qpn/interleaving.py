"""Naive interleaving-baseline exploration over observable words."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass

from .analysis import (
    EventOccurrence,
    _branch_guard_satisfied,
    _control_condition,
    _enabled_transitions,
    _evaluate_secret,
    _linearize_configuration,
    _producer_from_condition,
    _quantum_condition,
    _validate_concurrency_assumptions,
    _within_event_bounds,
)
from .backend import Backend, DenseMatrixBackend
from .model import AnalysisSpec, Branch, SPOQPN, Transition
from .termination import ensure_terminating_search_space


@dataclass
class LinearExecutionRecord:
    node_id: int
    event_ids: frozenset[str]
    control_conditions: dict[str, str]
    quantum_conditions: dict[str, str]
    marking: frozenset[str]
    state: object
    observation: tuple[str, ...]
    depth: int
    visible_depth: int


@dataclass(frozen=True)
class LinearStructuralSuccessor:
    event_id: str
    event_ids: frozenset[str]
    control_conditions: dict[str, str]
    quantum_conditions: dict[str, str]
    marking: frozenset[str]
    observation: tuple[str, ...]
    depth: int
    visible_depth: int


def analyze_interleaving_sequences(
    model: SPOQPN,
    target_sequences: tuple[tuple[str, ...], ...] | list[tuple[str, ...]],
    analysis: AnalysisSpec | None = None,
    backend: Backend | None = None,
) -> dict[str, object]:
    model.validate()
    analysis = analysis or AnalysisSpec()
    backend = backend or DenseMatrixBackend(model)
    ensure_terminating_search_space(model, analysis)
    sequence_family = tuple(tuple(sequence) for sequence in target_sequences)
    if not sequence_family:
        raise ValueError("Interleaving baseline requires at least one target sequence.")
    prefix_family = _sequence_prefixes(sequence_family)
    target_set = set(sequence_family)
    event_registry: dict[str, EventOccurrence] = {}
    initial = LinearExecutionRecord(
        node_id=0,
        event_ids=frozenset(),
        control_conditions={place: _control_condition(place, None) for place in model.initial_marking},
        quantum_conditions={register: _quantum_condition(register, None) for register in model.register_names()},
        marking=model.initial_marking,
        state=backend.initialize(),
        observation=(),
        depth=0,
        visible_depth=0,
    )
    records: dict[int, LinearExecutionRecord] = {0: initial}
    has_tau_successor: set[int] = set()
    worklist = deque([initial])
    next_node_id = 1
    while worklist:
        current = worklist.popleft()
        enabled = _enabled_transitions(model, current.marking, analysis)
        _validate_concurrency_assumptions(enabled)
        for transition in enabled:
            for branch in transition.branches:
                if not _branch_guard_satisfied(current, branch, event_registry):
                    continue
                if not _within_event_bounds(current, branch, analysis):
                    continue
                successor = _build_linear_successor(
                    current=current,
                    transition=transition,
                    branch=branch,
                    event_registry=event_registry,
                    tau_label=analysis.tau_label,
                )
                if successor.observation not in prefix_family:
                    continue
                next_state = backend.update(current.state, transition, branch)
                if not backend.is_positive_weight(next_state, analysis.probability_tolerance):
                    continue
                candidate = LinearExecutionRecord(
                    node_id=next_node_id,
                    event_ids=successor.event_ids,
                    control_conditions=successor.control_conditions,
                    quantum_conditions=successor.quantum_conditions,
                    marking=successor.marking,
                    state=next_state,
                    observation=successor.observation,
                    depth=successor.depth,
                    visible_depth=successor.visible_depth,
                )
                next_node_id += 1
                records[candidate.node_id] = candidate
                if branch.label == analysis.tau_label:
                    has_tau_successor.add(current.node_id)
                worklist.append(candidate)
    return _aggregate_sequence_results(
        model=model,
        analysis=analysis,
        backend=backend,
        records=records,
        event_registry=event_registry,
        target_set=target_set,
        has_tau_successor=has_tau_successor,
    )


def _build_linear_successor(
    current: LinearExecutionRecord,
    transition: Transition,
    branch: Branch,
    event_registry: dict[str, EventOccurrence],
    tau_label: str,
) -> LinearStructuralSuccessor:
    control_preset = tuple(sorted((place, current.control_conditions[place]) for place in transition.pre))
    quantum_preset = tuple(sorted((register, current.quantum_conditions[register]) for register in transition.access))
    payload = {
        "transition": transition.name,
        "branch": branch.name,
        "control_preset": control_preset,
        "quantum_preset": quantum_preset,
    }
    event_id = "e:" + hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    predecessors = sorted(
        predecessor
        for predecessor in (
            _producer_from_condition(condition_id)
            for _, condition_id in control_preset + quantum_preset
        )
        if predecessor is not None
    )
    event_registry.setdefault(
        event_id,
        EventOccurrence(
            event_id=event_id,
            transition=transition.name,
            branch=branch.name,
            label=branch.label,
            predecessors=tuple(predecessors),
        ),
    )
    next_control = dict(current.control_conditions)
    for place in transition.pre:
        next_control.pop(place, None)
    for place in transition.post:
        next_control[place] = _control_condition(place, event_id)
    next_quantum = dict(current.quantum_conditions)
    for register in transition.access:
        next_quantum[register] = _quantum_condition(register, event_id)
    next_event_ids = frozenset(set(current.event_ids) | {event_id})
    next_observation = current.observation if branch.label == tau_label else current.observation + (branch.label,)
    next_visible_depth = current.visible_depth + (0 if branch.label == tau_label else 1)
    return LinearStructuralSuccessor(
        event_id=event_id,
        event_ids=next_event_ids,
        control_conditions=next_control,
        quantum_conditions=next_quantum,
        marking=frozenset(next_control.keys()),
        observation=next_observation,
        depth=current.depth + 1,
        visible_depth=next_visible_depth,
    )


def _aggregate_sequence_results(
    model: SPOQPN,
    analysis: AnalysisSpec,
    backend: Backend,
    records: dict[int, LinearExecutionRecord],
    event_registry: dict[str, EventOccurrence],
    target_set: set[tuple[str, ...]],
    has_tau_successor: set[int],
) -> dict[str, object]:
    aggregates: dict[tuple[tuple[str, ...], int], object] = {}
    flags: dict[tuple[tuple[str, ...], int], bool] = defaultdict(bool)
    witnesses: dict[tuple[tuple[str, ...], int], list[str]] = {}
    terminal_count = 0
    for record in records.values():
        if record.observation not in target_set or record.node_id in has_tau_successor:
            continue
        terminal_count += 1
        secret_bit = _evaluate_secret(model, record, event_registry)
        key = (record.observation, secret_bit)
        reduced = backend.reduce_to_attacker(record.state)
        aggregates[key] = reduced if key not in aggregates else backend.add_operator(aggregates[key], reduced)
        flags[key] = True
        witnesses.setdefault(key, _linearize_configuration(record.event_ids, event_registry))
    reports = []
    worst_case_leakage = 0.0
    structural_opaque = True
    for sequence in sorted(target_set):
        secret_present = flags[(sequence, 1)]
        nonsecret_present = flags[(sequence, 0)]
        if secret_present and not nonsecret_present:
            structural_opaque = False
        secret_aggregate = aggregates.get((sequence, 1))
        nonsecret_aggregate = aggregates.get((sequence, 0))
        secret_weight = backend.operator_trace_float(secret_aggregate) if secret_aggregate is not None else 0.0
        nonsecret_weight = backend.operator_trace_float(nonsecret_aggregate) if nonsecret_aggregate is not None else 0.0
        leakage_info = backend.leakage(
            secret_aggregate,
            nonsecret_aggregate,
            analysis.probability_tolerance,
        )
        worst_case_leakage = max(worst_case_leakage, float(leakage_info["value_float"]))
        reports.append(
            {
                "observation_sequence": list(sequence),
                "structural_secret_reachable": secret_present,
                "structural_nonsecret_reachable": nonsecret_present,
                "joint_probability": {"secret": secret_weight, "nonsecret": nonsecret_weight},
                "leakage": leakage_info["value_float"],
                "leakage_exact": leakage_info.get("value_exact"),
                "secret_posterior": backend.pretty_operator(leakage_info["secret_normalized"]) if leakage_info["secret_normalized"] is not None else None,
                "nonsecret_posterior": backend.pretty_operator(leakage_info["nonsecret_normalized"]) if leakage_info["nonsecret_normalized"] is not None else None,
                "secret_witness": witnesses.get((sequence, 1)),
                "nonsecret_witness": witnesses.get((sequence, 0)),
            }
        )
    return {
        "backend": getattr(backend, "backend_name", backend.__class__.__name__),
        "analysis_scope": "targeted_sequences",
        "structural_current_state_opaque": structural_opaque,
        "epsilon_current_state_opaque": worst_case_leakage <= analysis.epsilon,
        "epsilon": analysis.epsilon,
        "worst_case_leakage": worst_case_leakage,
        "observation_reports": reports,
        "reachable_linear_state_count": len(records),
        "maximal_observation_sequence_count": len(reports),
        "terminal_execution_count": terminal_count,
    }


def _sequence_prefixes(target_sequences: tuple[tuple[str, ...], ...]) -> set[tuple[str, ...]]:
    family: set[tuple[str, ...]] = {()}
    for sequence in target_sequences:
        for length in range(1, len(sequence) + 1):
            family.add(sequence[:length])
    return family
