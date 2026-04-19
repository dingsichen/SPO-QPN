"""Configuration-based exact opacity analysis for SPO-QPN models."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass

from .backend import Backend, DenseMatrixBackend
from .model import AnalysisSpec, Branch, SPOQPN, Transition
from .pomset import PomsetKey, build_target_family, build_target_prefix_family, canonical_pomset_key
from .termination import ensure_terminating_search_space


@dataclass(frozen=True)
class EventOccurrence:
    event_id: str
    transition: str
    branch: str
    label: str
    predecessors: tuple[str, ...]


@dataclass
class ConfigurationRecord:
    event_ids: frozenset[str]
    control_conditions: dict[str, str]
    quantum_conditions: dict[str, str]
    marking: frozenset[str]
    state: object
    observation: PomsetKey
    depth: int
    visible_depth: int


@dataclass(frozen=True)
class StructuralSuccessor:
    event_id: str
    event_ids: frozenset[str]
    control_conditions: dict[str, str]
    quantum_conditions: dict[str, str]
    marking: frozenset[str]
    observation: PomsetKey
    depth: int
    visible_depth: int


def analyze_model(
    model: SPOQPN,
    analysis: AnalysisSpec | None = None,
    backend: Backend | None = None,
) -> dict[str, object]:
    model.validate()
    analysis = analysis or AnalysisSpec()
    backend = backend or DenseMatrixBackend(model)
    ensure_terminating_search_space(model, analysis)
    event_registry: dict[str, EventOccurrence] = {}
    target_prefixes = (
        build_target_prefix_family(analysis.target_observations)
        if analysis.target_observations
        else None
    )
    target_family = (
        build_target_family(analysis.target_observations)
        if analysis.target_observations
        else None
    )
    transition_lookup = {transition.name: transition for transition in model.transitions}
    initial_record = ConfigurationRecord(
        event_ids=frozenset(),
        control_conditions={place: _control_condition(place, None) for place in model.initial_marking},
        quantum_conditions={register: _quantum_condition(register, None) for register in model.register_names()},
        marking=model.initial_marking,
        state=backend.initialize(),
        observation=PomsetKey(labels=(), order_pairs=()),
        depth=0,
        visible_depth=0,
    )
    visited: set[frozenset[str]] = {initial_record.event_ids}
    reachable: dict[frozenset[str], ConfigurationRecord] = {initial_record.event_ids: initial_record}
    reachable_tau_extensions: set[frozenset[str]] = set()
    worklist = deque([initial_record])
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
                successor = _build_structural_successor(
                    current=current,
                    transition=transition,
                    branch=branch,
                    event_registry=event_registry,
                    tau_label=analysis.tau_label,
                )
                if target_prefixes is not None and successor.observation not in target_prefixes:
                    continue
                if successor.event_ids in visited:
                    continue
                visited.add(successor.event_ids)
                next_state = backend.update(current.state, transition, branch)
                if not backend.is_positive_weight(next_state, analysis.probability_tolerance):
                    continue
                candidate = ConfigurationRecord(
                    event_ids=successor.event_ids,
                    control_conditions=successor.control_conditions,
                    quantum_conditions=successor.quantum_conditions,
                    marking=successor.marking,
                    state=next_state,
                    observation=successor.observation,
                    depth=successor.depth,
                    visible_depth=successor.visible_depth,
                )
                reachable[candidate.event_ids] = candidate
                if branch.label == analysis.tau_label:
                    reachable_tau_extensions.add(current.event_ids)
                worklist.append(candidate)
    return _aggregate_results(
        model=model,
        analysis=analysis,
        backend=backend,
        reachable=reachable,
        event_registry=event_registry,
        transition_lookup=transition_lookup,
        target_family=target_family,
        reachable_tau_extensions=reachable_tau_extensions,
    )


def _build_structural_successor(
    current: ConfigurationRecord,
    transition: Transition,
    branch: Branch,
    event_registry: dict[str, EventOccurrence],
    tau_label: str,
) -> StructuralSuccessor:
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
    next_visible_depth = current.visible_depth + (0 if branch.label == tau_label else 1)
    return StructuralSuccessor(
        event_id=event_id,
        event_ids=next_event_ids,
        control_conditions=next_control,
        quantum_conditions=next_quantum,
        marking=frozenset(next_control.keys()),
        observation=_compute_observation(next_event_ids, event_registry, tau_label),
        depth=current.depth + 1,
        visible_depth=next_visible_depth,
    )


def _aggregate_results(
    model: SPOQPN,
    analysis: AnalysisSpec,
    backend: Backend,
    reachable: dict[frozenset[str], ConfigurationRecord],
    event_registry: dict[str, EventOccurrence],
    transition_lookup: dict[str, Transition],
    target_family: set[PomsetKey] | None,
    reachable_tau_extensions: set[frozenset[str]],
) -> dict[str, object]:
    aggregates: dict[tuple[PomsetKey, int], object] = {}
    flags: dict[tuple[PomsetKey, int], bool] = defaultdict(bool)
    witnesses: dict[tuple[PomsetKey, int], list[str]] = {}
    observations_seen: set[PomsetKey] = set(target_family or [])
    for record in reachable.values():
        if target_family is None:
            observations_seen.add(record.observation)
        if not _is_maximal_unobservable_reach(
            record,
            reachable_tau_extensions,
            transition_lookup,
            event_registry,
            analysis.tau_label,
        ):
            continue
        if target_family is not None and record.observation not in target_family:
            continue
        secret_bit = _evaluate_secret(model, record, event_registry)
        key = (record.observation, secret_bit)
        reduced = backend.reduce_to_attacker(record.state)
        aggregates[key] = reduced if key not in aggregates else backend.add_operator(aggregates[key], reduced)
        flags[key] = True
        witnesses.setdefault(key, _linearize_configuration(record.event_ids, event_registry))
    observation_reports = []
    worst_case_leakage = 0.0
    structural_opaque = True
    for observation in sorted(observations_seen, key=lambda item: (item.labels, item.order_pairs)):
        secret_present = flags[(observation, 1)]
        nonsecret_present = flags[(observation, 0)]
        if secret_present and not nonsecret_present:
            structural_opaque = False
        secret_aggregate = aggregates.get((observation, 1))
        nonsecret_aggregate = aggregates.get((observation, 0))
        secret_weight = backend.operator_trace_float(secret_aggregate) if secret_aggregate is not None else 0.0
        nonsecret_weight = backend.operator_trace_float(nonsecret_aggregate) if nonsecret_aggregate is not None else 0.0
        leakage_info = backend.leakage(
            secret_aggregate,
            nonsecret_aggregate,
            analysis.probability_tolerance,
        )
        worst_case_leakage = max(worst_case_leakage, float(leakage_info["value_float"]))
        observation_reports.append(
            {
                "observation": observation.to_display_dict(),
                "structural_secret_reachable": secret_present,
                "structural_nonsecret_reachable": nonsecret_present,
                "joint_probability": {"secret": secret_weight, "nonsecret": nonsecret_weight},
                "leakage": leakage_info["value_float"],
                "leakage_exact": leakage_info.get("value_exact"),
                "secret_posterior": backend.pretty_operator(leakage_info["secret_normalized"]) if leakage_info["secret_normalized"] is not None else None,
                "nonsecret_posterior": backend.pretty_operator(leakage_info["nonsecret_normalized"]) if leakage_info["nonsecret_normalized"] is not None else None,
                "secret_witness": witnesses.get((observation, 1)),
                "nonsecret_witness": witnesses.get((observation, 0)),
            }
        )
    return {
        "backend": getattr(backend, "backend_name", backend.__class__.__name__),
        "analysis_scope": "targeted" if analysis.target_observations else "global",
        "target_observation_count": len(analysis.target_observations),
        "structural_current_state_opaque": structural_opaque,
        "epsilon_current_state_opaque": worst_case_leakage <= analysis.epsilon,
        "epsilon": analysis.epsilon,
        "worst_case_leakage": worst_case_leakage,
        "observation_reports": observation_reports,
        "reachable_configuration_count": len(reachable),
        "maximal_observation_count": len(observation_reports),
    }


def _compute_observation(
    event_ids: frozenset[str],
    event_registry: dict[str, EventOccurrence],
    tau_label: str,
) -> PomsetKey:
    visible = sorted(event_id for event_id in event_ids if event_registry[event_id].label != tau_label)
    labels = {event_id: event_registry[event_id].label for event_id in visible}
    strict_order = {event_id: set() for event_id in visible}
    memo: dict[str, set[str]] = {}

    def visible_ancestors(event_id: str) -> set[str]:
        cached = memo.get(event_id)
        if cached is not None:
            return cached
        result: set[str] = set()
        for predecessor in event_registry[event_id].predecessors:
            result.update(visible_ancestors(predecessor))
            if event_registry[predecessor].label != tau_label:
                result.add(predecessor)
        memo[event_id] = result
        return result

    for event_id in visible:
        strict_order[event_id] = visible_ancestors(event_id)
    return canonical_pomset_key(visible, labels, strict_order)


def _enabled_transitions(model: SPOQPN, marking: frozenset[str], analysis: AnalysisSpec) -> list[Transition]:
    enabled = []
    forbidden = set(analysis.forbidden_transitions)
    for transition in model.transitions:
        if transition.name in forbidden:
            continue
        if transition.pre.issubset(marking) and not ((marking - transition.pre) & transition.post):
            enabled.append(transition)
    return enabled


def _validate_concurrency_assumptions(enabled: list[Transition]) -> None:
    for left_index, left in enumerate(enabled):
        left_support = left.pre | left.post
        left_access = set(left.access)
        for right in enabled[left_index + 1 :]:
            if left_support.isdisjoint(right.pre | right.post) and (left_access & set(right.access)):
                raise ValueError(
                    "Two simultaneously enabled control-independent transitions overlap on quantum accesses: "
                    f"{left.name} and {right.name}."
                )


def _within_event_bounds(
    current: ConfigurationRecord,
    branch: Branch,
    analysis: AnalysisSpec,
) -> bool:
    if analysis.max_total_events is not None and current.depth + 1 > analysis.max_total_events:
        return False
    next_visible = current.visible_depth + (0 if branch.label == analysis.tau_label else 1)
    if analysis.max_observable_events is not None and next_visible > analysis.max_observable_events:
        return False
    return True


def _branch_guard_satisfied(
    current: ConfigurationRecord,
    branch: Branch,
    event_registry: dict[str, EventOccurrence],
) -> bool:
    for source, allowed in branch.predecessor_guards:
        kind, name = source.split(":", 1)
        if kind == "place":
            condition_id = current.control_conditions.get(name)
        elif kind == "register":
            condition_id = current.quantum_conditions.get(name)
        else:
            raise ValueError(f"Unsupported predecessor guard kind: {kind}")
        if condition_id is None:
            return False
        producer = _producer_from_condition(condition_id)
        producer_branch = "init" if producer is None else event_registry[producer].branch
        if producer_branch not in allowed:
            return False
    return True


def _is_maximal_unobservable_reach(
    record: ConfigurationRecord,
    reachable_tau_extensions: set[frozenset[str]],
    transition_lookup: dict[str, Transition],
    event_registry: dict[str, EventOccurrence],
    tau_label: str,
) -> bool:
    del transition_lookup
    del event_registry
    del tau_label
    return record.event_ids not in reachable_tau_extensions


def _evaluate_secret(
    model: SPOQPN,
    record: ConfigurationRecord,
    event_registry: dict[str, EventOccurrence],
) -> int:
    predicate = model.secret_predicate
    if predicate.kind == "marking_membership":
        markings = {tuple(marking) for marking in predicate.values}
        return int(tuple(sorted(record.marking)) in markings)
    if predicate.kind == "event_occurrence":
        watched = set(predicate.values)
        return int(any(event_registry[event_id].transition in watched for event_id in record.event_ids))
    if predicate.kind == "branch_occurrence":
        watched = set(predicate.values)
        return int(
            any(
                f"{event_registry[event_id].transition}/{event_registry[event_id].branch}" in watched
                for event_id in record.event_ids
            )
        )
    raise ValueError(f"Unsupported secret predicate kind: {predicate.kind}")
def _linearize_configuration(
    event_ids: frozenset[str],
    event_registry: dict[str, EventOccurrence],
) -> list[str]:
    pending = set(event_ids)
    done: set[str] = set()
    out: list[str] = []
    while pending:
        enabled = sorted(
            event_id
            for event_id in pending
            if set(event_registry[event_id].predecessors).issubset(done)
        )
        if not enabled:
            raise ValueError("Configuration contains a cyclic dependency.")
        chosen = enabled[0]
        occurrence = event_registry[chosen]
        out.append(f"{occurrence.transition}/{occurrence.branch}[{occurrence.label}]")
        pending.remove(chosen)
        done.add(chosen)
    return out


def _control_condition(place: str, producer_event_id: str | None) -> str:
    return f"cc|{place}|{producer_event_id or 'init'}"


def _quantum_condition(register: str, producer_event_id: str | None) -> str:
    return f"qq|{register}|{producer_event_id or 'init'}"


def _producer_from_condition(condition_id: str) -> str | None:
    producer = condition_id.rsplit("|", 1)[1]
    return None if producer == "init" else producer
