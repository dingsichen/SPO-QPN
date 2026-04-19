"""JSON loading helpers for SPO-QPN models and analysis specs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .linear_algebra import parse_matrix, parse_vector
from .model import (
    AnalysisSpec,
    Branch,
    QuantumRegister,
    SPOQPN,
    SecretPredicate,
    Transition,
    build_density_from_pure_state,
)


def load_model_from_json(path: str | Path) -> SPOQPN:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    registers = tuple(
        QuantumRegister(name=item["name"], dimension=item.get("dimension", 2))
        for item in data["quantum_registers"]
    )
    transitions = []
    for item in data["transitions"]:
        branches = []
        for branch_data in item["branches"]:
            kraus = [parse_matrix(matrix) for matrix in branch_data["kraus"]]
            branches.append(
                Branch.from_matrices(
                    name=branch_data["name"],
                    label=branch_data["label"],
                    kraus=kraus,
                    metadata=branch_data.get("metadata"),
                    symbolic_spec=branch_data.get("symbolic"),
                    predecessor_guards=branch_data.get("predecessor_guards"),
                )
            )
        transitions.append(
            Transition(
                name=item["name"],
                pre=frozenset(item.get("pre", [])),
                post=frozenset(item.get("post", [])),
                access=tuple(item.get("access", [])),
                branches=tuple(branches),
            )
        )
    initial_state = _parse_initial_state(data["initial_state"])
    model = SPOQPN(
        control_places=tuple(data["control_places"]),
        quantum_registers=registers,
        transitions=tuple(transitions),
        initial_marking=frozenset(data["initial_marking"]),
        initial_state=tuple(tuple(entry for entry in row) for row in initial_state),
        attacker_interface=tuple(data["attacker_interface"]),
        secret_predicate=_parse_secret_predicate(data["secret_predicate"]),
        observable_alphabet=tuple(data.get("observable_alphabet", [])),
        symbolic_initial_state=None if data.get("symbolic_initial_state") is None else _freeze_json_object(data["symbolic_initial_state"]),
    )
    model.validate()
    return model


def load_analysis_from_json(path: str | Path) -> AnalysisSpec:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return AnalysisSpec(
        epsilon=float(data.get("epsilon", 0.0)),
        target_observations=tuple(data.get("target_observations", [])),
        max_total_events=data.get("max_total_events"),
        max_observable_events=data.get("max_observable_events"),
        tau_label=data.get("tau_label", "tau"),
        probability_tolerance=float(data.get("probability_tolerance", 1e-12)),
        matrix_tolerance=float(data.get("matrix_tolerance", 1e-12)),
        forbidden_transitions=tuple(data.get("forbidden_transitions", [])),
        enforce_termination_guard=bool(data.get("enforce_termination_guard", True)),
    )


def _parse_initial_state(raw: dict[str, Any]):
    if raw["type"] == "density_matrix":
        return parse_matrix(raw["data"])
    if raw["type"] == "pure_state":
        return build_density_from_pure_state(parse_vector(raw["data"]))
    raise ValueError(f"Unsupported initial_state type: {raw['type']}")


def _parse_secret_predicate(raw: dict[str, Any]) -> SecretPredicate:
    kind = raw["type"]
    if kind == "marking_membership":
        return SecretPredicate.marking_membership(raw["markings"])
    if kind == "event_occurrence":
        return SecretPredicate.event_occurrence(raw["transitions"])
    if kind == "branch_occurrence":
        return SecretPredicate.branch_occurrence(raw["branches"])
    raise ValueError(f"Unsupported secret predicate type: {kind}")


def _freeze_json_object(value: Any) -> object:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_json_object(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze_json_object(item) for item in value)
    return value
