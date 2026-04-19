"""Core model dataclasses for SPO-QPN analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .linear_algebra import Matrix, dagger, dimension_product, identity, is_square, matrix_add, matrix_mul, outer, trace


@dataclass(frozen=True)
class QuantumRegister:
    name: str
    dimension: int = 2


@dataclass(frozen=True)
class Branch:
    name: str
    label: str
    kraus: tuple[tuple[tuple[complex, ...], ...], ...]
    metadata: tuple[tuple[str, str], ...] = ()
    symbolic_spec: tuple[tuple[str, object], ...] | None = None
    predecessor_guards: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @staticmethod
    def from_matrices(
        name: str,
        label: str,
        kraus: Sequence[Matrix],
        metadata: dict[str, str] | None = None,
        symbolic_spec: dict[str, object] | None = None,
        predecessor_guards: dict[str, Sequence[str] | str] | None = None,
    ) -> "Branch":
        frozen = tuple(tuple(tuple(entry for entry in row) for row in operator) for operator in kraus)
        normalized_symbolic = None if symbolic_spec is None else _freeze_object(symbolic_spec)
        normalized_guards = _normalize_predecessor_guards(predecessor_guards)
        return Branch(
            name=name,
            label=label,
            kraus=frozen,
            metadata=tuple(sorted((metadata or {}).items())),
            symbolic_spec=normalized_symbolic,
            predecessor_guards=normalized_guards,
        )

    def kraus_matrices(self) -> list[Matrix]:
        return [[list(row) for row in operator] for operator in self.kraus]

    def symbolic(self) -> dict[str, object] | None:
        if self.symbolic_spec is None:
            return None
        return _thaw_object(self.symbolic_spec)

    def guards(self) -> dict[str, list[str]]:
        return {source: list(branches) for source, branches in self.predecessor_guards}


@dataclass(frozen=True)
class Transition:
    name: str
    pre: frozenset[str]
    post: frozenset[str]
    access: tuple[str, ...]
    branches: tuple[Branch, ...]


@dataclass(frozen=True)
class SecretPredicate:
    kind: str
    values: tuple[object, ...]

    @staticmethod
    def marking_membership(markings: Sequence[Sequence[str]]) -> "SecretPredicate":
        return SecretPredicate(
            kind="marking_membership",
            values=tuple(tuple(sorted(marking)) for marking in markings),
        )

    @staticmethod
    def event_occurrence(transitions: Sequence[str]) -> "SecretPredicate":
        return SecretPredicate(kind="event_occurrence", values=tuple(sorted(transitions)))

    @staticmethod
    def branch_occurrence(qualified_branches: Sequence[str]) -> "SecretPredicate":
        return SecretPredicate(kind="branch_occurrence", values=tuple(sorted(qualified_branches)))


@dataclass(frozen=True)
class AnalysisSpec:
    epsilon: float = 0.0
    target_observations: tuple[dict[str, object], ...] = ()
    max_total_events: int | None = None
    max_observable_events: int | None = None
    tau_label: str = "tau"
    probability_tolerance: float = 1e-12
    matrix_tolerance: float = 1e-12
    forbidden_transitions: tuple[str, ...] = ()
    enforce_termination_guard: bool = True


@dataclass(frozen=True)
class SPOQPN:
    control_places: tuple[str, ...]
    quantum_registers: tuple[QuantumRegister, ...]
    transitions: tuple[Transition, ...]
    initial_marking: frozenset[str]
    initial_state: tuple[tuple[complex, ...], ...]
    attacker_interface: tuple[str, ...]
    secret_predicate: SecretPredicate
    observable_alphabet: tuple[str, ...] = ()
    symbolic_initial_state: tuple[tuple[str, object], ...] | None = None

    def register_dimensions(self) -> dict[str, int]:
        return {register.name: register.dimension for register in self.quantum_registers}

    def register_names(self) -> tuple[str, ...]:
        return tuple(register.name for register in self.quantum_registers)

    def full_dimension(self) -> int:
        return dimension_product([register.dimension for register in self.quantum_registers])

    def symbolic_initial(self) -> dict[str, object] | None:
        if self.symbolic_initial_state is None:
            return None
        return _thaw_object(self.symbolic_initial_state)

    def validate(self) -> None:
        place_set = set(self.control_places)
        if not self.initial_marking.issubset(place_set):
            raise ValueError("Initial marking contains places outside the control layer.")
        register_names = self.register_names()
        register_set = set(register_names)
        if len(register_set) != len(register_names):
            raise ValueError("Quantum register names must be unique.")
        if not set(self.attacker_interface).issubset(register_set):
            raise ValueError("Attacker interface contains unknown quantum registers.")
        if len(self.initial_state) != self.full_dimension():
            raise ValueError("Initial state size does not match the register dimensions.")
        if not is_square([list(row) for row in self.initial_state]):
            raise ValueError("Initial state must be square.")
        if not _is_hermitian([list(row) for row in self.initial_state]):
            raise ValueError("Initial state must be Hermitian.")
        if abs(trace([list(row) for row in self.initial_state]) - 1.0) > 1e-9:
            raise ValueError("Initial state must have unit trace.")
        dimensions = self.register_dimensions()
        for transition in self.transitions:
            if not transition.pre.issubset(place_set):
                raise ValueError(f"Transition {transition.name} has unknown preset places.")
            if not transition.post.issubset(place_set):
                raise ValueError(f"Transition {transition.name} has unknown postset places.")
            if len(transition.access) != len(set(transition.access)):
                raise ValueError(f"Transition {transition.name} repeats quantum accesses.")
            if not set(transition.access).issubset(register_set):
                raise ValueError(f"Transition {transition.name} accesses unknown registers.")
            if not transition.branches:
                raise ValueError(f"Transition {transition.name} must have at least one branch.")
            local_dim = dimension_product([dimensions[name] for name in transition.access])
            for branch in transition.branches:
                for operator in branch.kraus_matrices():
                    if len(operator) != local_dim or any(len(row) != local_dim for row in operator):
                        raise ValueError(
                            f"Branch {transition.name}/{branch.name} has an invalid Kraus shape."
                        )
                for source, allowed in branch.predecessor_guards:
                    if not allowed:
                        raise ValueError(
                            f"Branch {transition.name}/{branch.name} has an empty predecessor guard for {source}."
                        )
                    if ":" not in source:
                        raise ValueError(
                            f"Branch {transition.name}/{branch.name} uses malformed predecessor guard key {source!r}."
                        )
                    kind, name = source.split(":", 1)
                    if kind == "place":
                        if name not in place_set:
                            raise ValueError(
                                f"Branch {transition.name}/{branch.name} guards on unknown place {name}."
                            )
                    elif kind == "register":
                        if name not in register_set:
                            raise ValueError(
                                f"Branch {transition.name}/{branch.name} guards on unknown register {name}."
                            )
                    else:
                        raise ValueError(
                            f"Branch {transition.name}/{branch.name} uses unsupported predecessor guard kind {kind!r}."
                        )
            guard_groups: dict[tuple[tuple[str, tuple[str, ...]], ...], list[Branch]] = {}
            for branch in transition.branches:
                guard_groups.setdefault(branch.predecessor_guards, []).append(branch)
            for guard_key, guard_branches in guard_groups.items():
                completeness = [[0j for _ in range(local_dim)] for _ in range(local_dim)]
                for branch in guard_branches:
                    for operator in branch.kraus_matrices():
                        completeness = matrix_add(completeness, matrix_mul(dagger(operator), operator))
                if not _matrix_close(completeness, identity(local_dim), tolerance=1e-9):
                    guard_suffix = "" if not guard_key else f" under predecessor guards {dict(guard_key)!r}"
                    raise ValueError(
                        f"Transition {transition.name} branches do not sum to a trace-preserving instrument{guard_suffix}."
                    )


def build_density_from_pure_state(vector: Sequence[complex]) -> tuple[tuple[complex, ...], ...]:
    return tuple(tuple(entry for entry in row) for row in outer(list(vector)))


def _is_hermitian(matrix: Matrix, tolerance: float = 1e-9) -> bool:
    rows = len(matrix)
    for row in range(rows):
        for col in range(rows):
            if abs(matrix[row][col] - matrix[col][row].conjugate()) > tolerance:
                return False
    return True


def _matrix_close(left: Matrix, right: Matrix, tolerance: float = 1e-9) -> bool:
    rows = len(left)
    cols = len(left[0]) if rows else 0
    for row in range(rows):
        for col in range(cols):
            if abs(left[row][col] - right[row][col]) > tolerance:
                return False
    return True


def _freeze_object(value: object) -> object:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_object(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_object(item) for item in value)
    return value


def _normalize_predecessor_guards(
    predecessor_guards: dict[str, Sequence[str] | str] | None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not predecessor_guards:
        return ()
    normalized: list[tuple[str, tuple[str, ...]]] = []
    for source, allowed in sorted(predecessor_guards.items()):
        if isinstance(allowed, str):
            branches = (allowed,)
        else:
            branches = tuple(str(item) for item in allowed)
        normalized.append((str(source), tuple(sorted(set(branches)))))
    return tuple(normalized)


def _thaw_object(value: object) -> object:
    if isinstance(value, tuple):
        if value and all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {key: _thaw_object(item) for key, item in value}
        return [_thaw_object(item) for item in value]
    return value
