"""Exact symbolic backend for stabilizer-fragment SPO-QPN models."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import sqrt
from typing import Iterable

from .exact_types import ExactComplex, I, MINUS_ONE, MINUS_I, ONE, ZERO, sqrt_fraction_string
from .linear_algebra import Matrix, normalize_density, pretty_matrix, trace_distance
from .model import Branch, SPOQPN, Transition

ExactMatrix = list[list[ExactComplex]]
ExactOperator = dict[tuple[str, ...], ExactComplex]


@dataclass(frozen=True)
class PauliString:
    labels: tuple[str, ...]
    phase_exp: int = 0

    def multiply(self, other: "PauliString") -> "PauliString":
        phase = (self.phase_exp + other.phase_exp) % 4
        out: list[str] = []
        for left, right in zip(self.labels, other.labels):
            delta, label = _PAULI_PRODUCT[(left, right)]
            phase = (phase + delta) % 4
            out.append(label)
        return PauliString(tuple(out), phase)

    def commutes_with(self, other: "PauliString") -> bool:
        anticommutes = 0
        for left, right in zip(self.labels, other.labels):
            if left == "I" or right == "I" or left == right:
                continue
            anticommutes ^= 1
        return anticommutes == 0

    def with_sign(self, sign: int) -> "PauliString":
        return PauliString(self.labels, (self.phase_exp + (0 if sign > 0 else 2)) % 4)

    def exact_phase(self) -> ExactComplex:
        return _PHASES[self.phase_exp]

    def is_identity_outside(self, kept_positions: set[int]) -> bool:
        for index, label in enumerate(self.labels):
            if index in kept_positions:
                continue
            if label != "I":
                return False
        return True

    def restrict(self, kept_positions: list[int]) -> "PauliString":
        return PauliString(tuple(self.labels[index] for index in kept_positions), self.phase_exp)


@dataclass(frozen=True)
class WeightedStabilizerState:
    generators: tuple[PauliString, ...]
    weight: Fraction


class ExactSymbolicBackend:
    """Exact symbolic propagation over the stabilizer fragment."""

    backend_name = "exact_symbolic_stabilizer"

    def __init__(self, model: SPOQPN):
        self.model = model
        self.register_order = model.register_names()
        if any(register.dimension != 2 for register in model.quantum_registers):
            raise ValueError("The exact symbolic backend currently supports qubits only.")
        self.register_index = {name: index for index, name in enumerate(self.register_order)}
        self.attacker_positions = [self.register_index[name] for name in model.attacker_interface]
        self.num_qubits = len(self.register_order)
        self._validate_symbolic_specs()

    def initialize(self) -> WeightedStabilizerState:
        spec = self.model.symbolic_initial()
        if spec is None:
            raise ValueError("Exact symbolic backend requires model.symbolic_initial_state.")
        state = WeightedStabilizerState(
            generators=tuple(
                PauliString(tuple("Z" if idx == qubit else "I" for idx in range(self.num_qubits)))
                for qubit in range(self.num_qubits)
            ),
            weight=Fraction(1, 1),
        )
        return self._apply_symbolic_state_preparation(state, spec)

    def update(
        self,
        state: WeightedStabilizerState,
        transition: Transition,
        branch: Branch,
    ) -> WeightedStabilizerState:
        spec = branch.symbolic()
        if spec is None:
            raise ValueError(
                f"Branch {transition.name}/{branch.name} has no symbolic specification for exact execution."
            )
        ops = spec.get("ops", [])
        current = state
        for op in ops:
            current = self._apply_op(current, op)
            if current.weight == 0:
                break
        return current

    def weight(self, state: WeightedStabilizerState) -> Fraction:
        return state.weight

    def is_positive_weight(self, state: WeightedStabilizerState, tolerance: float = 0.0) -> bool:
        del tolerance
        return state.weight > 0

    def zero_operator(self) -> ExactOperator:
        return {}

    def add_operator(self, left: ExactOperator, right: ExactOperator) -> ExactOperator:
        out = dict(left)
        for key, value in right.items():
            out[key] = out.get(key, ZERO) + value
            if out[key].is_zero():
                out.pop(key)
        return out

    def reduce_to_attacker(self, state: WeightedStabilizerState) -> ExactOperator:
        if state.weight == 0:
            return {}
        kept = self.attacker_positions
        kept_set = set(kept)
        factor = Fraction(1, 2 ** len(kept))
        expansion: ExactOperator = {}
        for element in _enumerate_stabilizer_group(state.generators):
            if not element.is_identity_outside(kept_set):
                continue
            restricted = element.restrict(kept)
            coefficient = restricted.exact_phase().scale(state.weight * factor)
            key = restricted.labels
            expansion[key] = expansion.get(key, ZERO) + coefficient
            if expansion[key].is_zero():
                expansion.pop(key)
        return expansion

    def operator_trace_float(self, operator: ExactOperator) -> float:
        trace_value = self._operator_trace_exact(operator)
        return float(trace_value)

    def normalize_operator(self, operator: ExactOperator) -> ExactOperator:
        weight = self._operator_trace_exact(operator)
        if weight == 0:
            raise ValueError("Cannot normalize a zero operator.")
        return {key: value.divide_fraction(weight) for key, value in operator.items()}

    def pretty_operator(self, operator: ExactOperator) -> list[list[str]]:
        return [[entry.pretty() for entry in row] for row in self.operator_to_exact_matrix(operator)]

    def operator_to_matrix(self, operator: ExactOperator) -> Matrix:
        exact_matrix = self.operator_to_exact_matrix(operator)
        return [[entry.to_complex() for entry in row] for row in exact_matrix]

    def leakage(
        self,
        secret_operator: ExactOperator | None,
        nonsecret_operator: ExactOperator | None,
        tolerance: float,
    ) -> dict[str, object]:
        del tolerance
        secret_weight = Fraction(0, 1) if secret_operator is None else self._operator_trace_exact(secret_operator)
        nonsecret_weight = Fraction(0, 1) if nonsecret_operator is None else self._operator_trace_exact(nonsecret_operator)
        if secret_weight == 0 and nonsecret_weight == 0:
            return {
                "value_float": 0.0,
                "value_exact": "0",
                "secret_normalized": None,
                "nonsecret_normalized": None,
            }
        if secret_weight == 0:
            return {
                "value_float": 0.0,
                "value_exact": "0",
                "secret_normalized": None,
                "nonsecret_normalized": self.normalize_operator(nonsecret_operator or {}),
            }
        if nonsecret_weight == 0:
            return {
                "value_float": 1.0,
                "value_exact": "1",
                "secret_normalized": self.normalize_operator(secret_operator or {}),
                "nonsecret_normalized": None,
            }
        normalized_secret = self.normalize_operator(secret_operator or {})
        normalized_nonsecret = self.normalize_operator(nonsecret_operator or {})
        if len(self.attacker_positions) == 1:
            exact_value, float_value = self._single_qubit_trace_distance(normalized_secret, normalized_nonsecret)
            return {
                "value_float": float_value,
                "value_exact": exact_value,
                "secret_normalized": normalized_secret,
                "nonsecret_normalized": normalized_nonsecret,
            }
        dense_secret = self.operator_to_matrix(normalized_secret)
        dense_nonsecret = self.operator_to_matrix(normalized_nonsecret)
        return {
            "value_float": trace_distance(dense_secret, dense_nonsecret),
            "value_exact": None,
            "secret_normalized": normalized_secret,
            "nonsecret_normalized": normalized_nonsecret,
        }

    def _validate_symbolic_specs(self) -> None:
        if self.model.symbolic_initial() is None:
            raise ValueError("Exact symbolic backend requires a symbolic initial-state specification.")
        for transition in self.model.transitions:
            for branch in transition.branches:
                if branch.symbolic() is None:
                    raise ValueError(
                        f"Transition {transition.name} branch {branch.name} is missing symbolic ops."
                    )

    def _apply_symbolic_state_preparation(
        self,
        state: WeightedStabilizerState,
        spec: dict[str, object],
    ) -> WeightedStabilizerState:
        if spec.get("kind") != "stabilizer_state":
            raise ValueError("Unsupported symbolic initial state kind.")
        current = state
        for op in spec.get("ops", []):
            current = self._apply_op(current, op)
        return current

    def _apply_op(
        self,
        state: WeightedStabilizerState,
        op: dict[str, object],
    ) -> WeightedStabilizerState:
        kind = op["kind"]
        if kind in {"h", "s", "x", "y", "z"}:
            qubit = self.register_index[op["qubit"]]
            return WeightedStabilizerState(
                generators=tuple(_apply_single_qubit_gate(generator, kind, qubit) for generator in state.generators),
                weight=state.weight,
            )
        if kind == "cnot":
            control = self.register_index[op["control"]]
            target = self.register_index[op["target"]]
            return WeightedStabilizerState(
                generators=tuple(_apply_cnot(generator, control, target) for generator in state.generators),
                weight=state.weight,
            )
        if kind == "measure":
            qubit = self.register_index[op["qubit"]]
            pauli = _single_qubit_pauli(self.num_qubits, qubit, str(op.get("basis", "Z")).upper())
            desired_sign = 1 if int(op["outcome"]) == 0 else -1
            return _measure_pauli(state, pauli, desired_sign)
        if kind == "replace_state":
            replacement = self.model.symbolic_initial() if op.get("state") == "model_initial" else op.get("state")
            if replacement is None:
                raise ValueError("replace_state requires a symbolic stabilizer state.")
            return WeightedStabilizerState(
                generators=self._apply_symbolic_state_preparation(
                    WeightedStabilizerState(
                        generators=tuple(
                            PauliString(tuple("Z" if idx == qubit else "I" for idx in range(self.num_qubits)))
                            for qubit in range(self.num_qubits)
                        ),
                        weight=Fraction(1, 1),
                    ),
                    replacement,
                ).generators,
                weight=state.weight,
            )
        raise ValueError(f"Unsupported symbolic op kind: {kind}")

    def _operator_trace_exact(self, operator: ExactOperator) -> Fraction:
        identity_key = tuple("I" for _ in self.attacker_positions)
        coefficient = operator.get(identity_key, ZERO)
        if coefficient.imag != 0:
            raise ValueError("Operator trace is unexpectedly complex.")
        return coefficient.real * (2 ** len(self.attacker_positions))

    def operator_to_exact_matrix(self, operator: ExactOperator) -> ExactMatrix:
        dim = 2 ** len(self.attacker_positions)
        out = [[ZERO for _ in range(dim)] for _ in range(dim)]
        for labels, coefficient in operator.items():
            local = _pauli_labels_to_exact_matrix(labels)
            for row in range(dim):
                for col in range(dim):
                    out[row][col] = out[row][col] + (coefficient * local[row][col])
        return out

    def _single_qubit_trace_distance(
        self,
        secret_operator: ExactOperator,
        nonsecret_operator: ExactOperator,
    ) -> tuple[str, float]:
        secret = self.operator_to_exact_matrix(secret_operator)
        nonsecret = self.operator_to_exact_matrix(nonsecret_operator)
        delta00 = secret[0][0] - nonsecret[0][0]
        delta01 = secret[0][1] - nonsecret[0][1]
        radicand = delta00.abs_squared() + delta01.abs_squared()
        return sqrt_fraction_string(radicand), sqrt(float(radicand))


def _apply_single_qubit_gate(generator: PauliString, gate: str, qubit: int) -> PauliString:
    phase_delta, new_label = _SINGLE_QUBIT_CONJUGATION[gate][generator.labels[qubit]]
    labels = list(generator.labels)
    labels[qubit] = new_label
    return PauliString(tuple(labels), (generator.phase_exp + phase_delta) % 4)


def _apply_cnot(generator: PauliString, control: int, target: int) -> PauliString:
    left_pair = _CNOT_CONTROL_IMAGE[generator.labels[control]]
    right_pair = _CNOT_TARGET_IMAGE[generator.labels[target]]
    pair = left_pair.multiply(right_pair)
    labels = list(generator.labels)
    labels[control] = pair.labels[0]
    labels[target] = pair.labels[1]
    return PauliString(tuple(labels), (generator.phase_exp + pair.phase_exp) % 4)


def _measure_pauli(
    state: WeightedStabilizerState,
    pauli: PauliString,
    desired_sign: int,
) -> WeightedStabilizerState:
    generators = [PauliString(generator.labels, generator.phase_exp) for generator in state.generators]
    anticommuting = [index for index, generator in enumerate(generators) if not generator.commutes_with(pauli)]
    if not anticommuting:
        sign = _membership_sign(tuple(generators), pauli)
        if sign is None:
            raise ValueError("Failed to determine deterministic measurement outcome.")
        if sign != desired_sign:
            return WeightedStabilizerState(tuple(generators), Fraction(0, 1))
        return WeightedStabilizerState(tuple(generators), state.weight)
    pivot_index = anticommuting[0]
    pivot = generators[pivot_index]
    for index in anticommuting[1:]:
        generators[index] = generators[index].multiply(pivot)
    generators[pivot_index] = pauli.with_sign(desired_sign)
    return WeightedStabilizerState(tuple(generators), state.weight / 2)


@lru_cache(maxsize=None)
def _enumerate_stabilizer_group(generators: tuple[PauliString, ...]) -> tuple[PauliString, ...]:
    group = [PauliString(tuple("I" for _ in generators[0].labels), 0)] if generators else [PauliString((), 0)]
    for generator in generators:
        group += [element.multiply(generator) for element in list(group)]
    return tuple(group)


def _membership_sign(generators: tuple[PauliString, ...], pauli: PauliString) -> int | None:
    for element in _enumerate_stabilizer_group(generators):
        if element.labels != pauli.labels:
            continue
        phase_delta = (element.phase_exp - pauli.phase_exp) % 4
        if phase_delta == 0:
            return 1
        if phase_delta == 2:
            return -1
    return None


def _single_qubit_pauli(num_qubits: int, qubit: int, basis: str) -> PauliString:
    labels = ["I" for _ in range(num_qubits)]
    labels[qubit] = basis
    return PauliString(tuple(labels), 0)


def _pauli_labels_to_exact_matrix(labels: tuple[str, ...]) -> ExactMatrix:
    result = [[ONE]]
    for label in labels:
        result = _kron_exact(result, _EXACT_PAULI_MATRICES[label])
    return result


def _kron_exact(left: ExactMatrix, right: ExactMatrix) -> ExactMatrix:
    left_rows = len(left)
    left_cols = len(left[0]) if left_rows else 0
    right_rows = len(right)
    right_cols = len(right[0]) if right_rows else 0
    out = [[ZERO for _ in range(left_cols * right_cols)] for _ in range(left_rows * right_rows)]
    for left_row in range(left_rows):
        for left_col in range(left_cols):
            coeff = left[left_row][left_col]
            if coeff.is_zero():
                continue
            for right_row in range(right_rows):
                for right_col in range(right_cols):
                    out[left_row * right_rows + right_row][left_col * right_cols + right_col] = (
                        coeff * right[right_row][right_col]
                    )
    return out


_PHASES = {
    0: ONE,
    1: I,
    2: MINUS_ONE,
    3: MINUS_I,
}


_PAULI_PRODUCT = {
    ("I", "I"): (0, "I"),
    ("I", "X"): (0, "X"),
    ("I", "Y"): (0, "Y"),
    ("I", "Z"): (0, "Z"),
    ("X", "I"): (0, "X"),
    ("Y", "I"): (0, "Y"),
    ("Z", "I"): (0, "Z"),
    ("X", "X"): (0, "I"),
    ("Y", "Y"): (0, "I"),
    ("Z", "Z"): (0, "I"),
    ("X", "Y"): (1, "Z"),
    ("Y", "X"): (3, "Z"),
    ("X", "Z"): (3, "Y"),
    ("Z", "X"): (1, "Y"),
    ("Y", "Z"): (1, "X"),
    ("Z", "Y"): (3, "X"),
}


_SINGLE_QUBIT_CONJUGATION = {
    "h": {
        "I": (0, "I"),
        "X": (0, "Z"),
        "Y": (2, "Y"),
        "Z": (0, "X"),
    },
    "s": {
        "I": (0, "I"),
        "X": (0, "Y"),
        "Y": (2, "X"),
        "Z": (0, "Z"),
    },
    "x": {
        "I": (0, "I"),
        "X": (0, "X"),
        "Y": (2, "Y"),
        "Z": (2, "Z"),
    },
    "y": {
        "I": (0, "I"),
        "X": (2, "X"),
        "Y": (0, "Y"),
        "Z": (2, "Z"),
    },
    "z": {
        "I": (0, "I"),
        "X": (2, "X"),
        "Y": (2, "Y"),
        "Z": (0, "Z"),
    },
}


_CNOT_CONTROL_IMAGE = {
    "I": PauliString(("I", "I"), 0),
    "X": PauliString(("X", "X"), 0),
    "Y": PauliString(("Y", "X"), 0),
    "Z": PauliString(("Z", "I"), 0),
}

_CNOT_TARGET_IMAGE = {
    "I": PauliString(("I", "I"), 0),
    "X": PauliString(("I", "X"), 0),
    "Y": PauliString(("Z", "Y"), 0),
    "Z": PauliString(("Z", "Z"), 0),
}


_EXACT_PAULI_MATRICES = {
    "I": [[ONE, ZERO], [ZERO, ONE]],
    "X": [[ZERO, ONE], [ONE, ZERO]],
    "Y": [[ZERO, MINUS_I], [I, ZERO]],
    "Z": [[ONE, ZERO], [ZERO, MINUS_ONE]],
}
