"""Exact dense-matrix backend for SPO-QPN exploration."""

from __future__ import annotations

from typing import Protocol

from .linear_algebra import Matrix, close_to_zero, dimension_product, lift_local_operator, matrix_add, matrix_mul, normalize_density, partial_trace, pretty_matrix, trace, trace_distance
from .model import Branch, SPOQPN, Transition


class Backend(Protocol):
    def initialize(self) -> Matrix:
        ...

    def update(self, state: Matrix, transition: Transition, branch: Branch) -> Matrix:
        ...

    def weight(self, state: Matrix) -> complex:
        ...

    def is_positive_weight(self, state: Matrix, tolerance: float = 1e-12) -> bool:
        ...

    def reduce_to_attacker(self, state: Matrix) -> Matrix:
        ...

    def zero_operator(self) -> Matrix:
        ...

    def add_operator(self, left: Matrix, right: Matrix) -> Matrix:
        ...

    def operator_trace_float(self, operator: Matrix) -> float:
        ...

    def normalize_operator(self, operator: Matrix) -> Matrix:
        ...

    def pretty_operator(self, operator: Matrix) -> list[list[str]]:
        ...

    def operator_to_matrix(self, operator: Matrix) -> Matrix:
        ...

    def leakage(self, secret_operator: Matrix | None, nonsecret_operator: Matrix | None, tolerance: float) -> dict[str, object]:
        ...


class DenseMatrixBackend:
    """Finite-dimensional CP-map propagation using dense matrices."""

    backend_name = "dense_matrix"

    def __init__(self, model: SPOQPN):
        self.model = model
        self.register_order = model.register_names()
        self.register_index = {name: index for index, name in enumerate(self.register_order)}
        self.global_dimensions = [model.register_dimensions()[name] for name in self.register_order]
        self.full_dim = dimension_product(self.global_dimensions)
        self.attacker_positions = tuple(self.register_index[name] for name in model.attacker_interface)
        self._compiled_kraus: dict[tuple[str, str], list[Matrix]] = {}
        self._compile_kraus()

    def initialize(self) -> Matrix:
        return [list(row) for row in self.model.initial_state]

    def update(self, state: Matrix, transition: Transition, branch: Branch) -> Matrix:
        out = [[0j for _ in range(self.full_dim)] for _ in range(self.full_dim)]
        for operator in self._compiled_kraus[(transition.name, branch.name)]:
            left = matrix_mul(operator, state)
            contribution = matrix_mul(left, _dagger_cached(operator))
            out = matrix_add(out, contribution)
        return out

    def weight(self, state: Matrix) -> complex:
        return trace(state)

    def is_positive_weight(self, state: Matrix, tolerance: float = 1e-12) -> bool:
        return not close_to_zero(trace(state), tolerance)

    def reduce_to_attacker(self, state: Matrix) -> Matrix:
        return partial_trace(state, self.attacker_positions, self.global_dimensions)

    def zero_operator(self) -> Matrix:
        attacker_dim = 1
        for position in self.attacker_positions:
            attacker_dim *= self.global_dimensions[position]
        return [[0j for _ in range(attacker_dim)] for _ in range(attacker_dim)]

    def add_operator(self, left: Matrix, right: Matrix) -> Matrix:
        return matrix_add(left, right)

    def operator_trace_float(self, operator: Matrix) -> float:
        return float(trace(operator).real)

    def normalize_operator(self, operator: Matrix) -> Matrix:
        return normalize_density(operator)

    def pretty_operator(self, operator: Matrix) -> list[list[str]]:
        return pretty_matrix(operator)

    def operator_to_matrix(self, operator: Matrix) -> Matrix:
        return operator

    def leakage(
        self,
        secret_operator: Matrix | None,
        nonsecret_operator: Matrix | None,
        tolerance: float,
    ) -> dict[str, object]:
        if secret_operator is None or close_to_zero(trace(secret_operator), tolerance):
            if nonsecret_operator is None or close_to_zero(trace(nonsecret_operator), tolerance):
                return {
                    "value_float": 0.0,
                    "value_exact": "0",
                    "secret_normalized": None,
                    "nonsecret_normalized": None,
                }
            return {
                "value_float": 0.0,
                "value_exact": "0",
                "secret_normalized": None,
                "nonsecret_normalized": normalize_density(nonsecret_operator),
            }
        if nonsecret_operator is None or close_to_zero(trace(nonsecret_operator), tolerance):
            return {
                "value_float": 1.0,
                "value_exact": "1",
                "secret_normalized": normalize_density(secret_operator),
                "nonsecret_normalized": None,
            }
        normalized_secret = normalize_density(secret_operator)
        normalized_nonsecret = normalize_density(nonsecret_operator)
        return {
            "value_float": trace_distance(normalized_secret, normalized_nonsecret),
            "value_exact": None,
            "secret_normalized": normalized_secret,
            "nonsecret_normalized": normalized_nonsecret,
        }

    def _compile_kraus(self) -> None:
        register_dimensions = self.model.register_dimensions()
        for transition in self.model.transitions:
            access_positions = tuple(self.register_index[name] for name in transition.access)
            access_dims = [register_dimensions[name] for name in transition.access]
            for branch in transition.branches:
                self._compiled_kraus[(transition.name, branch.name)] = [
                    lift_local_operator(operator, access_positions, access_dims, self.global_dimensions)
                    for operator in branch.kraus_matrices()
                ]


_DAGGER_CACHE: dict[tuple[tuple[complex, ...], ...], Matrix] = {}


def _dagger_cached(matrix: Matrix) -> Matrix:
    key = tuple(tuple(entry for entry in row) for row in matrix)
    cached = _DAGGER_CACHE.get(key)
    if cached is None:
        cached = [[entry.conjugate() for entry in column] for column in zip(*matrix)]
        _DAGGER_CACHE[key] = cached
    return cached
