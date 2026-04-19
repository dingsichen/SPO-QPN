"""Small dense linear-algebra helpers built only on the standard library."""

from __future__ import annotations

import ast
import cmath
import math
from itertools import product
from typing import Sequence

Matrix = list[list[complex]]
Vector = list[complex]


def parse_scalar(value: object) -> complex:
    if isinstance(value, complex):
        return value
    if isinstance(value, (int, float)):
        return complex(value)
    if not isinstance(value, str):
        raise TypeError(f"Unsupported scalar literal: {value!r}")
    return _safe_eval_complex(value)


def parse_matrix(data: Sequence[Sequence[object]]) -> Matrix:
    return [[parse_scalar(entry) for entry in row] for row in data]


def parse_vector(data: Sequence[object]) -> Vector:
    return [parse_scalar(entry) for entry in data]


def zeros(rows: int, cols: int) -> Matrix:
    return [[0j for _ in range(cols)] for _ in range(rows)]


def identity(dim: int) -> Matrix:
    out = zeros(dim, dim)
    for index in range(dim):
        out[index][index] = 1.0 + 0j
    return out


def copy_matrix(matrix: Matrix) -> Matrix:
    return [row[:] for row in matrix]


def matrix_add(left: Matrix, right: Matrix) -> Matrix:
    rows = len(left)
    cols = len(left[0]) if rows else 0
    out = zeros(rows, cols)
    for row in range(rows):
        for col in range(cols):
            out[row][col] = left[row][col] + right[row][col]
    return out


def matrix_sub(left: Matrix, right: Matrix) -> Matrix:
    rows = len(left)
    cols = len(left[0]) if rows else 0
    out = zeros(rows, cols)
    for row in range(rows):
        for col in range(cols):
            out[row][col] = left[row][col] - right[row][col]
    return out


def scalar_mul(scalar: complex, matrix: Matrix) -> Matrix:
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    out = zeros(rows, cols)
    for row in range(rows):
        for col in range(cols):
            out[row][col] = scalar * matrix[row][col]
    return out


def matrix_mul(left: Matrix, right: Matrix) -> Matrix:
    rows = len(left)
    inner = len(left[0]) if rows else 0
    cols = len(right[0]) if right else 0
    out = zeros(rows, cols)
    for row in range(rows):
        for mid in range(inner):
            coeff = left[row][mid]
            if coeff == 0:
                continue
            for col in range(cols):
                out[row][col] += coeff * right[mid][col]
    return out


def dagger(matrix: Matrix) -> Matrix:
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    out = zeros(cols, rows)
    for row in range(rows):
        for col in range(cols):
            out[col][row] = matrix[row][col].conjugate()
    return out


def trace(matrix: Matrix) -> complex:
    total = 0j
    for index in range(len(matrix)):
        total += matrix[index][index]
    return total


def kron(left: Matrix, right: Matrix) -> Matrix:
    left_rows = len(left)
    left_cols = len(left[0]) if left_rows else 0
    right_rows = len(right)
    right_cols = len(right[0]) if right_rows else 0
    out = zeros(left_rows * right_rows, left_cols * right_cols)
    for left_row in range(left_rows):
        for left_col in range(left_cols):
            coeff = left[left_row][left_col]
            if coeff == 0:
                continue
            for right_row in range(right_rows):
                for right_col in range(right_cols):
                    out[left_row * right_rows + right_row][left_col * right_cols + right_col] = (
                        coeff * right[right_row][right_col]
                    )
    return out


def tensor_all(factors: Sequence[Matrix]) -> Matrix:
    if not factors:
        return [[1.0 + 0j]]
    result = copy_matrix(factors[0])
    for factor in factors[1:]:
        result = kron(result, factor)
    return result


def outer(vector: Vector) -> Matrix:
    dim = len(vector)
    out = zeros(dim, dim)
    for row in range(dim):
        for col in range(dim):
            out[row][col] = vector[row] * vector[col].conjugate()
    return out


def is_square(matrix: Matrix) -> bool:
    return all(len(row) == len(matrix) for row in matrix)


def dimension_product(dimensions: Sequence[int]) -> int:
    out = 1
    for dim in dimensions:
        out *= dim
    return out


def basis_tuples(dimensions: Sequence[int]) -> list[tuple[int, ...]]:
    if not dimensions:
        return [()]
    return list(product(*(range(dim) for dim in dimensions)))


def ravel_index(digits: Sequence[int], dimensions: Sequence[int]) -> int:
    index = 0
    for digit, dim in zip(digits, dimensions):
        index = index * dim + digit
    return index


def projective_measurement_kraus(
    acting_dimensions: Sequence[int],
    measured_positions: Sequence[int],
) -> list[tuple[str, Matrix]]:
    full_dim = dimension_product(acting_dimensions)
    measured_dimensions = [acting_dimensions[pos] for pos in measured_positions]
    outcomes = basis_tuples(measured_dimensions)
    local_basis = basis_tuples(acting_dimensions)
    branches: list[tuple[str, Matrix]] = []
    for outcome in outcomes:
        projector = zeros(full_dim, full_dim)
        for basis_digits in local_basis:
            if tuple(basis_digits[pos] for pos in measured_positions) != outcome:
                continue
            basis_index = ravel_index(basis_digits, acting_dimensions)
            projector[basis_index][basis_index] = 1.0 + 0j
        branch_name = "".join(str(bit) for bit in outcome)
        branches.append((branch_name, projector))
    return branches


def lift_local_operator(
    local_operator: Matrix,
    subsystem_positions: Sequence[int],
    subsystem_dimensions: Sequence[int],
    global_dimensions: Sequence[int],
) -> Matrix:
    full_dim = dimension_product(global_dimensions)
    out = zeros(full_dim, full_dim)
    all_basis = basis_tuples(global_dimensions)
    position_set = set(subsystem_positions)
    for row_digits in all_basis:
        subsystem_row = tuple(row_digits[pos] for pos in subsystem_positions)
        rest_row = tuple(
            row_digits[pos] for pos in range(len(global_dimensions)) if pos not in position_set
        )
        local_row = ravel_index(subsystem_row, subsystem_dimensions)
        global_row = ravel_index(row_digits, global_dimensions)
        for col_digits in all_basis:
            rest_col = tuple(
                col_digits[pos] for pos in range(len(global_dimensions)) if pos not in position_set
            )
            if rest_row != rest_col:
                continue
            subsystem_col = tuple(col_digits[pos] for pos in subsystem_positions)
            local_col = ravel_index(subsystem_col, subsystem_dimensions)
            global_col = ravel_index(col_digits, global_dimensions)
            out[global_row][global_col] = local_operator[local_row][local_col]
    return out


def partial_trace(
    density: Matrix,
    kept_positions: Sequence[int],
    global_dimensions: Sequence[int],
) -> Matrix:
    kept_positions = tuple(kept_positions)
    kept_dimensions = [global_dimensions[pos] for pos in kept_positions]
    traced_positions = tuple(
        pos for pos in range(len(global_dimensions)) if pos not in set(kept_positions)
    )
    traced_dimensions = [global_dimensions[pos] for pos in traced_positions]
    reduced = zeros(
        dimension_product(kept_dimensions),
        dimension_product(kept_dimensions),
    )
    all_kept = basis_tuples(kept_dimensions)
    all_traced = basis_tuples(traced_dimensions)
    for kept_row in all_kept:
        for kept_col in all_kept:
            total = 0j
            for traced_basis in all_traced:
                full_row = _merge_subsystems(
                    kept_row,
                    traced_basis,
                    kept_positions,
                    traced_positions,
                    global_dimensions,
                )
                full_col = _merge_subsystems(
                    kept_col,
                    traced_basis,
                    kept_positions,
                    traced_positions,
                    global_dimensions,
                )
                total += density[
                    ravel_index(full_row, global_dimensions)
                ][
                    ravel_index(full_col, global_dimensions)
                ]
            reduced[ravel_index(kept_row, kept_dimensions)][
                ravel_index(kept_col, kept_dimensions)
            ] = total
    return reduced


def hermitian_eigenvalues(matrix: Matrix, max_iterations: int = 128) -> list[float]:
    current = copy_matrix(matrix)
    for _ in range(max_iterations):
        q, r = _qr_decompose(current)
        current = matrix_mul(r, q)
    return sorted(float(current[index][index].real) for index in range(len(current)))


def trace_distance(left: Matrix, right: Matrix) -> float:
    diff = matrix_sub(left, right)
    hermitian = scalar_mul(0.5, matrix_add(diff, dagger(diff)))
    eigenvalues = hermitian_eigenvalues(hermitian)
    return 0.5 * sum(abs(value) for value in eigenvalues)


def normalize_density(density: Matrix, tolerance: float = 1e-12) -> Matrix:
    weight = trace(density)
    if abs(weight) <= tolerance:
        raise ValueError("Cannot normalize a zero-mass operator.")
    return scalar_mul(1.0 / weight, density)


def close_to_zero(value: complex, tolerance: float = 1e-12) -> bool:
    return abs(value) <= tolerance


def pretty_scalar(value: complex, precision: int = 10) -> str:
    real = round(value.real, precision)
    imag = round(value.imag, precision)
    if abs(imag) <= 10 ** (-precision):
        return str(real)
    if abs(real) <= 10 ** (-precision):
        return f"{imag}j"
    sign = "+" if imag >= 0 else "-"
    return f"{real}{sign}{abs(imag)}j"


def pretty_matrix(matrix: Matrix, precision: int = 6) -> list[list[str]]:
    return [[pretty_scalar(entry, precision=precision) for entry in row] for row in matrix]


def _merge_subsystems(
    kept_digits: Sequence[int],
    traced_digits: Sequence[int],
    kept_positions: Sequence[int],
    traced_positions: Sequence[int],
    global_dimensions: Sequence[int],
) -> tuple[int, ...]:
    merged = [0 for _ in global_dimensions]
    for digit, position in zip(kept_digits, kept_positions):
        merged[position] = digit
    for digit, position in zip(traced_digits, traced_positions):
        merged[position] = digit
    return tuple(merged)


def _safe_eval_complex(expression: str) -> complex:
    tree = ast.parse(expression, mode="eval")
    return complex(_eval_ast(tree.body))


def _eval_ast(node: ast.AST) -> complex:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return complex(node.value)
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in {"i", "j"}:
            return 1j
        if node.id == "pi":
            return complex(math.pi)
        raise ValueError(f"Unsupported symbol: {node.id}")
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
        raise ValueError(f"Unsupported binary operator: {node.op!r}")
    if isinstance(node, ast.UnaryOp):
        value = _eval_ast(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise ValueError(f"Unsupported unary operator: {node.op!r}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are supported.")
        if node.func.id == "sqrt" and len(node.args) == 1:
            return cmath.sqrt(_eval_ast(node.args[0]))
        raise ValueError(f"Unsupported function: {node.func.id}")
    raise ValueError(f"Unsupported expression node: {ast.dump(node)}")


def _qr_decompose(matrix: Matrix) -> tuple[Matrix, Matrix]:
    n = len(matrix)
    columns = [[matrix[row][col] for row in range(n)] for col in range(n)]
    orthonormal: list[list[complex]] = []
    r = zeros(n, n)
    for col_index, column in enumerate(columns):
        vector = column[:]
        for basis_index, basis_vector in enumerate(orthonormal):
            coeff = _dot(basis_vector, column)
            r[basis_index][col_index] = coeff
            vector = [entry - coeff * basis_vector[row] for row, entry in enumerate(vector)]
        norm = math.sqrt(max(_dot(vector, vector).real, 0.0))
        if norm <= 1e-15:
            basis_vector = [0j for _ in range(n)]
        else:
            basis_vector = [entry / norm for entry in vector]
        orthonormal.append(basis_vector)
        r[col_index][col_index] = norm
    q = zeros(n, n)
    for col_index, basis_vector in enumerate(orthonormal):
        for row in range(n):
            q[row][col_index] = basis_vector[row]
    return q, r


def _dot(left: Sequence[complex], right: Sequence[complex]) -> complex:
    return sum(entry.conjugate() * other for entry, other in zip(left, right))
