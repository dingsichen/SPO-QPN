"""Section 7 repeater-service case study for the SPO-QPN verifier."""

from __future__ import annotations

import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from spo_qpn.analysis import analyze_model
from spo_qpn.backend import DenseMatrixBackend
from spo_qpn.exact_backend import ExactSymbolicBackend
from spo_qpn.linear_algebra import Matrix, identity, kron, matrix_mul, outer, projective_measurement_kraus
from spo_qpn.model import AnalysisSpec, Branch, QuantumRegister, SPOQPN, SecretPredicate, Transition


def build_repeater_case() -> tuple[SPOQPN, AnalysisSpec]:
    """Return the full Section 7 model together with the base completed-round target.

    The model itself includes the calibration loop, the observable reject transition,
    and the hidden reset transition. The paired analysis simply selects the successful
    completed-round observation used for the base leakage derivation.
    """
    model = build_repeater_section7_model()
    analysis = build_completed_round_analysis()
    return model, analysis


def build_repeater_section7_model() -> SPOQPN:
    """Build the complete repeater-service model used in Section 7.

    This includes:
    - the non-secret foreground lane,
    - the secret purification lane,
    - branch-conditioned feed-forward,
    - the observable reject transition,
    - the concurrent calibration loop, and
    - the unobservable reset transition.
    """
    registers = tuple(QuantumRegister(name=name, dimension=2) for name in ["q1", "q2", "q3", "q4", "qM"])
    bell = [1 / (2 ** 0.5), 0, 0, 1 / (2 ** 0.5)]
    zero = [1, 0]
    psi0 = _tensor_vector([bell, bell, zero])
    rho0 = outer(psi0)
    symbolic_initial = _freeze(
        {
            "kind": "stabilizer_state",
            "ops": [
                {"kind": "h", "qubit": "q1"},
                {"kind": "cnot", "control": "q1", "target": "q2"},
                {"kind": "h", "qubit": "q3"},
                {"kind": "cnot", "control": "q3", "target": "q4"},
            ],
        }
    )
    transitions = [
        Transition(
            name="t_req",
            pre=frozenset({"p0"}),
            post=frozenset({"p1"}),
            access=(),
            branches=(
                Branch.from_matrices("main", "req", [[[1]]], symbolic_spec={"ops": []}),
            ),
        ),
        Transition(
            name="t_swap_nonsec",
            pre=frozenset({"p1"}),
            post=frozenset({"p2_nonsec"}),
            access=("q2", "q3"),
            branches=tuple(_nonsecret_bsm_branches()),
        ),
        Transition(
            name="t_pur_sec",
            pre=frozenset({"p1"}),
            post=frozenset({"p2_sec"}),
            access=("q2", "q3", "qM"),
            branches=tuple(_secret_purification_bsm_branches()),
        ),
        Transition(
            name="t_ok_nonsec",
            pre=frozenset({"p2_nonsec"}),
            post=frozenset({"p3_nonsec"}),
            access=("q4",),
            branches=tuple(_feedforward_branches("p2_nonsec")),
        ),
        Transition(
            name="t_ok_sec",
            pre=frozenset({"p2_sec"}),
            post=frozenset({"p3_sec"}),
            access=("q4",),
            branches=tuple(_feedforward_branches("p2_sec")),
        ),
        Transition(
            name="t_done_nonsec",
            pre=frozenset({"p3_nonsec"}),
            post=frozenset({"p_finish"}),
            access=(),
            branches=(
                Branch.from_matrices("main", "done", [[[1]]], symbolic_spec={"ops": []}),
            ),
        ),
        Transition(
            name="t_done_sec",
            pre=frozenset({"p3_sec"}),
            post=frozenset({"p_finish"}),
            access=(),
            branches=(
                Branch.from_matrices("main", "done", [[[1]]], symbolic_spec={"ops": []}),
            ),
        ),
        Transition(
            name="t_reject",
            pre=frozenset({"p2_sec"}),
            post=frozenset({"p_finish"}),
            access=(),
            branches=(
                Branch.from_matrices("main", "fail", [[[1]]], symbolic_spec={"ops": []}),
            ),
        ),
        Transition(
            name="t_cal",
            pre=frozenset({"p_bg"}),
            post=frozenset({"p_bg"}),
            access=(),
            branches=(
                Branch.from_matrices("main", "cal", [[[1]]], symbolic_spec={"ops": []}),
            ),
        ),
        Transition(
            name="t_reset",
            pre=frozenset({"p_finish"}),
            post=frozenset({"p0"}),
            access=("q1", "q2", "q3", "q4", "qM"),
            branches=(
                Branch.from_matrices(
                    "main",
                    "tau",
                    _reset_kraus_operators(psi0),
                    symbolic_spec={"ops": [{"kind": "replace_state", "state": "model_initial"}]},
                ),
            ),
        ),
    ]
    return SPOQPN(
        control_places=("p0", "p1", "p2_nonsec", "p3_nonsec", "p2_sec", "p3_sec", "p_finish", "p_bg"),
        quantum_registers=registers,
        transitions=tuple(transitions),
        initial_marking=frozenset({"p0", "p_bg"}),
        initial_state=tuple(tuple(entry for entry in row) for row in rho0),
        attacker_interface=("qM",),
        secret_predicate=SecretPredicate.event_occurrence(["t_pur_sec"]),
        observable_alphabet=("req", "swap_ok", "done", "cal", "fail"),
        symbolic_initial_state=symbolic_initial,
    )


def build_completed_round_analysis() -> AnalysisSpec:
    """Base analysis target: the successful completed-round foreground observation."""
    return AnalysisSpec(
        epsilon=0.0,
        target_observations=(_foreground_success_target(),),
        tau_label="tau",
        forbidden_transitions=("t_reset",),
    )


def build_scaling_analysis(m: int) -> AnalysisSpec:
    """Scaling target for E_m: successful foreground chain plus an m-event calibration chain."""
    return AnalysisSpec(
        epsilon=0.0,
        target_observations=(_scaling_target(m),),
        tau_label="tau",
        forbidden_transitions=("t_reset",),
    )


def interleaving_word_count(m: int) -> int:
    return math.comb(m + 3, 3)


def enumerate_scaling_words(m: int) -> list[tuple[str, ...]]:
    """Enumerate all observable shuffles of m calibration events with the successful foreground chain."""
    foreground = ("req", "swap_ok", "done")
    out: list[tuple[str, ...]] = []

    def build(prefix: list[str], remaining_cal: int, foreground_index: int) -> None:
        if remaining_cal == 0 and foreground_index == len(foreground):
            out.append(tuple(prefix))
            return
        if remaining_cal > 0:
            prefix.append("cal")
            build(prefix, remaining_cal - 1, foreground_index)
            prefix.pop()
        if foreground_index < len(foreground):
            prefix.append(foreground[foreground_index])
            build(prefix, remaining_cal, foreground_index + 1)
            prefix.pop()

    build([], m, 0)
    return out


def run_base_report(backend_name: str = "dense") -> dict[str, object]:
    model, analysis = build_repeater_case()
    backend = DenseMatrixBackend(model) if backend_name == "dense" else ExactSymbolicBackend(model)
    return analyze_model(model, analysis, backend=backend)


def main() -> None:
    dense_report = run_base_report("dense")
    exact_report = run_base_report("exact")
    print(
        json.dumps(
            {
                "dense": dense_report,
                "exact_symbolic": exact_report,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _foreground_success_target() -> dict[str, object]:
    return {
        "events": ["req", "swap_ok", "done"],
        "labels": {"req": "req", "swap_ok": "swap_ok", "done": "done"},
        "order": [["req", "swap_ok"], ["swap_ok", "done"]],
    }


def _scaling_target(m: int) -> dict[str, object]:
    events = ["req", "swap_ok", "done"] + [f"cal_{index}" for index in range(m)]
    labels = {"req": "req", "swap_ok": "swap_ok", "done": "done"}
    order = [["req", "swap_ok"], ["swap_ok", "done"]]
    for index in range(m):
        event_id = f"cal_{index}"
        labels[event_id] = "cal"
        if index > 0:
            order.append([f"cal_{index - 1}", event_id])
    return {"events": events, "labels": labels, "order": order}


def _nonsecret_bsm_branches():
    bsm = _bsm_unitary()
    for name, projector in projective_measurement_kraus((2, 2), (0, 1)):
        operator = matrix_mul(projector, bsm)
        yield Branch.from_matrices(
            name=name,
            label="tau",
            kraus=[operator],
            symbolic_spec={"ops": _measurement_ops(secret=False, outcome=name)},
        )


def _secret_purification_bsm_branches():
    bsm = kron(_bsm_unitary(), identity(2))
    hidden = _cnot_control_target(num_qubits=3, control=0, target=2)
    full_pre = matrix_mul(bsm, hidden)
    for name, projector in projective_measurement_kraus((2, 2, 2), (0, 1)):
        operator = matrix_mul(projector, full_pre)
        yield Branch.from_matrices(
            name=name,
            label="tau",
            kraus=[operator],
            symbolic_spec={"ops": _measurement_ops(secret=True, outcome=name)},
        )


def _feedforward_branches(upstream_place: str):
    for outcome in ("00", "01", "10", "11"):
        yield Branch.from_matrices(
            name=outcome,
            label="swap_ok",
            kraus=[_feedforward_correction(outcome)],
            symbolic_spec={"ops": _feedforward_ops(outcome)},
            predecessor_guards={f"place:{upstream_place}": [outcome]},
        )


def _measurement_ops(secret: bool, outcome: str) -> list[dict[str, object]]:
    ops: list[dict[str, object]] = []
    if secret:
        ops.append({"kind": "cnot", "control": "q2", "target": "qM"})
    ops.extend(
        [
            {"kind": "cnot", "control": "q2", "target": "q3"},
            {"kind": "h", "qubit": "q2"},
            {"kind": "measure", "basis": "Z", "qubit": "q2", "outcome": int(outcome[0])},
            {"kind": "measure", "basis": "Z", "qubit": "q3", "outcome": int(outcome[1])},
        ]
    )
    return ops


def _feedforward_ops(outcome: str) -> list[dict[str, object]]:
    ops: list[dict[str, object]] = []
    if outcome[1] == "1":
        ops.append({"kind": "x", "qubit": "q4"})
    if outcome[0] == "1":
        ops.append({"kind": "z", "qubit": "q4"})
    return ops


def _bsm_unitary() -> Matrix:
    return matrix_mul(kron(_hadamard(), identity(2)), _cnot_control_target(num_qubits=2, control=0, target=1))


def _feedforward_correction(outcome: str) -> Matrix:
    x = [[0, 1], [1, 0]]
    z = [[1, 0], [0, -1]]
    if outcome == "00":
        return identity(2)
    if outcome == "01":
        return x
    if outcome == "10":
        return z
    return matrix_mul(x, z)


def _hadamard() -> Matrix:
    scale = 1 / (2 ** 0.5)
    return [[scale, scale], [scale, -scale]]


def _cnot_control_target(num_qubits: int, control: int, target: int) -> Matrix:
    dim = 2 ** num_qubits
    matrix = [[0j for _ in range(dim)] for _ in range(dim)]
    for basis in range(dim):
        bits = [(basis >> shift) & 1 for shift in reversed(range(num_qubits))]
        next_bits = bits[:]
        if bits[control] == 1:
            next_bits[target] ^= 1
        output = 0
        for bit in next_bits:
            output = (output << 1) | bit
        matrix[output][basis] = 1
    return matrix


def _tensor_vector(vectors: list[list[complex]]) -> list[complex]:
    out = [1 + 0j]
    for vector in vectors:
        out = [left * right for left in out for right in vector]
    return out


def _reset_kraus_operators(psi0: list[complex]) -> list[Matrix]:
    dim = len(psi0)
    operators: list[Matrix] = []
    for basis_index in range(dim):
        operator = [[0j for _ in range(dim)] for _ in range(dim)]
        for row in range(dim):
            operator[row][basis_index] = psi0[row]
        operators.append(operator)
    return operators


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


if __name__ == "__main__":
    main()
