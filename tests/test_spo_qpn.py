"""Integration tests for the generic SPO-QPN verifier."""

from __future__ import annotations

import unittest

from examples.repeater_case import (
    build_repeater_case,
    build_scaling_analysis,
    build_repeater_section7_model,
    enumerate_scaling_words,
    interleaving_word_count,
)
from spo_qpn.analysis import analyze_model
from spo_qpn.exact_backend import ExactSymbolicBackend
from spo_qpn.interleaving import analyze_interleaving_sequences
from spo_qpn.model import AnalysisSpec, Branch, QuantumRegister, SPOQPN, SecretPredicate, Transition
from spo_qpn.pomset import canonical_pomset_from_spec


class SPOQPNAnalysisTests(unittest.TestCase):
    def test_repeater_section7_model_matches_published_architecture(self) -> None:
        model = build_repeater_section7_model()
        self.assertEqual(
            set(model.control_places),
            {"p0", "p1", "p2_nonsec", "p3_nonsec", "p2_sec", "p3_sec", "p_finish", "p_bg"},
        )
        self.assertEqual(model.initial_marking, frozenset({"p0", "p_bg"}))
        self.assertEqual(model.attacker_interface, ("qM",))
        self.assertEqual(model.observable_alphabet, ("req", "swap_ok", "done", "cal", "fail"))

        by_name = {transition.name: transition for transition in model.transitions}
        self.assertEqual(
            set(by_name),
            {
                "t_req",
                "t_swap_nonsec",
                "t_pur_sec",
                "t_ok_nonsec",
                "t_ok_sec",
                "t_done_nonsec",
                "t_done_sec",
                "t_reject",
                "t_cal",
                "t_reset",
            },
        )
        self.assertEqual(by_name["t_cal"].pre, frozenset({"p_bg"}))
        self.assertEqual(by_name["t_cal"].post, frozenset({"p_bg"}))
        self.assertEqual(by_name["t_reject"].pre, frozenset({"p2_sec"}))
        self.assertEqual(by_name["t_reject"].post, frozenset({"p_finish"}))
        self.assertEqual(by_name["t_reject"].branches[0].label, "fail")
        self.assertEqual(by_name["t_reset"].pre, frozenset({"p_finish"}))
        self.assertEqual(by_name["t_reset"].post, frozenset({"p0"}))
        self.assertEqual(by_name["t_reset"].branches[0].label, "tau")
        self.assertEqual(by_name["t_swap_nonsec"].access, ("q2", "q3"))
        self.assertEqual(by_name["t_pur_sec"].access, ("q2", "q3", "qM"))
        self.assertEqual(by_name["t_ok_nonsec"].access, ("q4",))
        self.assertEqual(by_name["t_ok_sec"].access, ("q4",))

    def test_repeater_leakage_matches_case_study(self) -> None:
        model, analysis = build_repeater_case()
        report = analyze_model(model, analysis)
        self.assertEqual(report["analysis_scope"], "targeted")
        self.assertTrue(report["structural_current_state_opaque"])
        self.assertAlmostEqual(report["worst_case_leakage"], 0.5, places=6)
        self.assertEqual(len(report["observation_reports"]), 1)
        target = report["observation_reports"][0]
        self.assertEqual(target["observation"]["one_linearization"], ["req", "swap_ok", "done"])
        self.assertAlmostEqual(target["leakage"], 0.5, places=6)

    def test_repeater_exact_backend_reports_exact_leakage(self) -> None:
        model, analysis = build_repeater_case()
        report = analyze_model(model, analysis, backend=ExactSymbolicBackend(model))
        self.assertAlmostEqual(report["worst_case_leakage"], 0.5, places=6)
        self.assertEqual(report["observation_reports"][0]["leakage_exact"], "1/2")

    def test_scaling_target_keeps_calibration_chain_concurrent_with_foreground(self) -> None:
        model = build_repeater_section7_model()
        report = analyze_model(model, build_scaling_analysis(1))
        observation = report["observation_reports"][0]["observation"]
        label_to_id = {event["label"]: event["id"] for event in observation["events"]}
        order_pairs = {tuple(pair) for pair in observation["order"]}
        self.assertIn((label_to_id["req"], label_to_id["swap_ok"]), order_pairs)
        self.assertIn((label_to_id["swap_ok"], label_to_id["done"]), order_pairs)
        self.assertNotIn((label_to_id["cal"], label_to_id["req"]), order_pairs)
        self.assertNotIn((label_to_id["req"], label_to_id["cal"]), order_pairs)
        self.assertNotIn((label_to_id["cal"], label_to_id["swap_ok"]), order_pairs)
        self.assertNotIn((label_to_id["swap_ok"], label_to_id["cal"]), order_pairs)
        self.assertNotIn((label_to_id["cal"], label_to_id["done"]), order_pairs)
        self.assertNotIn((label_to_id["done"], label_to_id["cal"]), order_pairs)

    def test_canonical_pomset_identifies_isomorphic_scaling_targets(self) -> None:
        left = {
            "events": ["req", "swap_ok", "done", "cal_0", "cal_1"],
            "labels": {
                "req": "req",
                "swap_ok": "swap_ok",
                "done": "done",
                "cal_0": "cal",
                "cal_1": "cal",
            },
            "order": [["req", "swap_ok"], ["swap_ok", "done"], ["cal_0", "cal_1"]],
        }
        right = {
            "events": ["u3", "u1", "u4", "u0", "u2"],
            "labels": {
                "u3": "cal",
                "u1": "done",
                "u4": "req",
                "u0": "cal",
                "u2": "swap_ok",
            },
            "order": [["u4", "u2"], ["u2", "u1"], ["u0", "u3"]],
        }
        self.assertEqual(canonical_pomset_from_spec(left), canonical_pomset_from_spec(right))

    def test_interleaving_word_enumerator_matches_closed_form(self) -> None:
        words = enumerate_scaling_words(3)
        self.assertEqual(len(words), interleaving_word_count(3))
        self.assertEqual(len(set(words)), len(words))

    def test_interleaving_baseline_agrees_with_true_concurrency_scaling_instance(self) -> None:
        model = build_repeater_section7_model()
        analysis = build_scaling_analysis(2)
        words = tuple(enumerate_scaling_words(2))
        baseline = analyze_interleaving_sequences(model, words, analysis=analysis)
        quotient = analyze_model(build_repeater_section7_model(), analysis)
        self.assertEqual(len(words), interleaving_word_count(2))
        self.assertAlmostEqual(baseline["worst_case_leakage"], 0.5, places=6)
        self.assertAlmostEqual(baseline["worst_case_leakage"], quotient["worst_case_leakage"], places=6)
        self.assertEqual(len(baseline["observation_reports"]), interleaving_word_count(2))

    def test_true_concurrency_quotients_independent_orders(self) -> None:
        model = SPOQPN(
            control_places=("pa0", "pa1", "pb0", "pb1"),
            quantum_registers=(QuantumRegister("qA"), QuantumRegister("qB")),
            transitions=(
                Transition(
                    name="t_a",
                    pre=frozenset({"pa0"}),
                    post=frozenset({"pa1"}),
                    access=("qA",),
                    branches=(Branch.from_matrices("main", "a", [[[1, 0], [0, 1]]]),),
                ),
                Transition(
                    name="t_b",
                    pre=frozenset({"pb0"}),
                    post=frozenset({"pb1"}),
                    access=("qB",),
                    branches=(Branch.from_matrices("main", "b", [[[1, 0], [0, 1]]]),),
                ),
            ),
            initial_marking=frozenset({"pa0", "pb0"}),
            initial_state=((1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
            attacker_interface=("qA",),
            secret_predicate=SecretPredicate.event_occurrence(["t_a"]),
            observable_alphabet=("a", "b"),
        )
        report = analyze_model(model, AnalysisSpec())
        self.assertEqual(report["reachable_configuration_count"], 4)
        concurrent = None
        for item in report["observation_reports"]:
            if sorted(item["observation"]["one_linearization"]) == ["a", "b"]:
                concurrent = item
                break
        self.assertIsNotNone(concurrent)
        self.assertEqual(concurrent["observation"]["order"], [])

    def test_predecessor_guards_realize_branch_conditioned_follow_up(self) -> None:
        identity_2 = [[1, 0], [0, 1]]
        projector_0 = [[1, 0], [0, 0]]
        projector_1 = [[0, 0], [0, 1]]
        pauli_x = [[0, 1], [1, 0]]
        model = SPOQPN(
            control_places=("p0", "p1", "p2"),
            quantum_registers=(QuantumRegister("q"),),
            transitions=(
                Transition(
                    name="t_measure",
                    pre=frozenset({"p0"}),
                    post=frozenset({"p1"}),
                    access=("q",),
                    branches=(
                        Branch.from_matrices("0", "tau", [projector_0]),
                        Branch.from_matrices("1", "tau", [projector_1]),
                    ),
                ),
                Transition(
                    name="t_follow",
                    pre=frozenset({"p1"}),
                    post=frozenset({"p2"}),
                    access=("q",),
                    branches=(
                        Branch.from_matrices(
                            "0",
                            "a",
                            [identity_2],
                            predecessor_guards={"place:p1": ["0"]},
                        ),
                        Branch.from_matrices(
                            "1",
                            "a",
                            [pauli_x],
                            predecessor_guards={"place:p1": ["1"]},
                        ),
                    ),
                ),
            ),
            initial_marking=frozenset({"p0"}),
            initial_state=((0.5, 0.5), (0.5, 0.5)),
            attacker_interface=("q",),
            secret_predicate=SecretPredicate.branch_occurrence(["t_measure/1"]),
            observable_alphabet=("a",),
        )
        report = analyze_model(model, AnalysisSpec())
        self.assertEqual(report["reachable_configuration_count"], 5)
        final_observation = None
        for item in report["observation_reports"]:
            if item["observation"]["one_linearization"] == ["a"]:
                final_observation = item
                break
        self.assertIsNotNone(final_observation)
        self.assertTrue(final_observation["structural_secret_reachable"])
        self.assertTrue(final_observation["structural_nonsecret_reachable"])

    def test_tau_cycle_requires_explicit_bound_or_cut(self) -> None:
        model = SPOQPN(
            control_places=("p0",),
            quantum_registers=(),
            transitions=(
                Transition(
                    name="t_tau",
                    pre=frozenset({"p0"}),
                    post=frozenset({"p0"}),
                    access=(),
                    branches=(Branch.from_matrices("main", "tau", [[[1]]]),),
                ),
            ),
            initial_marking=frozenset({"p0"}),
            initial_state=((1,),),
            attacker_interface=(),
            secret_predicate=SecretPredicate.event_occurrence(["t_tau"]),
            observable_alphabet=(),
        )
        with self.assertRaises(ValueError):
            analyze_model(model, AnalysisSpec())
        bounded = analyze_model(model, AnalysisSpec(max_total_events=2))
        self.assertEqual(bounded["reachable_configuration_count"], 3)

    def test_targeted_termination_guard_respects_branch_guards(self) -> None:
        identity_2 = [[1, 0], [0, 1]]
        model = SPOQPN(
            control_places=("p0",),
            quantum_registers=(QuantumRegister("q"),),
            transitions=(
                Transition(
                    name="t_mix",
                    pre=frozenset({"p0"}),
                    post=frozenset({"p0"}),
                    access=("q",),
                    branches=(
                        Branch.from_matrices("obs", "a", [identity_2]),
                        Branch.from_matrices(
                            "never",
                            "tau",
                            [identity_2],
                            predecessor_guards={"place:p0": ["impossible"]},
                        ),
                    ),
                ),
            ),
            initial_marking=frozenset({"p0"}),
            initial_state=((1, 0), (0, 0)),
            attacker_interface=("q",),
            secret_predicate=SecretPredicate.event_occurrence(["never"]),
            observable_alphabet=("a",),
        )
        report = analyze_model(model, AnalysisSpec(target_observations=({"sequence": ["a"]},)))
        self.assertEqual(report["analysis_scope"], "targeted")
        self.assertEqual(report["reachable_configuration_count"], 2)
        self.assertEqual(report["observation_reports"][0]["observation"]["one_linearization"], ["a"])

    def test_targeted_report_does_not_imply_global_opacity(self) -> None:
        model = SPOQPN(
            control_places=("p0", "p1", "p2_public", "p2_secret", "p_finish"),
            quantum_registers=(),
            transitions=(
                Transition(
                    name="t_req",
                    pre=frozenset({"p0"}),
                    post=frozenset({"p1"}),
                    access=(),
                    branches=(Branch.from_matrices("main", "req", [[[1]]]),),
                ),
                Transition(
                    name="t_public",
                    pre=frozenset({"p1"}),
                    post=frozenset({"p2_public"}),
                    access=(),
                    branches=(Branch.from_matrices("main", "tau", [[[1]]]),),
                ),
                Transition(
                    name="t_secret",
                    pre=frozenset({"p1"}),
                    post=frozenset({"p2_secret"}),
                    access=(),
                    branches=(Branch.from_matrices("main", "tau", [[[1]]]),),
                ),
                Transition(
                    name="t_done_public",
                    pre=frozenset({"p2_public"}),
                    post=frozenset({"p_finish"}),
                    access=(),
                    branches=(Branch.from_matrices("main", "done", [[[1]]]),),
                ),
                Transition(
                    name="t_done_secret",
                    pre=frozenset({"p2_secret"}),
                    post=frozenset({"p_finish"}),
                    access=(),
                    branches=(Branch.from_matrices("main", "done", [[[1]]]),),
                ),
                Transition(
                    name="t_fail_secret",
                    pre=frozenset({"p2_secret"}),
                    post=frozenset({"p_finish"}),
                    access=(),
                    branches=(Branch.from_matrices("main", "fail", [[[1]]]),),
                ),
            ),
            initial_marking=frozenset({"p0"}),
            initial_state=((1,),),
            attacker_interface=(),
            secret_predicate=SecretPredicate.event_occurrence(["t_secret"]),
            observable_alphabet=("req", "done", "fail"),
        )
        targeted = analyze_model(model, AnalysisSpec(target_observations=({"sequence": ["req", "done"]},)))
        self.assertTrue(targeted["structural_current_state_opaque"])
        global_report = analyze_model(
            model,
            AnalysisSpec(
                tau_label="tau",
                max_observable_events=2,
            ),
        )
        self.assertEqual(global_report["analysis_scope"], "global")
        self.assertFalse(global_report["structural_current_state_opaque"])
        secret_only = {
            tuple(item["observation"]["one_linearization"])
            for item in global_report["observation_reports"]
            if item["structural_secret_reachable"] and not item["structural_nonsecret_reachable"]
        }
        self.assertIn(("req", "fail"), secret_only)


if __name__ == "__main__":
    unittest.main()
