"""Reproducible runtime benchmark for the Section 7 scaling family."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from repeater_case import (
    build_repeater_section7_model,
    build_scaling_analysis,
    enumerate_scaling_words,
    interleaving_word_count,
)
from spo_qpn.analysis import analyze_model
from spo_qpn.backend import DenseMatrixBackend
from spo_qpn.exact_backend import ExactSymbolicBackend
from spo_qpn.interleaving import analyze_interleaving_sequences


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the repeater scaling family E_m.")
    parser.add_argument(
        "--m-values",
        nargs="+",
        type=int,
        default=[0, 4, 8, 12],
        help="Calibration-count values m to benchmark.",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Measured repetitions per approach.")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up repetitions per approach.")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    results = run_scaling_benchmark(
        m_values=args.m_values,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    rendered = json.dumps(results, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


def run_scaling_benchmark(
    m_values: list[int] | tuple[int, ...],
    repeats: int,
    warmup: int,
) -> dict[str, object]:
    results = {
        "benchmark": "repeater_scaling_family",
        "m_values": list(m_values),
        "repeats": repeats,
        "warmup": warmup,
        "approaches": [
            "interleaving_dense",
            "true_concurrency_dense",
            "true_concurrency_exact_symbolic",
        ],
        "entries": [],
    }

    for m in m_values:
        words = tuple(enumerate_scaling_words(m))
        analysis = build_scaling_analysis(m)
        interleaving = _measure(
            repeats=repeats,
            warmup=warmup,
            thunk=lambda: _run_interleaving(m, words, analysis),
        )
        quotient_dense = _measure(
            repeats=repeats,
            warmup=warmup,
            thunk=lambda: _run_true_concurrency(m, backend_name="dense"),
        )
        quotient_exact = _measure(
            repeats=repeats,
            warmup=warmup,
            thunk=lambda: _run_true_concurrency(m, backend_name="exact_symbolic"),
        )
        _validate_consistency(m, words, interleaving["report"], quotient_dense["report"], quotient_exact["report"])
        entry = {
            "m": m,
            "observable_interleavings": interleaving_word_count(m),
            "enumerated_target_words": len(words),
            "pomset_targets_after_quotient": quotient_dense["report"]["maximal_observation_count"],
            "structural_workload": {
                "reachable_linear_states": interleaving["report"]["reachable_linear_state_count"],
                "terminal_linear_executions": interleaving["report"]["terminal_execution_count"],
                "reachable_quotient_configurations": quotient_dense["report"]["reachable_configuration_count"],
            },
            "interleaving_dense": _summarize_measurement(interleaving),
            "true_concurrency_dense": _summarize_measurement(quotient_dense),
            "true_concurrency_exact_symbolic": _summarize_measurement(quotient_exact),
            "speedups": {
                "dense_quotient_vs_interleaving": (
                    interleaving["median_seconds"] / quotient_dense["median_seconds"]
                    if quotient_dense["median_seconds"] > 0
                    else None
                ),
                "exact_quotient_vs_interleaving": (
                    interleaving["median_seconds"] / quotient_exact["median_seconds"]
                    if quotient_exact["median_seconds"] > 0
                    else None
                ),
            },
            "compression_factors": {
                "word_to_pomset": (
                    interleaving_word_count(m) / quotient_dense["report"]["maximal_observation_count"]
                    if quotient_dense["report"]["maximal_observation_count"] > 0
                    else None
                ),
                "linear_state_to_configuration": (
                    interleaving["report"]["reachable_linear_state_count"]
                    / quotient_dense["report"]["reachable_configuration_count"]
                    if quotient_dense["report"]["reachable_configuration_count"] > 0
                    else None
                ),
            },
            "worst_case_leakage": {
                "interleaving_dense": interleaving["report"]["worst_case_leakage"],
                "true_concurrency_dense": quotient_dense["report"]["worst_case_leakage"],
                "true_concurrency_exact_symbolic": quotient_exact["report"]["worst_case_leakage"],
            },
            "exact_leakage": quotient_exact["report"]["observation_reports"][0]["leakage_exact"],
        }
        results["entries"].append(entry)
    return results


def _run_true_concurrency(m: int, backend_name: str) -> dict[str, object]:
    model = build_repeater_section7_model()
    analysis = build_scaling_analysis(m)
    if backend_name == "dense":
        backend = DenseMatrixBackend(model)
    else:
        backend = ExactSymbolicBackend(model)
    return analyze_model(model, analysis=analysis, backend=backend)


def _run_interleaving(
    m: int,
    words: tuple[tuple[str, ...], ...],
    analysis,
) -> dict[str, object]:
    del m
    model = build_repeater_section7_model()
    return analyze_interleaving_sequences(
        model,
        words,
        analysis=analysis,
        backend=DenseMatrixBackend(model),
    )


def _measure(
    repeats: int,
    warmup: int,
    thunk,
) -> dict[str, object]:
    report = None
    for _ in range(warmup):
        report = thunk()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        report = thunk()
        samples.append(time.perf_counter() - start)
    assert report is not None
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.fmean(samples),
        "report": report,
    }


def _summarize_measurement(measurement: dict[str, object]) -> dict[str, object]:
    report = measurement["report"]
    return {
        "median_seconds": measurement["median_seconds"],
        "mean_seconds": measurement["mean_seconds"],
        "samples_seconds": measurement["samples_seconds"],
        "backend": report["backend"],
        "worst_case_leakage": report["worst_case_leakage"],
    }


def _validate_consistency(
    m: int,
    words: tuple[tuple[str, ...], ...],
    interleaving_report: dict[str, object],
    quotient_dense_report: dict[str, object],
    quotient_exact_report: dict[str, object],
) -> None:
    if len(words) != interleaving_word_count(m):
        raise ValueError(f"Enumerated word count disagrees with closed-form count for m={m}.")
    dense_leakage = quotient_dense_report["worst_case_leakage"]
    exact_leakage = quotient_exact_report["worst_case_leakage"]
    baseline_leakage = interleaving_report["worst_case_leakage"]
    if abs(baseline_leakage - dense_leakage) > 1e-9 or abs(dense_leakage - exact_leakage) > 1e-9:
        raise ValueError(f"Benchmark approaches disagree on leakage for m={m}.")


if __name__ == "__main__":
    main()
