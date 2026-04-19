# SPO-QPN

Reference implementation for current-state opacity verification in safe partially observed quantum Petri nets (SPO-QPNs).

This repository contains:
- the core SPO-QPN data structures,
- configuration-based verification of structural current-state opacity and posterior-state leakage,
- observation-pomset / true-concurrency semantics,
- dense and exact symbolic backends,
- the Section VII repeater case study,
- the scaling benchmark code,
- unit and integration tests.

## Assumptions

The implementation is intended for the model class studied in the paper. In particular, the analysis is used under the assumptions that:
- the control net is safe,
- the model is bounded,
- the model is divergence-free with respect to unobservable events.

## Repository Layout

```text
spo_qpn/         Core library
tests/           Regression and integration tests
examples/        Minimal example, repeater case, and benchmark scripts
```

Important files:
- `spo_qpn/analysis.py`
- `spo_qpn/interleaving.py`
- `spo_qpn/backend.py`
- `spo_qpn/exact_backend.py`
- `examples/repeater_case.py`
- `tests/test_spo_qpn.py`

## Installation

Recommended environment: Python 3.11+.

```bash
pip install -r requirements.txt
```

The verification core itself uses only the Python standard library. `numpy` and `matplotlib` are only needed for plotting scripts in `examples/`.

## Quick Start

Run the test suite:

```bash
python -m unittest -v tests.test_spo_qpn
```

Run the repeater case study:

```bash
python examples/repeater_case.py
```

Run the minimal JSON example:

```bash
python -m spo_qpn.cli examples/minimal_model.json --analysis examples/minimal_analysis.json
```

## Benchmark

Run the scaling benchmark:

```bash
python examples/benchmark_repeater_scaling.py
```

## License

MIT
