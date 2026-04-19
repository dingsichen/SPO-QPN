"""Command-line interface for the SPO-QPN verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze_model
from .backend import DenseMatrixBackend
from .exact_backend import ExactSymbolicBackend
from .io import load_analysis_from_json, load_model_from_json
from .model import AnalysisSpec


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze current-state opacity for an SPO-QPN.")
    parser.add_argument("model", help="Path to the JSON model specification.")
    parser.add_argument("--analysis", help="Optional JSON analysis-spec file.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument(
        "--backend",
        choices=("dense", "exact_symbolic"),
        default="dense",
        help="Backend to use for quantum propagation.",
    )
    args = parser.parse_args()
    model = load_model_from_json(args.model)
    analysis = load_analysis_from_json(args.analysis) if args.analysis else AnalysisSpec()
    backend = DenseMatrixBackend(model) if args.backend == "dense" else ExactSymbolicBackend(model)
    report = analyze_model(model, analysis=analysis, backend=backend)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
