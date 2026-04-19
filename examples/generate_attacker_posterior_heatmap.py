"""Generate Fig. 2 with the original notation and an updated blue-white palette."""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from repeater_case import run_base_report


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "fig_attacker_posteriors_heatmap.pdf"


def build_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "repeater_blue",
        ["#f4f7fb", "#9bc4df", "#153f7f"],
    )


def annotate(ax: plt.Axes, matrix: np.ndarray) -> None:
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value >= 0.75 else "black",
                fontsize=18,
            )


def draw_panel(ax: plt.Axes, matrix: np.ndarray, title: str, cmap: LinearSegmentedColormap) -> plt.AxesImage:
    image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title(title, fontsize=19, pad=10)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([r"$|0\rangle$", r"$|1\rangle$"], fontsize=17)
    ax.set_yticklabels([r"$\langle 0|$", r"$\langle 1|$"], fontsize=17)
    ax.tick_params(axis="both", labelsize=16, width=1.8, length=7)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    annotate(ax, matrix)
    return image


def generate(output_path: Path) -> None:
    nonsecret_posterior, secret_posterior = _load_posteriors_from_report()
    cmap = build_colormap()

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.linewidth": 1.6,
            "mathtext.fontset": "dejavusans",
        }
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.6, 4.8),
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.03, wspace=0.08, hspace=0.02)
    fig.suptitle(r"Attacker posteriors on $q_M$ (base repeater instance)", fontsize=21)

    image_left = draw_panel(
        axes[0],
        nonsecret_posterior,
        r"$\bar{\sigma}_{O_{\mathrm{fg}},0}(\rho_0)=|0\rangle\langle 0|$",
        cmap,
    )
    image_right = draw_panel(
        axes[1],
        secret_posterior,
        r"$\bar{\sigma}_{O_{\mathrm{fg}},1}(\rho_0)=I_2/2$",
        cmap,
    )

    colorbar_left = fig.colorbar(
        image_left,
        ax=axes[0],
        fraction=0.046,
        pad=0.04,
        ticks=np.linspace(0.0, 1.0, 6),
    )
    colorbar_right = fig.colorbar(
        image_right,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
        ticks=np.linspace(0.0, 1.0, 6),
    )
    for colorbar in (colorbar_left, colorbar_right):
        colorbar.ax.tick_params(labelsize=15, width=1.6, length=6)
        colorbar.outline.set_linewidth(1.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _load_posteriors_from_report() -> tuple[np.ndarray, np.ndarray]:
    report = run_base_report("dense")
    if not report["observation_reports"]:
        raise ValueError("Repeater base report did not produce any observation posteriors.")
    observation = report["observation_reports"][0]
    secret = observation["secret_posterior"]
    nonsecret = observation["nonsecret_posterior"]
    if secret is None or nonsecret is None:
        raise ValueError("Repeater base report is missing posterior matrices.")
    return _parse_pretty_matrix(nonsecret), _parse_pretty_matrix(secret)


def _parse_pretty_matrix(raw: list[list[str]]) -> np.ndarray:
    return np.array([[float(_parse_scalar(entry)) for entry in row] for row in raw], dtype=float)


def _parse_scalar(text: str) -> float:
    stripped = text.strip()
    if "/" in stripped and all(marker not in stripped for marker in ("j", "J", "i", "I")):
        return float(Fraction(stripped))
    return float(stripped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
