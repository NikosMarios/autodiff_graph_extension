"""Generate the experimental validation pipeline figure for Section 5.6."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "demo_outputs" / "figures"
MPL_CONFIG_DIR = REPO_ROOT / "demo_outputs" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    completed = subprocess.run([str(VENV_PYTHON), str(Path(__file__).resolve())])
    raise SystemExit(completed.returncode)

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


FONT_FAMILY = "DejaVu Sans"
TITLE_COLOR = "#1f2933"
TEXT_COLOR = "#25313d"
ARROW_COLOR = "#52616f"


def _draw_box(ax, y: float, label: str, fill: str, edge: str) -> None:
    box = FancyBboxPatch(
        (2.25, y - 0.38),
        5.5,
        0.76,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(box)
    ax.text(
        5,
        y,
        label,
        ha="center",
        va="center",
        color=TEXT_COLOR,
        fontsize=12.5,
        fontfamily=FONT_FAMILY,
        fontweight="bold",
    )


def _draw_arrow(ax, upper_y: float, lower_y: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (5, upper_y - 0.5),
            (5, lower_y + 0.5),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.25,
            color=ARROW_COLOR,
        )
    )


def main() -> None:
    """Render the validation flow as SVG and high-resolution PNG."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 8.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(
        5,
        9.45,
        "Experimental Validation Pipeline",
        ha="center",
        va="center",
        color=TITLE_COLOR,
        fontsize=16,
        fontfamily=FONT_FAMILY,
        fontweight="bold",
    )

    stages = [
        (8.35, "Symbolic Expression", "#f7f9fb", "#5c6b73"),
        (7.15, "Expression Tree", "#e8f3ec", "#4f7d5d"),
        (5.95, "PyTorch Module", "#f1ecf7", "#6f5b8c"),
        (4.75, "Forward Pass", "#e6f0f4", "#3d6f82"),
        (3.55, "Loss", "#f8f6ed", "#8a7a43"),
        (2.35, "Backward Pass", "#f7eeee", "#8b5d5d"),
        (1.15, "Gradient", "#edf2f7", "#64748b"),
    ]

    for y, label, fill, edge in stages:
        _draw_box(ax, y, label, fill, edge)

    for (upper_y, *_), (lower_y, *_) in zip(stages, stages[1:]):
        _draw_arrow(ax, upper_y, lower_y)

    svg_path = OUTPUT_DIR / "experimental_validation_pipeline.svg"
    png_path = OUTPUT_DIR / "experimental_validation_pipeline.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png_path, format="png", dpi=320, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    print("Saved experimental validation figure files:")
    print(svg_path)
    print(png_path)


if __name__ == "__main__":
    main()
