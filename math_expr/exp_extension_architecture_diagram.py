"""Generate report-ready architecture diagrams for the autodiff extension.

This script is intentionally isolated from the original math_expr framework. It
uses Matplotlib only as a rendering backend and does not import or modify any
framework modules.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "demo_outputs" / "figures"
MPL_CONFIG_DIR = REPO_ROOT / ".tmp" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
except ModuleNotFoundError as exc:
    if exc.name != "matplotlib":
        raise

    venv_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    current_python = Path(sys.executable).resolve()
    if venv_python.exists() and current_python != venv_python.resolve():
        completed = subprocess.run([str(venv_python), str(Path(__file__).resolve())])
        raise SystemExit(completed.returncode)

    raise SystemExit(
        "Matplotlib is required to generate the diagrams. "
        "Install project requirements or run with .venv\\Scripts\\python.exe."
    ) from exc


FONT_FAMILY = "DejaVu Sans"
TITLE_COLOR = "#1f2933"
TEXT_COLOR = "#25313d"
ARROW_COLOR = "#52616f"
FRAME_COLOR = "#5c6b73"


def _draw_rounded_box(
    ax: plt.Axes,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    label: str,
    facecolor: str,
    edgecolor: str = FRAME_COLOR,
    linewidth: float = 1.15,
    fontweight: str = "normal",
) -> None:
    """Draw one rounded, paper-style process box."""
    left = center_x - width / 2
    bottom = center_y - height / 2
    box = FancyBboxPatch(
        (left, bottom),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        mutation_aspect=1.0,
    )
    ax.add_patch(box)
    ax.text(
        center_x,
        center_y,
        label,
        ha="center",
        va="center",
        color=TEXT_COLOR,
        fontsize=13,
        fontfamily=FONT_FAMILY,
        fontweight=fontweight,
        linespacing=1.2,
    )


def _draw_down_arrow(
    ax: plt.Axes,
    x: float,
    start_y: float,
    end_y: float,
    color: str = ARROW_COLOR,
) -> None:
    """Draw a clean downward arrow between two vertically stacked stages."""
    arrow = FancyArrowPatch(
        (x, start_y),
        (x, end_y),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.25,
        color=color,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)


def _save_figure(fig: plt.Figure, output_stem: Path) -> Sequence[Path]:
    """Save one figure as both SVG and high-resolution PNG."""
    svg_path = output_stem.with_suffix(".svg")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png_path, format="png", dpi=320, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return svg_path, png_path


def _configure_canvas(width: float, height: float) -> tuple[plt.Figure, plt.Axes]:
    """Create a white-background canvas suitable for Word/PDF reports."""
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    return fig, ax


def generate_expression_pipeline(output_dir: Path) -> Sequence[Path]:
    """Figure 5.1: Expression Graph Construction Pipeline."""
    fig, ax = _configure_canvas(width=5.6, height=7.2)
    ax.text(
        5,
        9.35,
        "Expression Graph Construction Pipeline",
        ha="center",
        va="center",
        color=TITLE_COLOR,
        fontsize=16,
        fontfamily=FONT_FAMILY,
        fontweight="bold",
    )

    stages = [
        (7.95, ".graph Program\n(Symbolic DSL)"),
        (6.35, "Parser"),
        (4.75, "Node Objects"),
        (3.15, "Expression Graph"),
    ]

    box_width = 4.5
    box_height = 0.72
    for y, label in stages:
        _draw_rounded_box(
            ax=ax,
            center_x=5,
            center_y=y,
            width=box_width,
            height=box_height,
            label=label,
            facecolor="#f7f9fb",
        )

    for (upper_y, _), (lower_y, _) in zip(stages, stages[1:]):
        _draw_down_arrow(
            ax=ax,
            x=5,
            start_y=upper_y - box_height / 2 - 0.12,
            end_y=lower_y + box_height / 2 + 0.12,
        )

    return _save_figure(fig, output_dir / "figure_5_1_expression_pipeline")


def generate_extension_architecture(output_dir: Path) -> Sequence[Path]:
    """Figure 5.2: Extension Architecture."""
    fig, ax = _configure_canvas(width=6.0, height=8.3)
    ax.text(
        5,
        9.35,
        "Extension Architecture",
        ha="center",
        va="center",
        color=TITLE_COLOR,
        fontsize=16,
        fontfamily=FONT_FAMILY,
        fontweight="bold",
    )

    stages = [
        {
            "y": 8.1,
            "label": "Original Framework",
            "face": "#eef1f4",
            "edge": "#77818a",
            "weight": "normal",
        },
        {
            "y": 6.85,
            "label": "Executor Adapter",
            "face": "#e6f0f4",
            "edge": "#3d6f82",
            "weight": "bold",
        },
        {
            "y": 5.6,
            "label": "Expression Graph Layer",
            "face": "#e8f3ec",
            "edge": "#4f7d5d",
            "weight": "bold",
        },
        {
            "y": 4.35,
            "label": "Visualization",
            "face": "#f8f6ed",
            "edge": "#8a7a43",
            "weight": "normal",
        },
        {
            "y": 3.1,
            "label": "PyTorch Modules",
            "face": "#f1ecf7",
            "edge": "#6f5b8c",
            "weight": "bold",
        },
        {
            "y": 1.85,
            "label": "Automatic Differentiation",
            "face": "#f7eeee",
            "edge": "#8b5d5d",
            "weight": "bold",
        },
    ]

    box_width = 4.9
    box_height = 0.7
    for stage in stages:
        _draw_rounded_box(
            ax=ax,
            center_x=5,
            center_y=stage["y"],
            width=box_width,
            height=box_height,
            label=stage["label"],
            facecolor=stage["face"],
            edgecolor=stage["edge"],
            linewidth=1.2,
            fontweight=stage["weight"],
        )

    for upper, lower in zip(stages, stages[1:]):
        _draw_down_arrow(
            ax=ax,
            x=5,
            start_y=upper["y"] - box_height / 2 - 0.1,
            end_y=lower["y"] + box_height / 2 + 0.1,
        )

    # A subtle bracket labels which layers belong to the developed extension.
    bracket_x = 8.05
    ax.plot(
        [bracket_x, bracket_x],
        [1.5, 7.2],
        color="#6b7780",
        linewidth=1.0,
    )
    ax.plot([bracket_x - 0.25, bracket_x], [7.2, 7.2], color="#6b7780", linewidth=1.0)
    ax.plot([bracket_x - 0.25, bracket_x], [1.5, 1.5], color="#6b7780", linewidth=1.0)
    ax.text(
        8.55,
        4.35,
        "Developed\nExtension",
        ha="center",
        va="center",
        rotation=90,
        color="#46525c",
        fontsize=10.5,
        fontfamily=FONT_FAMILY,
    )

    return _save_figure(fig, output_dir / "figure_5_2_extension_architecture")


def main() -> None:
    """Generate all extension architecture figures and print their locations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for paths in (
        generate_expression_pipeline(OUTPUT_DIR),
        generate_extension_architecture(OUTPUT_DIR),
    ):
        saved_paths.extend(paths)

    print("Saved architecture diagram files:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
