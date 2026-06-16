from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    completed = subprocess.run([str(VENV_PYTHON), str(Path(__file__).resolve())])
    raise SystemExit(completed.returncode)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

try:
    from math_expr.executor_adapter import evaluate_graph_with_tree, load_program_from_graph_file
    from math_expr.expression_graph import tree_to_edges, tree_to_string
except ImportError:  # pragma: no cover - supports running from inside math_expr/
    from executor_adapter import evaluate_graph_with_tree, load_program_from_graph_file
    from expression_graph import tree_to_edges, tree_to_string


DEMO_GRAPH = (
    REPO_ROOT
    / "external"
    / "stasinos_and_boura_repository"
    / "math_expr"
    / "graphs"
    / "sine_in_exp.graph"
)


def main() -> None:
    program = load_program_from_graph_file(DEMO_GRAPH)
    x = np.linspace(-1.0, 1.0, 5)
    result, tree = evaluate_graph_with_tree(DEMO_GRAPH, x=x)

    print("Input .graph program:")
    print(program.strip())
    print()
    print("Pretty expression:")
    print(tree_to_string(tree))
    print()
    print("Tree edges:")
    for parent, child in tree_to_edges(tree):
        print(f"{parent} -> {child}")
    print()
    print("Original executor result successfully obtained:")
    print(result.execution_successful)


if __name__ == "__main__":
    main()
