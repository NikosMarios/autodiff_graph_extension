from __future__ import annotations

from pathlib import Path

from external.stasinos_and_boura_repository.math_expr.executor import execute as original_execute

try:
    from math_expr.expression_graph import parse_program_to_tree
except ImportError:  # pragma: no cover - supports running examples from inside math_expr/
    from expression_graph import parse_program_to_tree


def load_program_from_graph_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def execute_with_tree(graph_path: str, x=None, *, return_tree: bool = False):
    """
    Read a .graph file, optionally parse it to an expression tree, then call
    the original executor without changing its implementation.

    The original executor accepts a program stem, not a path or program string:
    execute("sine", x) opens graphs/sine.graph relative to the process cwd.
    """
    program = load_program_from_graph_file(graph_path)
    tree = parse_program_to_tree(program) if return_tree else None

    y = original_execute(Path(graph_path).stem, x)

    if return_tree:
        return y, tree
    return y
