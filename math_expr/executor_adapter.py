from __future__ import annotations

import shutil
import os
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MPL_CONFIG_DIR = REPO_ROOT / "demo_outputs" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

from external.stasinos_and_boura_repository.math_expr.executor import execute as original_execute

try:
    from math_expr.expression_graph import Node, parse_program_to_tree
except ImportError:  # pragma: no cover - supports running examples from inside math_expr/
    from expression_graph import Node, parse_program_to_tree


TEMP_ROOT = REPO_ROOT / "demo_outputs" / "executor_adapter_work"


@dataclass(frozen=True)
class ExecutorAdapterResult:
    """Execution metadata returned by the adapter wrapper."""

    graph_path: Path
    program_name: str
    original_return: Any
    execution_successful: bool


def load_program_from_graph_file(path: str | PathLike[str]) -> str:
    return Path(path).read_text(encoding="utf-8")


def evaluate_graph_with_tree(
    graph_path: str | PathLike[str],
    x=None,
) -> tuple[ExecutorAdapterResult, Node]:
    """
    Parse a .graph file to an expression tree and evaluate it with the original
    executor without modifying the external framework.

    The original executor expects to open graphs/<program>.graph relative to the
    process cwd and writes plot files as a side effect. This wrapper creates a
    temporary compatible workspace, calls the unchanged executor there, and then
    returns execution metadata together with the parsed tree.
    """
    graph_file = Path(graph_path).resolve()
    program = load_program_from_graph_file(graph_file)
    tree = parse_program_to_tree(program)

    with _original_executor_workspace(graph_file) as program_name:
        original_return = original_execute(program_name, x)

    result = ExecutorAdapterResult(
        graph_path=graph_file,
        program_name=program_name,
        original_return=original_return,
        execution_successful=True,
    )
    return result, tree


def execute_with_tree(
    graph_path: str | PathLike[str],
    x=None,
    *,
    return_tree: bool = False,
):
    """
    Read a .graph file, optionally parse it to an expression tree, then call
    the original executor without changing its implementation.

    The original executor accepts a program stem, not a path or program string:
    execute("sine", x) opens graphs/sine.graph relative to the process cwd.
    """
    result, tree = evaluate_graph_with_tree(graph_path, x)

    if return_tree:
        return result.original_return, tree
    return result.original_return


@contextmanager
def _original_executor_workspace(graph_file: Path) -> Iterator[str]:
    """Create the graphs/<program>.graph layout expected by original_execute."""
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    workspace_path = TEMP_ROOT / f"executor_adapter_{uuid.uuid4().hex}"
    graphs_dir = workspace_path / "graphs"
    graphs_dir.mkdir(parents=True)
    shutil.copy2(graph_file, graphs_dir / graph_file.name)

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace_path)
        yield graph_file.stem
    finally:
        os.chdir(previous_cwd)
        shutil.rmtree(workspace_path, ignore_errors=True)
