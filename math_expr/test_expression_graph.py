from pathlib import Path
import shutil

import torch

import executor_adapter
from exp_expression_graph_ad import evaluate_tree_torch
from expression_graph import parse_program_to_tree, tree_to_mermaid, tree_to_string
from module_from_tree import node_to_module


def test_parse_composition_to_string():
    tree = parse_program_to_tree("sin(), exp()")

    assert tree.op_type == "SIN"
    assert tree_to_string(tree) == "sin(exp(x))"


def test_multiline_expression_contains_addition():
    tree = parse_program_to_tree("sin(), poly([0,3]), exp()\nsinc()")

    assert "+" in tree_to_string(tree)


def test_torch_evaluator_supports_backward():
    tree = parse_program_to_tree("sin(), exp()")
    x = torch.linspace(-1.0, 1.0, steps=5, requires_grad=True)
    y = evaluate_tree_torch(tree, x)
    loss = y.mean()

    loss.backward()

    assert isinstance(y, torch.Tensor)
    assert x.grad is not None


def test_node_to_module_matches_recursive_torch_evaluator():
    tree = parse_program_to_tree("sin(), poly([1, 2]), exp()")
    model = node_to_module(tree)
    x = torch.linspace(-1.0, 1.0, steps=5)

    assert torch.allclose(model(x), evaluate_tree_torch(tree, x))


def test_node_to_module_supports_backward_for_sum():
    tree = parse_program_to_tree("sin(), exp()\nsinc()")
    model = node_to_module(tree)
    x = torch.linspace(-1.0, 1.0, steps=5, requires_grad=True)
    loss = model(x).mean()

    loss.backward()

    assert x.grad is not None


def test_tree_to_mermaid_exports_basic_composition():
    tree = parse_program_to_tree("sin(), exp()")
    mermaid = tree_to_mermaid(tree)

    assert "graph TD" in mermaid
    assert '["SIN"]' in mermaid
    assert '["EXP"]' in mermaid
    assert '["X"]' in mermaid


def test_evaluate_graph_with_tree_calls_original_executor_in_compatible_workspace(monkeypatch):
    executor_adapter.TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    source_workspace = executor_adapter.TEMP_ROOT / "adapter_test_source"
    shutil.rmtree(source_workspace, ignore_errors=True)
    source_workspace.mkdir(parents=True)
    try:
        graph_file = source_workspace / "demo.graph"
        graph_file.write_text("sin(), exp()", encoding="utf-8")
        observed = {}

        def fake_original_execute(program_name, x=None):
            observed["program_name"] = program_name
            observed["graph_exists"] = Path("graphs/demo.graph").exists()
            observed["x"] = x
            return "original-result"

        monkeypatch.setattr(executor_adapter, "original_execute", fake_original_execute)

        result, tree = executor_adapter.evaluate_graph_with_tree(graph_file, x="sample-x")

        assert observed == {
            "program_name": "demo",
            "graph_exists": True,
            "x": "sample-x",
        }
        assert result.original_return == "original-result"
        assert result.execution_successful is True
        assert tree_to_string(tree) == "sin(exp(x))"
    finally:
        shutil.rmtree(source_workspace, ignore_errors=True)
