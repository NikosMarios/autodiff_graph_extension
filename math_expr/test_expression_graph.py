import torch

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
