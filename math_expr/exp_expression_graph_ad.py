import torch

from expression_graph import Node, parse_program_to_tree, tree_to_string


def evaluate_tree_torch(node: Node, x: torch.Tensor) -> torch.Tensor:
    op_type = node.op_type.upper()
    if op_type == "X":
        return x
    if op_type == "SUM":
        return sum(evaluate_tree_torch(child, x) for child in node.children)

    child_value = evaluate_tree_torch(node.children[0], x)
    if op_type == "SIN":
        return torch.sin(child_value)
    if op_type == "EXP":
        return torch.exp(child_value)
    if op_type in {"LN", "LOG"}:
        return torch.log(child_value)
    if op_type == "SINC":
        return torch.sinc(child_value)
    if op_type == "POLY":
        result = torch.zeros_like(child_value)
        for power, coefficient in enumerate(node.params or []):
            result = result + float(coefficient) * child_value.pow(power)
        return result

    raise ValueError(f"Unsupported operation for torch evaluation: {node.op_type}")


def main():
    program = "sin(), exp()"
    tree = parse_program_to_tree(program)
    x = torch.linspace(-1.0, 1.0, steps=5, requires_grad=True)
    y = evaluate_tree_torch(tree, x)
    loss = (y ** 2).mean()
    loss.backward()

    print("Expression:")
    print(tree_to_string(tree))
    print()
    print("Loss:")
    print(loss.item())
    print()
    print("x.grad:")
    print(x.grad)
    print()
    print("Gradients are propagated through the same expression graph structure used to represent the DSL program.")


if __name__ == "__main__":
    main()
