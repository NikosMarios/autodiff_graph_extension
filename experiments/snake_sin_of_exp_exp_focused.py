from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    completed = subprocess.run([str(VENV_PYTHON), str(Path(__file__).resolve())])
    raise SystemExit(completed.returncode)

MPL_CONFIG_DIR = REPO_ROOT / "demo_outputs" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

EXTERNAL_MATH_EXPR = REPO_ROOT / "external" / "stasinos_and_boura_repository" / "math_expr"
for path in (REPO_ROOT, EXTERNAL_MATH_EXPR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def ensure_snake_activation_importable() -> None:
    try:
        import snake.activations  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    snake_dir = EXTERNAL_MATH_EXPR / "snake"
    activations_pyc = snake_dir / "__pycache__" / "activations.cpython-312.pyc"
    if not activations_pyc.exists():
        return

    snake_package = types.ModuleType("snake")
    snake_package.__path__ = [str(snake_dir)]  # type: ignore[attr-defined]
    sys.modules.setdefault("snake", snake_package)

    spec = importlib.util.spec_from_file_location("snake.activations", activations_pyc)
    if spec is None or spec.loader is None:
        return

    module = importlib.util.module_from_spec(spec)
    sys.modules["snake.activations"] = module
    spec.loader.exec_module(module)


ensure_snake_activation_importable()

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from math_expr.expression_graph import Node, parse_program_to_tree, tree_to_string
from networks import SinFunComposition


FIGURE_PATH = REPO_ROOT / "demo_outputs" / "figures" / "snake_sin_of_exp_improved.png"
METRICS_PATH = REPO_ROOT / "demo_outputs" / "results" / "snake_sin_of_exp_metrics.json"


@dataclass(frozen=True)
class TrainingConfig:
    x_min: float = -1.5
    x_max: float = 1.5
    train_points: int = 8192
    eval_points: int = 1000
    batch_size: int = 64
    epochs: int = 80
    lr: float = 1e-3
    lambda_exp: float = 3.0
    seed: int = 42


def build_dataset_sin_of_exp(config: TrainingConfig) -> TensorDataset:
    x = torch.linspace(config.x_min, config.x_max, config.train_points).unsqueeze(1)
    y_exp = torch.exp(x)
    y_f = torch.sin(y_exp)
    return TensorDataset(x, y_f, y_exp)


def build_models() -> SinFunComposition:
    return SinFunComposition(a=1.0, small=True)


def parse_and_validate_expression_graph(program: str) -> Node:
    tree = parse_program_to_tree(program)
    assert tree.op_type.upper() == "SIN", f"Expected SIN root, found {tree.op_type}"
    assert len(tree.children) == 1, f"Expected one SIN child, found {len(tree.children)}"

    inner = tree.children[0]
    assert inner.op_type.upper() == "EXP", f"Expected EXP inner node, found {inner.op_type}"
    assert len(inner.children) == 1, f"Expected one EXP child, found {len(inner.children)}"
    assert inner.children[0].op_type.upper() == "X", "Expected inner EXP branch to consume x"

    print("Expression graph program:")
    print(program)
    print("Parsed expression tree:", tree_to_string(tree))
    print("Outer node:", tree.op_type.upper())
    print("Inner node:", inner.op_type.upper())
    print()
    return tree


def forward_components(
    model: SinFunComposition,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    outer_model, inner_model = model.get_internal()
    g2_hat = inner_model(x)
    f_hat = outer_model(g2_hat)
    return f_hat, g2_hat


def diagnostic_outer_sin_loss(model: SinFunComposition, config: TrainingConfig) -> float:
    outer_model, _ = model.get_internal()
    z_min = torch.exp(torch.tensor(config.x_min)).item()
    z_max = torch.exp(torch.tensor(config.x_max)).item()
    z_eval = torch.linspace(z_min, z_max, config.eval_points).unsqueeze(1)

    with torch.no_grad():
        sin_hat = outer_model(z_eval)
        sin_true = torch.sin(z_eval)
        return nn.functional.mse_loss(sin_hat, sin_true).item()


def train_sin_of_exp_with_exp_aux_loss(
    model: SinFunComposition,
    train_loader: DataLoader,
    config: TrainingConfig,
) -> list[dict[str, float]]:
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        totals = {"loss_f": 0.0, "loss_exp": 0.0, "total_loss": 0.0}
        batches = 0

        for x, y_f, y_exp in train_loader:
            optimizer.zero_grad()
            f_hat, g2_hat = forward_components(model, x)

            loss_f = loss_fn(f_hat, y_f)
            loss_exp = loss_fn(g2_hat, y_exp)
            total_loss = loss_f + config.lambda_exp * loss_exp

            total_loss.backward()
            optimizer.step()

            totals["loss_f"] += loss_f.item()
            totals["loss_exp"] += loss_exp.item()
            totals["total_loss"] += total_loss.item()
            batches += 1

        epoch_metrics = {name: value / batches for name, value in totals.items()}
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch:03d}/{config.epochs} "
            f"loss_f={epoch_metrics['loss_f']:.6f} "
            f"loss_exp={epoch_metrics['loss_exp']:.6f} "
            f"total={epoch_metrics['total_loss']:.6f}"
        )

    return history


def plot_sin_of_exp_results(
    model: SinFunComposition,
    config: TrainingConfig,
    output_path: Path = FIGURE_PATH,
) -> dict[str, float]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    x_eval = torch.linspace(config.x_min, config.x_max, config.eval_points).unsqueeze(1)
    with torch.no_grad():
        f_hat, exp_hat = forward_components(model, x_eval)
        exp_true = torch.exp(x_eval)
        f_true = torch.sin(exp_true)

        metrics = {
            "mse_f": nn.functional.mse_loss(f_hat, f_true).item(),
            "mse_exp": nn.functional.mse_loss(exp_hat, exp_true).item(),
            "mse_outer_sin_diagnostic": diagnostic_outer_sin_loss(model, config),
        }

    x_np = x_eval.squeeze(1).cpu().numpy()
    f_true_np = f_true.squeeze(1).cpu().numpy()
    f_hat_np = f_hat.squeeze(1).cpu().numpy()
    exp_true_np = exp_true.squeeze(1).cpu().numpy()
    exp_hat_np = exp_hat.squeeze(1).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160)

    axes[0].plot(x_np, f_true_np, linewidth=2.0, label="true f(x) = sin(exp(x))")
    axes[0].plot(x_np, f_hat_np, "--", linewidth=2.0, label="predicted f_hat(x)")
    axes[0].set_title("Composite Function Approximation")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(x_np, exp_true_np, linewidth=2.0, label="true g2(x) = exp(x)")
    axes[1].plot(x_np, exp_hat_np, "--", linewidth=2.0, label="predicted g2_hat(x)")
    axes[1].set_title("Focused Inner Function Approximation")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("g2(x)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return metrics


def save_metrics(metrics: dict[str, float], output_path: Path = METRICS_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    config = TrainingConfig()
    torch.manual_seed(config.seed)

    parse_and_validate_expression_graph("sin(), exp()")

    dataset = build_dataset_sin_of_exp(config)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = build_models()

    print("Model:")
    print(model)
    print()
    print(
        "Training with total_loss = MSE(f_hat, f) "
        f"+ {config.lambda_exp} * MSE(g2_hat, exp)"
    )

    train_sin_of_exp_with_exp_aux_loss(model, train_loader, config)
    metrics = plot_sin_of_exp_results(model, config)
    save_metrics(metrics)

    print()
    print("Evaluation metrics:")
    print(f"mse_f={metrics['mse_f']:.8f}")
    print(f"mse_exp={metrics['mse_exp']:.8f}")
    print(f"mse_outer_sin_diagnostic={metrics['mse_outer_sin_diagnostic']:.8f}")
    print(f"Saved figure: {FIGURE_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
