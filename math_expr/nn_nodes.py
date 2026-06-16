from __future__ import annotations

import torch
from torch import nn


class SinNode(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class ExpNode(nn.Module):
    def forward(self, x):
        return torch.exp(x)


class LnNode(nn.Module):
    def forward(self, x):
        return torch.log(x)


class SincNode(nn.Module):
    def forward(self, x):
        return torch.sinc(x)


class PolyNode(nn.Module):
    def __init__(self, coeffs):
        super().__init__()
        self.coeffs = [float(coefficient) for coefficient in (coeffs or [])]

    def forward(self, x):
        result = torch.zeros_like(x)
        for power, coefficient in enumerate(self.coeffs):
            result = result + coefficient * x.pow(power)
        return result


class SumNode(nn.Module):
    def __init__(self, children: list[nn.Module]):
        super().__init__()
        self.terms = nn.ModuleList(children)

    def forward(self, x):
        return sum(child(x) for child in self.terms)


class ComposeNode(nn.Module):
    def __init__(self, outer: nn.Module, inner: nn.Module):
        super().__init__()
        self.outer = outer
        self.inner = inner

    def forward(self, x):
        return self.outer(self.inner(x))
