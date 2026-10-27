"""Networks cut into named role slots, so a task's network can be composed slot by slot."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

RoleFactory = Callable[[], nn.Module]


class Architecture:
    """Ordered role slots and a factory per role. The last role is the classifier head."""

    def __init__(self, roles: list[tuple[str, RoleFactory]], input_shape: tuple[int, ...]):
        self.roles = [r for r, _ in roles]
        self._factories = dict(roles)
        self.input_shape = input_shape

    def make(self, role: str) -> nn.Module:
        return self._factories[role]()

    def make_all(self) -> dict[str, nn.Module]:
        return {r: self.make(r) for r in self.roles}

    @property
    def head(self) -> str:
        return self.roles[-1]


def mlp(in_dim: int = 784, hidden: int = 256, n_out: int = 10) -> Architecture:
    return Architecture([
        ("in", lambda: nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())),
        ("hidden", lambda: nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())),
        ("out", lambda: nn.Linear(hidden, n_out)),
    ], (in_dim,))


def cnn(n_out: int, width: int = 32) -> Architecture:
    w = width

    def block(cin, cout):
        return lambda: nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))

    return Architecture([
        ("block1", block(3, w)),
        ("block2", block(w, 2 * w)),
        ("block3", block(2 * w, 4 * w)),
        ("fc", lambda: nn.Sequential(nn.Flatten(), nn.Linear(4 * w * 16, 256), nn.ReLU())),
        ("head", lambda: nn.Linear(256, n_out)),
    ], (3, 32, 32))


def make_architecture(benchmark_name: str, n_out: int) -> Architecture:
    if benchmark_name == "pmnist":
        return mlp(n_out=n_out)
    return cnn(n_out)


class Stack(nn.Module):
    """Plain sequential network over role modules, with optional per-task heads."""

    def __init__(self, arch: Architecture, multi_head: bool):
        super().__init__()
        self.arch = arch
        self.body = nn.ModuleDict({r: arch.make(r) for r in arch.roles[:-1]})
        self.multi_head = multi_head
        self.heads = nn.ModuleDict()
        if not multi_head:
            self.heads["0"] = arch.make(arch.head)

    def add_head(self, task_id: int) -> None:
        if self.multi_head and str(task_id) not in self.heads:
            self.heads[str(task_id)] = self.arch.make(self.arch.head)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        for r in self.arch.roles[:-1]:
            x = self.body[r](x)
        return x

    def head(self, task_id: int | None) -> nn.Module:
        return self.heads[str(task_id)] if self.multi_head else self.heads["0"]

    def forward(self, x: torch.Tensor, task_id: int | None = None) -> torch.Tensor:
        return self.head(task_id)(self.features(x))


# -- cost accounting --------------------------------------------------------------
def module_macs(module: nn.Module, x: torch.Tensor) -> tuple[int, torch.Tensor]:
    """Multiply-accumulates per sample for one forward pass, and the output."""
    total = [0]

    def hook(m, inp, out):
        if isinstance(m, nn.Linear):
            total[0] += m.in_features * m.out_features
        elif isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] * m.kernel_size[1]
            total[0] += m.in_channels // m.groups * m.out_channels * k * out.shape[-1] * out.shape[-2]

    handles = [m.register_forward_hook(hook) for m in module.modules()
               if isinstance(m, (nn.Linear, nn.Conv2d))]
    with torch.no_grad():
        out = module(x)
    for h in handles:
        h.remove()
    return total[0], out


def role_macs(arch: Architecture) -> dict[str, int]:
    x = torch.zeros(1, *arch.input_shape)
    out = {}
    for r in arch.roles:
        out[r], x = module_macs(arch.make(r), x)
    return out


class ComputeMeter:
    """Counts multiply-accumulates. A training step costs ~3x a forward pass."""

    def __init__(self):
        self.counts: dict[str, float] = {}

    def add(self, kind: str, macs_per_sample: float, samples: int, backward: bool = False) -> None:
        self.counts[kind] = self.counts.get(kind, 0.0) + macs_per_sample * samples * (3 if backward else 1)

    @property
    def total(self) -> float:
        return sum(self.counts.values())

    def as_dict(self) -> dict[str, float]:
        return {**self.counts, "total": self.total}


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
