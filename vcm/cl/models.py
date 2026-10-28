"""Networks cut into named role slots, so a task's network can be composed slot by slot.

A *template* says which role slots a task's network has. The fixed architectures
(`mlp`, `cnn`) have one template with every role in order. `compose_mlp` has two:
`unary` reads one image, `binary` reads two images with the same reader and mixes
the two codes. Roles of the reader act on each input item separately (the leading
item axis is kept), so one reader serves both images of a pair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

RoleFactory = Callable[..., nn.Module]


@dataclass(frozen=True)
class Template:
    name: str
    roles: tuple[str, ...]
    reader: int                       # the first `reader` roles act on each input item
    arity: int = 1                    # input items per sample


class Architecture:
    """Ordered role slots and a factory per role. The last role is the classifier head.

    With `widths` (role -> output width per item; None = number of classes), factories
    take (d_in, n_out, arity) and modules are sized to what they receive, so a slot can
    follow a concept of any output width. Without it, every factory takes no arguments.
    """

    def __init__(self, roles: list[tuple[str, RoleFactory]], input_shape: tuple[int, ...],
                 templates: list[Template] | None = None, widths: dict[str, int | None] | None = None):
        self.roles = [r for r, _ in roles]
        self._factories = dict(roles)
        self.input_shape = input_shape
        self.templated = widths is not None
        self._widths = widths or {}
        templates = templates or [Template("default", tuple(self.roles), len(self.roles) - 1)]
        self.templates = {t.name: t for t in templates}

    def make(self, role: str, d_in: int | None = None, n_out: int | None = None, arity: int = 1) -> nn.Module:
        if self.templated:
            return self._factories[role](d_in, n_out, arity)
        return self._factories[role]()

    def make_all(self) -> dict[str, nn.Module]:
        return {r: self.make(r) for r in self.roles}

    @property
    def head(self) -> str:
        return self.roles[-1]

    @property
    def item_dim(self) -> int:
        return math.prod(self.input_shape)

    def template(self, name: str) -> Template:
        return self.templates[name]

    def width(self, role: str, n_out: int | None) -> int | None:
        """Output width per item of a fresh module for `role` (templated architectures)."""
        w = self._widths.get(role)
        return n_out if w is None else w

    def build(self, roles, n_out: int | None = None, d_in: int | None = None,
              arity: int = 1) -> tuple[list[nn.Module], int | None]:
        """Fresh modules for consecutive roles, each sized to its input; returns the output width."""
        d, mods = d_in or self.item_dim, []
        for r in roles:
            mods.append(self.make(r, d, n_out, arity))
            d = self.width(r, n_out)
        return mods, d

    def network(self, template: str, n_out: int) -> nn.Sequential:
        t = self.template(template)
        return nn.Sequential(*self.build(t.roles, n_out, arity=t.arity)[0])


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


def compose_mlp(item_dim: int = 784, hidden: int = 256) -> Architecture:
    """Two templates over shared roles: unary = in, hidden, out (one image);
    binary = in, hidden on each of two images, then mix, out."""
    return Architecture([
        ("in", lambda d, n, k: nn.Sequential(nn.Linear(d, hidden), nn.ReLU())),
        ("hidden", lambda d, n, k: nn.Sequential(nn.Linear(d, hidden), nn.ReLU())),
        ("mix", lambda d, n, k: nn.Sequential(nn.Flatten(1), nn.Linear(k * d, hidden), nn.ReLU())),
        ("out", lambda d, n, k: nn.Linear(d, n)),
    ], (item_dim,),
        templates=[Template("unary", ("in", "hidden", "out"), reader=2),
                   Template("binary", ("in", "hidden", "mix", "out"), reader=2, arity=2)],
        widths={"in": hidden, "hidden": hidden, "mix": hidden, "out": None})


def make_architecture(benchmark_name: str, n_out: int) -> Architecture:
    if benchmark_name == "pmnist":
        return mlp(n_out=n_out)
    if benchmark_name == "mnist_compose":
        return compose_mlp()
    return cnn(n_out)


class Stack(nn.Module):
    """Plain sequential network over role modules, with optional per-task heads.

    With a templated architecture the body is the reader (the same roles in every
    template, applied to each input item) and a task's head is the rest of its template.
    """

    def __init__(self, arch: Architecture, multi_head: bool):
        super().__init__()
        self.arch = arch
        self.multi_head = multi_head
        self.heads = nn.ModuleDict()
        if arch.templated:
            readers = {t.roles[:t.reader] for t in arch.templates.values()}
            if not multi_head or len(readers) != 1:
                raise ValueError("a shared body needs one head per task and one reader for all templates")
            mods, self.d_body = arch.build(next(iter(readers)))
            self.body = nn.ModuleDict(dict(zip(next(iter(readers)), mods)))
            return
        self.body = nn.ModuleDict({r: arch.make(r) for r in arch.roles[:-1]})
        if not multi_head:
            self.heads["0"] = arch.make(arch.head)

    def add_head(self, task_id: int, template: str | None = None, n_out: int | None = None) -> None:
        if not self.multi_head or str(task_id) in self.heads:
            return
        if self.arch.templated:
            t = self.arch.template(template)
            mods, _ = self.arch.build(t.roles[t.reader:], n_out, self.d_body, t.arity)
            self.heads[str(task_id)] = nn.Sequential(*mods)
        else:
            self.heads[str(task_id)] = self.arch.make(self.arch.head)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.body.values():
            x = m(x)
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
        if isinstance(m, nn.Linear):  # once per input item (a reader sees each image of a pair)
            total[0] += m.in_features * m.out_features * (out.numel() // (out.shape[0] * out.shape[-1]))
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


def template_macs(arch: Architecture, template: str, n_out: int) -> int:
    """Forward multiply-accumulates per sample of a fresh network for this template."""
    t = arch.template(template)
    with torch.random.fork_rng(devices=[]):  # sizing a network must not shift the seed stream
        net = arch.network(template, n_out)
    x = torch.zeros(1, *((t.arity,) if t.arity > 1 else ()), *arch.input_shape)
    return module_macs(net, x)[0]


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
