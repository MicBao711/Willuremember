"""Common learner interface, training loop helpers, and the task router."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..data import Task
from ..models import Architecture, ComputeMeter, role_macs, template_macs


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 128
    lr: float = 1e-3
    eval_batch: int = 2000
    min_steps: int = 0        # train at least this many steps per task (small training sets)


class Learner:
    name = "base"

    def __init__(self, arch: Architecture, task_il: bool, cfg: TrainConfig, seed: int = 0):
        self.arch = arch
        self.task_il = task_il
        self.cfg = cfg
        self.seed = seed
        self.gen = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        self.meter = ComputeMeter()
        self.macs = {} if arch.templated else role_macs(arch)
        self.full_macs = float(sum(self.macs.values()))
        self._costs: dict[tuple[str, int], float] = {}
        self.seen: list[int] = []

    def cost(self, task: Task) -> float:
        """Forward multiply-accumulates per sample of a full network for this task."""
        if not self.arch.templated:
            return self.full_macs
        key = (task.template, task.n_classes)
        if key not in self._costs:
            self._costs[key] = float(template_macs(self.arch, *key))
        return self._costs[key]

    def n_steps(self, n: int) -> int:
        """Training steps for a task with n samples: the epochs, or `min_steps` if more."""
        return max(self.cfg.epochs * int(np.ceil(n / self.cfg.batch_size)), self.cfg.min_steps)

    def learn(self, task: Task) -> None:
        raise NotImplementedError

    def predict(self, x: torch.Tensor, task_id: int | None) -> torch.Tensor:
        raise NotImplementedError

    def memory(self) -> dict[str, int]:
        """Floats kept: model parameters (incl. snapshots) and other stored state."""
        raise NotImplementedError

    def info(self) -> dict:
        return {}

    # -- helpers ------------------------------------------------------------
    def batches(self, n: int, *, epochs: int | None = None, steps: int | None = None,
                batch_size: int | None = None) -> Iterator[torch.Tensor]:
        bs = batch_size or self.cfg.batch_size
        if steps is None and epochs is None and self.cfg.min_steps:
            if self.cfg.epochs * int(np.ceil(n / bs)) < self.cfg.min_steps:
                steps = self.cfg.min_steps
        if steps is not None:
            done = 0
            while done < steps:
                perm = torch.randperm(n, generator=self.gen)
                for i in range(0, n - bs + 1 if n >= bs else 1, bs):
                    yield perm[i:i + bs]
                    done += 1
                    if done >= steps:
                        return
            return
        for _ in range(epochs if epochs is not None else self.cfg.epochs):
            perm = torch.randperm(n, generator=self.gen)
            for i in range(0, n, bs):
                yield perm[i:i + bs]


def batched_predict(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, batch: int) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([fn(x[i:i + batch]) for i in range(0, len(x), batch)])


def accuracy(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, y: torch.Tensor,
             batch: int = 2000) -> float:
    if len(x) == 0:
        return float("nan")
    return (batched_predict(fn, x, batch) == y).float().mean().item()


def train_steps(params, loss_fn: Callable[[torch.Tensor], torch.Tensor], index_iter,
                lr: float | list[dict]) -> int:
    """Adam over `params`, or over the param groups passed as `lr`; returns steps taken."""
    groups = lr if isinstance(lr, list) else [{"params": list(params), "lr": lr}]
    groups = [g for g in groups if g["params"]]
    if not groups:
        return 0
    opt = torch.optim.Adam(groups)
    steps = 0
    for idx in index_iter:
        opt.zero_grad(set_to_none=True)
        loss_fn(idx).backward()
        opt.step()
        steps += 1
    return steps


def ce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y)


class TaskRouter:
    """Infers the hidden task variable from input statistics (nearest centroid, cosine)."""

    def __init__(self):
        self.ids: list[int] = []
        self.centroids: list[torch.Tensor] = []

    def add(self, task_id: int, x: torch.Tensor) -> None:
        self.ids.append(task_id)
        self.centroids.append(F.normalize(x.flatten(1).mean(0), dim=0))

    def route(self, x: torch.Tensor) -> torch.Tensor:
        sims = F.normalize(x.flatten(1), dim=1) @ torch.stack(self.centroids).T
        return torch.tensor(self.ids)[sims.argmax(1)]

    def floats(self) -> int:
        return sum(c.numel() for c in self.centroids)


def count_floats(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
