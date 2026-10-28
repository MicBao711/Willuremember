"""Reference learners: fine-tuning, EWC, experience replay, one network per task."""

from __future__ import annotations

import torch
from torch.func import functional_call, grad, vmap

from ..data import Task
from .base import Learner, TaskRouter, accuracy, ce, count_floats, train_steps
from ..models import Stack


class FineTune(Learner):
    """Plain sequential training of one shared network: the lower bound."""

    name = "finetune"

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.model = Stack(self.arch, multi_head=self.task_il)

    def head_id(self, task_id: int | None) -> int | None:
        return task_id if self.task_il else None

    def loss(self, task: Task, idx: torch.Tensor) -> torch.Tensor:
        x, y = task.x_train[idx], task.y_train[idx]
        self.meter.add("train", self.cost(task), len(idx), backward=True)
        return ce(self.model(x, self.head_id(task.id)), y) + self.penalty()

    def penalty(self) -> torch.Tensor | float:
        return 0.0

    def learn(self, task: Task) -> None:
        self.model.add_head(task.id, task.template, task.n_classes)
        self.model.train()
        train_steps(self.model.parameters(), lambda idx: self.loss(task, idx),
                    self.batches(len(task.x_train)), self.cfg.lr)
        self.seen.append(task.id)
        self.after_task(task)

    def after_task(self, task: Task) -> None:
        pass

    def predict(self, x: torch.Tensor, task_id: int | None) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(x, self.head_id(task_id)).argmax(1)

    def memory(self) -> dict[str, int]:
        return {"params": count_floats(self.model), "other": 0}


class EWC(FineTune):
    """Online EWC (Schwarz et al., 2018): one anchor, Fisher accumulated with decay gamma."""

    name = "ewc"

    def __init__(self, *args, lam: float = 100.0, gamma: float = 1.0, n_fisher: int = 1000, **kw):
        super().__init__(*args, **kw)
        self.lam, self.gamma, self.n_fisher = lam, gamma, n_fisher
        self.fisher: dict[str, torch.Tensor] = {}
        self.anchor: dict[str, torch.Tensor] = {}

    def penalty(self):
        if not self.fisher:
            return 0.0
        total = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                total = total + (self.fisher[n] * (p - self.anchor[n]) ** 2).sum()
        return 0.5 * self.lam * total

    def after_task(self, task: Task) -> None:
        self.model.eval()
        params = {n: p.detach() for n, p in self.model.named_parameters()}
        head = self.head_id(task.id)

        def sample_loss(p, x, y):
            out = functional_call(self.model, p, (x.unsqueeze(0), head))
            return ce(out, y.unsqueeze(0))

        per_sample = vmap(grad(sample_loss), in_dims=(None, 0, 0))
        idx = torch.randperm(len(task.x_train), generator=self.gen)[: self.n_fisher]
        new = {n: torch.zeros_like(p) for n, p in params.items()}
        for i in range(0, len(idx), 100):
            b = idx[i:i + 100]
            grads = per_sample(params, task.x_train[b], task.y_train[b])
            for n, g in grads.items():
                new[n] += (g ** 2).sum(0)
            self.meter.add("fisher", self.cost(task), len(b), backward=True)
        for n in new:
            new[n] /= len(idx)
            self.fisher[n] = self.gamma * self.fisher.get(n, torch.zeros_like(new[n])) + new[n]
            self.anchor[n] = params[n].clone()

    def memory(self) -> dict[str, int]:
        extra = sum(f.numel() for f in self.fisher.values()) + sum(a.numel() for a in self.anchor.values())
        return {"params": count_floats(self.model), "other": extra}


class Replay(FineTune):
    """Experience replay with a per-task buffer, mixed 1:1 into every batch."""

    name = "er"

    def __init__(self, *args, per_task: int = 200, **kw):
        super().__init__(*args, **kw)
        self.per_task = per_task
        self.buf_x: list[torch.Tensor] = []
        self.buf_y: list[torch.Tensor] = []
        self.buf_t: list[torch.Tensor] = []
        self.buf_cost: list[float] = []

    def loss(self, task: Task, idx: torch.Tensor) -> torch.Tensor:
        loss = super().loss(task, idx)
        if not self.buf_x:
            return loss
        if self.arch.templated:  # buffers of different templates have different input shapes
            r = torch.randint(sum(len(b) for b in self.buf_x), (len(idx),), generator=self.gen)
            return loss + self._replay_per_task(r)
        bx, by, bt = torch.cat(self.buf_x), torch.cat(self.buf_y), torch.cat(self.buf_t)
        r = torch.randint(len(bx), (len(idx),), generator=self.gen)
        bx, by, bt = bx[r], by[r], bt[r]
        self.meter.add("replay", self.full_macs, len(r), backward=True)
        if not self.task_il:
            return loss + ce(self.model(bx), by)
        feats = self.model.features(bx)
        out = 0.0
        for t in bt.unique().tolist():
            m = bt == t
            out = out + ce(self.model.head(t)(feats[m]), by[m]) * m.sum() / len(bt)
        return loss + out

    def _replay_per_task(self, r: torch.Tensor) -> torch.Tensor | float:
        """Replay loss for global buffer indices r, one forward pass per stored task."""
        out, start = 0.0, 0
        for bx, by, bt, cost in zip(self.buf_x, self.buf_y, self.buf_t, self.buf_cost):
            local = r[(r >= start) & (r < start + len(bx))] - start
            start += len(bx)
            if len(local):
                self.meter.add("replay", cost, len(local), backward=True)
                out = out + ce(self.model(bx[local], int(bt[0])), by[local]) * len(local) / len(r)
        return out

    def after_task(self, task: Task) -> None:
        idx = torch.randperm(len(task.x_train), generator=self.gen)[: self.per_task]
        self.buf_x.append(task.x_train[idx])
        self.buf_y.append(task.y_train[idx])
        self.buf_t.append(torch.full((len(idx),), task.id))
        self.buf_cost.append(self.cost(task))

    def memory(self) -> dict[str, int]:
        return {"params": count_floats(self.model), "other": sum(b[0].numel() * len(b) for b in self.buf_x)}


class Independent(Learner):
    """A fresh network per task: no forgetting, no transfer, memory grows linearly."""

    name = "independent"

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.models: dict[int, Stack] = {}
        self.router = TaskRouter()

    def learn(self, task: Task) -> None:
        model = (self.arch.network(task.template, task.n_classes) if self.arch.templated
                 else Stack(self.arch, multi_head=False))
        model.train()
        train_steps(model.parameters(),
                    lambda idx: self._loss(model, task, idx),
                    self.batches(len(task.x_train)), self.cfg.lr)
        self.models[task.id] = model.eval()
        self.router.add(task.id, task.x_train)
        self.seen.append(task.id)

    def _loss(self, model, task, idx):
        self.meter.add("train", self.cost(task), len(idx), backward=True)
        return ce(model(task.x_train[idx]), task.y_train[idx])

    def predict(self, x: torch.Tensor, task_id: int | None) -> torch.Tensor:
        with torch.no_grad():
            if task_id is not None:
                return self.models[task_id](x).argmax(1)
            routes = self.router.route(x)
            out = torch.empty(len(x), dtype=torch.long)
            for t in routes.unique().tolist():
                m = routes == t
                out[m] = self.models[t](x[m]).argmax(1)
            return out

    def memory(self) -> dict[str, int]:
        return {"params": sum(count_floats(m) for m in self.models.values()), "other": self.router.floats()}
