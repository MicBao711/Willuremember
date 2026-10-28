"""GRACE-CL: GRACE's discrete key-value codebook (Hartvigsen et al., NeurIPS 2023),
adapted from model editing to continual classification.

GRACE edits a frozen model by wrapping one layer with a codebook. A key is the layer's
input for an edited example, its value replaces the layer's output, and each key has a
deferral radius eps: inputs within eps of their nearest key use that key's value.
Codebook rules for an edit (h, y), with nearest key i at distance d, as in the official
implementation (github.com/Thartvigsen/GRACE, grace/editors/grace.py, eps_expand="coverage"):
  d > eps_i + eps_init       -> add key (h, value, eps_init, y)
  same label as key i        -> expand: eps_i = max(eps_i, d)
  different label            -> split:  eps_i = d / 2 - 1e-5, add key with eps = d / 2
Values are warm-started from the layer's own output, as in GRACE's "warm" value init.

Adaptation choices (ours, not from the paper):
* the base model is trained normally on the first task, then frozen;
* every later training example the current model gets wrong is an edit (one pass,
  capped at `max_keys` new keys per task); values are optimised by gradient descent so
  the frozen head outputs the key's label;
* eps_init = eps_scale x median nearest-neighbour distance of the task's keys;
* task-incremental benchmarks need new output heads, so a head per task is trained on
  the frozen features first, and the codebook is restricted to the task's own keys.
"""

from __future__ import annotations

import torch
from torch import nn

from ..data import Task
from ..models import Stack
from .base import Learner, ce, count_floats, train_steps


class GraceCL(Learner):
    name = "grace"

    def __init__(self, *args, eps_scale: float = 1.0, max_keys: int = 2000,
                 value_steps: int = 30, value_lr: float = 0.1, **kw):
        super().__init__(*args, **kw)
        if self.arch.templated:
            raise ValueError("GRACE-CL wraps one layer of a fixed architecture; templates are not supported")
        self.model = Stack(self.arch, multi_head=self.task_il)
        self.layer = self.arch.roles[-2]
        self.eps_scale, self.max_keys = eps_scale, max_keys
        self.value_steps, self.value_lr = value_steps, value_lr
        self.keys = torch.empty(0)
        self.values = torch.empty(0)
        self.eps = torch.empty(0)
        self.labels = torch.empty(0, dtype=torch.long)
        self.key_task = torch.empty(0, dtype=torch.long)
        self.edits = 0
        roles = self.arch.roles
        self.pre_macs = sum(self.macs[r] for r in roles[:-2])
        self.post_macs = self.macs[roles[-2]] + self.macs[roles[-1]]

    # -- forward pieces -----------------------------------------------------------
    def _key_input(self, x: torch.Tensor) -> torch.Tensor:
        for r in self.arch.roles[:-2]:
            x = self.model.body[r](x)
        return x.flatten(1)

    def _layer(self, h: torch.Tensor) -> torch.Tensor:
        shape = (len(h),) + tuple(self._key_shape)
        return self.model.body[self.layer](h.view(shape))

    def _lookup(self, h: torch.Tensor, task_id: int | None):
        if len(self.keys) == 0:
            return torch.zeros(len(h), dtype=torch.bool), torch.zeros(len(h), dtype=torch.long)
        d = torch.cdist(h, self.keys)
        if task_id is not None:
            d[:, self.key_task != task_id] = float("inf")
        dmin, idx = d.min(1)
        return dmin <= self.eps[idx], idx

    def _head_id(self, task_id):
        return task_id if self.task_il else None

    def _logits(self, x: torch.Tensor, task_id: int | None) -> torch.Tensor:
        h = self._key_input(x)
        out = self._layer(h)
        hit, idx = self._lookup(h, task_id if self.task_il else None)
        if hit.any():
            out = out.clone()
            out[hit] = self.values[idx[hit]]
        return self.model.head(self._head_id(task_id))(out)

    # -- learning -------------------------------------------------------------------
    def learn(self, task: Task) -> None:
        self.model.add_head(task.id)
        if not self.seen:
            self.model.train()
            train_steps(self.model.parameters(), lambda idx: self._full_loss(task, idx),
                        self.batches(len(task.x_train)), self.cfg.lr)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            with torch.no_grad():
                self._key_shape = self._shape_before_layer(task.x_train[:1])
            self.seen.append(task.id)
            return
        if self.task_il:
            head = self.model.head(task.id)
            for p in head.parameters():
                p.requires_grad_(True)
            train_steps(head.parameters(), lambda idx: self._head_loss(task, idx),
                        self.batches(len(task.x_train)), self.cfg.lr)
            for p in head.parameters():
                p.requires_grad_(False)
        self._edit_pass(task)
        self.seen.append(task.id)

    def _shape_before_layer(self, x):
        for r in self.arch.roles[:-2]:
            x = self.model.body[r](x)
        return x.shape[1:]

    def _full_loss(self, task, idx):
        self.meter.add("train", self.full_macs, len(idx), backward=True)
        return ce(self.model(task.x_train[idx], self._head_id(task.id)), task.y_train[idx])

    def _head_loss(self, task, idx):
        self.meter.add("train", self.full_macs, len(idx))  # frozen body: forward only
        self.meter.add("train", self.macs[self.arch.head], len(idx), backward=True)
        with torch.no_grad():
            f = self.model.features(task.x_train[idx])
        return ce(self.model.head(task.id)(f), task.y_train[idx])

    def _edit_pass(self, task: Task) -> None:
        head = self.model.head(self._head_id(task.id))
        tid = task.id if self.task_il else None
        with torch.no_grad():
            sample = self._key_input(task.x_train[:1000])
            nn_d = torch.cdist(sample, sample)
            nn_d.fill_diagonal_(float("inf"))
            eps_init = self.eps_scale * nn_d.min(1).values.median().item()
        added = 0
        for idx in self.batches(len(task.x_train), epochs=1):
            if added >= self.max_keys:
                break
            x, y = task.x_train[idx], task.y_train[idx]
            with torch.no_grad():
                h = self._key_input(x)
                pred = self._logits(x, task.id).argmax(1)
                base_out = self._layer(h)
            self.meter.add("edit", self.full_macs, len(idx))
            touched = set()
            for j in (pred != y).nonzero().flatten().tolist():
                if added >= self.max_keys:
                    break
                added += self._insert(h[j], base_out[j], int(y[j]), eps_init, task.id, tid, touched)
            if touched:
                self._fit_values(sorted(touched), head)
        self.edits += added

    def _insert(self, h, v, y, eps_init, task_id, tid, touched) -> int:
        candidates = torch.arange(len(self.keys))
        if tid is not None and len(self.keys):
            candidates = candidates[self.key_task == tid]
        if len(candidates):
            d = torch.norm(self.keys[candidates] - h, dim=1)
            j = int(d.argmin())
            dist, i = float(d[j]), int(candidates[j])
            if dist <= self.eps[i] + eps_init:
                if int(self.labels[i]) == y:
                    self.eps[i] = max(float(self.eps[i]), dist)
                    return 0
                self.eps[i] = dist / 2 - 1e-5
                touched.add(i)
                return self._append(h, v, y, dist / 2, task_id, touched)
        return self._append(h, v, y, eps_init, task_id, touched)

    def _append(self, h, v, y, eps, task_id, touched) -> int:
        cat = lambda a, b: torch.cat([a, b.unsqueeze(0)]) if len(a) else b.unsqueeze(0)
        self.keys = cat(self.keys, h)
        self.values = cat(self.values, v)
        self.eps = cat(self.eps, torch.tensor(float(eps)))
        self.labels = cat(self.labels, torch.tensor(y))
        self.key_task = cat(self.key_task, torch.tensor(task_id))
        touched.add(len(self.keys) - 1)
        return 1

    def _fit_values(self, idx: list[int], head: nn.Module) -> None:
        idx = torch.tensor(idx)
        v = self.values[idx].clone().requires_grad_(True)
        opt = torch.optim.Adam([v], lr=self.value_lr)
        for _ in range(self.value_steps):
            opt.zero_grad()
            ce(head(v), self.labels[idx]).backward()
            opt.step()
        self.meter.add("edit", self.macs[self.arch.head], len(idx) * self.value_steps, backward=True)
        self.values[idx] = v.detach()

    def predict(self, x: torch.Tensor, task_id: int | None) -> torch.Tensor:
        with torch.no_grad():
            return self._logits(x, task_id).argmax(1)

    def memory(self) -> dict[str, int]:
        book = self.keys.numel() + self.values.numel() + 2 * len(self.eps)
        return {"params": count_floats(self.model), "other": book}

    def info(self) -> dict:
        return {"keys": len(self.keys), "edits": self.edits}
