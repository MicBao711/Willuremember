"""MNTDP-D (Veniat, Denoyer & Ranzato, "Efficient Continual Learning with Modular Networks
and Task-Driven Priors", ICLR 2021), on the role slots used by the rest of this package.

Following the official implementation (github.com/TomVeniat/MNTDP: PSSN_llmodel.py,
ExhaustiveSearch.py; configuration "PSSN-search-6-fw", i.e. strat=search_all,
n_source_models=1, forward lateral connections):
* a task's network is a path with one module per layer; modules of earlier tasks are frozen
* task-driven prior: each earlier task's network without its output layer embeds up to
  1000 training samples of the new task; a 15-nearest-neighbour classifier fitted on them
  is scored on up to 1000 validation samples; the best-scoring task is the only source
* search space: the source's first k modules (k = 0 .. all but the output layer), new
  modules above them; the output layer is always new
* MNTDP-D: every path is trained and the one with the best validation accuracy is kept

Adaptation choices (ours):
* every path trains the same number of steps as one network of the other methods here
  (the official code trains each path to convergence with early stopping and a small
  learning-rate / weight-decay grid)
* modules are this package's role slots. With templates, a source's prefix is used (for
  the prior and for reuse) only as far as the new template has the same roles at the same
  positions, e.g. a two-image task can reuse a digit network's reader layers
* the prior's cost is counted: the source prefixes' forward passes and the
  nearest-neighbour distances (one multiply-accumulate per feature per pair)
* tasks are inferred at test time the same way as for VCM when no task id is given

It reuses VCM's machinery for storing frozen units and serving compositions; with frozen
units nothing ever goes stale, so no version check, judge, cache promotion or concept
is involved.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from ...similarity import SimilarityProfile
from ..data import Task
from ..models import module_macs
from .vcm_learner import Plan, Slot, VCMConfig, VCMLearner


def knn_accuracy(train_x, train_y, test_x, test_y, k: int = 15) -> float:
    """Euclidean k-nearest-neighbour majority vote (ties -> smallest label), like sklearn's default."""
    k = min(k, len(train_x))
    nearest = torch.cdist(test_x, train_x).topk(k, largest=False).indices
    votes = F.one_hot(train_y[nearest], int(train_y.max()) + 1).sum(1)
    return (votes.argmax(1) == test_y).float().mean().item()


class MNTDP(VCMLearner):
    name = "mntdp"

    def __init__(self, *args, n_neighbors: int = 15, prior_samples: int = 1000, config: VCMConfig | None = None, **kw):
        config = config or VCMConfig()
        config.share_mode, config.selection, config.parsimony = "freeze", "full", 0.0
        config.promotion = config.promote_compositions = False
        config.feedback_per_task = 0          # nothing goes stale, so no judge needs feedback
        super().__init__(*args, config=config, **kw)
        self.n_neighbors, self.prior_samples = n_neighbors, prior_samples
        self.prior: dict[int, dict[int, float]] = {}

    def _prefix(self, k: int, task: Task) -> int:
        """Slots of task k that the new task can reuse in place: its longest aligned prefix
        that ends on a slot boundary and leaves out k's output layer."""
        slots, tpl = self.cache.peek(k).spec, self.arch.template(task.template)
        best = 0
        for i, s in enumerate(slots[:-1]):
            if self.tmpl[k].roles[s.start:s.stop] != tpl.roles[s.start:s.stop] or s.stop >= len(tpl.roles):
                break
            best = i + 1
        return best

    def _candidates(self, task: Task, prof: dict) -> list[Plan]:
        tpl = self.arch.template(task.template)
        n = len(tpl.roles)
        new = lambda a, b: tuple(Slot(i, i + 1, None) for i in range(a, b))
        m = self.prior_samples
        tx, ty = task.x_train[:m], task.y_train[:m]
        vx, vy = task.x_val[:m], task.y_val[:m]
        scores = {}
        for k in self.seen:
            p = self._prefix(k, task)
            if p == 0:
                continue
            body = self.cache.peek(k).value[:p]
            with torch.no_grad():
                ft, fv = body(tx).flatten(1), body(vx).flatten(1)
            macs, _ = module_macs(body, tx[:1])
            self.meter.add("prior", macs, len(tx) + len(vx))
            self.meter.add("prior", ft.shape[1] * len(tx), len(vx))   # distances
            scores[k] = knn_accuracy(ft, ty, fv, vy, self.n_neighbors)
        self.prior[task.id] = scores
        plans = [Plan(new(0, n), "scratch", 0.0)]
        if scores:
            src = max(reversed(self.seen), key=lambda k: scores.get(k, -1.0))   # ties: most recent
            slots = self.cache.peek(src).spec
            for i in range(self._prefix(src, task)):
                keep = slots[: i + 1]
                sim = SimilarityProfile({"knn": scores[src]})
                plans.append(Plan(keep + new(keep[-1].stop, n), f"prefix{keep[-1].stop}<-t{src}",
                                  scores[src], {s.unit: sim for s in keep}))
        return [p for p in plans if self._fits(p, task)]

    def info(self) -> dict:
        return {**super().info(), "prior": self.prior}
