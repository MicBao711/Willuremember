"""Protocol: for each task, learn it, serve a stream of queries, then measure all seen tasks.

The query stream is what makes lazy maintenance observable. Queries favour recent tasks
(probability ~ 1 / (age + 1)); a learner with lazy invalidation only checks the tasks
that are actually queried. Measurement afterwards must not change the learner, so it
calls `evaluate`, which for VCM answers "what would be returned if asked now" without
committing any repair.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .data import Benchmark
from .learners.base import Learner, accuracy


def summarize(R: np.ndarray) -> dict:
    T = R.shape[0]
    diag = np.diag(R)
    final = R[-1]
    out = {
        "acc": float(np.nanmean(final)),
        "learning_acc": float(np.nanmean(diag)),
        "new_task_curve": diag.tolist(),
        "old_task_curve": [float(np.nanmean(R[i, :i])) for i in range(1, T)],
        "seen_task_curve": [float(np.nanmean(R[i, : i + 1])) for i in range(T)],
    }
    if T > 1:
        out["bwt"] = float(np.mean(final[:-1] - diag[:-1]))
        out["forgetting"] = float(np.mean([np.nanmax(R[j:-1, j]) - final[j] for j in range(T - 1)]))
    return out


def run(bench: Benchmark, learner: Learner, serve_batches: int = 20, serve_size: int = 100,
        seed: int = 0, log=print, eval_split: str = "test", settle: bool = False) -> dict:
    """`eval_split="val"` measures on held-out training data (for hyperparameter selection).
    `settle=True` asks a learner with pending lazy checks to run them all after the last
    measurement, and records what it retains then (`out["settled"]`); nothing else changes."""
    T = len(bench.tasks)
    R = np.full((T, T), np.nan)
    rng = np.random.default_rng(seed + 1234)
    history = []
    start = time.time()
    tid = lambda t: t.id if bench.task_id_at_test else None
    evaluate = getattr(learner, "evaluate", learner.predict)
    for i, task in enumerate(bench.tasks):
        t0 = time.time()
        learner.learn(task)
        train_time = time.time() - t0
        weights = 1.0 / (np.arange(i, -1, -1) + 1.0)
        weights /= weights.sum()
        for _ in range(serve_batches):
            j = int(rng.choice(i + 1, p=weights))
            t = bench.tasks[j]
            idx = torch.tensor(rng.choice(len(t.x_test), serve_size, replace=False))
            learner.predict(t.x_test[idx], tid(t))
            learner.meter.add("serve", learner.cost(t), serve_size)
        if hasattr(task, "release_train"):
            task.release_train()
        for j in range(i + 1):
            t = bench.tasks[j]
            x, y = (t.x_val, t.y_val) if eval_split == "val" else (t.x_test, t.y_test)
            R[i, j] = accuracy(lambda b: evaluate(b, tid(t)), x, y)
        mem = learner.memory()
        history.append({"task": i, "train_time": train_time, "memory": mem,
                        "compute": learner.meter.as_dict()})
        old = np.nanmean(R[i, :i]) if i else float("nan")
        log(f"  [{learner.name}] task {i}: new={R[i, i]:.4f} old={old:.4f} "
            f"mem={sum(mem.values()) / 1e6:.2f}M macs={learner.meter.total / 1e9:.1f}G "
            f"({train_time:.1f}s)")
    out = {"R": R.tolist(), "metrics": summarize(R), "history": history,
           "compute": learner.meter.as_dict(), "memory": learner.memory(),
           "time": time.time() - start, "info": learner.info()}
    router = getattr(learner, "router", None)
    if router is not None and router.ids and not bench.task_id_at_test:
        hits = [float((router.route(t.x_test) == t.id).float().mean()) for t in bench.tasks]
        out["router_acc"] = float(np.mean(hits))
    if settle and hasattr(learner, "settle"):
        out["settled"] = learner.settle()
    return out
