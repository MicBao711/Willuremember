import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vcm import Decision
from vcm.cl.data import Benchmark, Task
from vcm.cl.learners import (EWC, FineTune, GraceCL, Independent, Replay, TrainConfig,
                             VCMConfig, VCMLearner)
from vcm.cl.models import mlp, role_macs
from vcm.cl.runner import run, summarize


def toy_benchmark(n_tasks=3, dim=20, n=600, task_il=False, seed=0) -> Benchmark:
    """Gaussian blobs; each task permutes the input features (a tiny Permuted MNIST)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 2.0, (4, dim))
    tasks = []
    for t in range(n_tasks):
        perm = np.arange(dim) if t == 0 else rng.permutation(dim)

        def sample(m):
            y = rng.integers(0, 4, m)
            x = centers[y] + rng.normal(0, 1.0, (m, dim)) + 1.0
            return torch.tensor(x[:, perm], dtype=torch.float32), torch.tensor(y)

        (xtr, ytr), (xva, yva), (xte, yte) = sample(n), sample(200), sample(200)
        tasks.append(Task(t, f"toy{t}", xtr, ytr, xva, yva, xte, yte,
                          classes=["a", "b", "c", "d"], groups=["g"] * 4))
    return Benchmark("toy", tasks, (dim,), 4, task_id_at_test=task_il)


ARCH = mlp(in_dim=20, hidden=32, n_out=4)
CFG = TrainConfig(epochs=3, batch_size=32, lr=3e-3)


def test_role_macs():
    assert role_macs(ARCH) == {"in": 20 * 32, "hidden": 32 * 32, "out": 32 * 4}


def test_summarize_metrics():
    R = np.array([[0.9, np.nan], [0.7, 0.8]])
    m = summarize(R)
    assert m["acc"] == pytest.approx(0.75) and m["bwt"] == pytest.approx(-0.2)
    assert m["forgetting"] == pytest.approx(0.2) and m["old_task_curve"] == [pytest.approx(0.7)]


@pytest.mark.parametrize("cls,kw", [
    (FineTune, {}), (EWC, {"lam": 10.0}), (Replay, {"per_task": 50}),
    (Independent, {}), (GraceCL, {"max_keys": 50}),
])
@pytest.mark.parametrize("task_il", [False, True])
def test_baselines_run(cls, kw, task_il):
    bench = toy_benchmark(task_il=task_il)
    learner = cls(ARCH, task_il, CFG, seed=0, **kw)
    out = run(bench, learner, serve_batches=2, serve_size=20, log=lambda s: None)
    assert np.isfinite(out["metrics"]["acc"]) and out["metrics"]["acc"] > 0.3
    assert out["compute"]["total"] > 0 and sum(out["memory"].values()) > 0


def test_vcm_freeze_never_forgets():
    bench = toy_benchmark(task_il=True)
    learner = VCMLearner(ARCH, True, CFG, seed=0, config=VCMConfig(share_mode="freeze", probe_steps=10))
    out = run(bench, learner, serve_batches=3, serve_size=20, log=lambda s: None)
    R = np.array(out["R"])
    for j in range(3):
        assert np.allclose(R[j:, j], R[j, j])  # bit-identical answers after later tasks
    assert out["info"]["retained_old_snapshots"] == 0
    assert out["info"]["cache"]["stale"] == 0


def test_vcm_routes_hidden_task_variable():
    bench = toy_benchmark(task_il=False)
    learner = VCMLearner(ARCH, False, CFG, seed=0, config=VCMConfig(probe_steps=10))
    out = run(bench, learner, serve_batches=2, serve_size=20, log=lambda s: None)
    assert out["router_acc"] > 0.6  # chance is 1/3; the toy tasks overlap heavily


def _update_learner(**cfg):
    bench = toy_benchmark(task_il=True)
    c = VCMConfig(share_mode="update", probe_steps=10, semantic_threshold=0.0, form_threshold=0.0,
                  parsimony=1.0, **cfg)  # parsimony=1: always pick the plan with most reuse
    learner = VCMLearner(ARCH, True, CFG, seed=0, config=c)
    for t in bench.tasks[:2]:
        learner.learn(t)
    return bench, learner


def test_vcm_lazy_detects_staleness_only_on_use():
    bench, learner = _update_learner(maintenance="lazy")
    shared = [u for u, users in learner.users.items() if users == {0, 1}]
    if not shared:
        pytest.skip("task 1 did not reuse task 0's units on this seed")
    entry = learner.cache.peek(0)
    assert learner.cache.stale_units(entry)               # task 0 is stale ...
    assert sum(learner.decisions.values()) == 0           # ... but nothing was checked yet
    before = {k: v for k, v in learner.decisions.items()}
    view = learner.evaluate(bench.tasks[0].x_test, 0)     # dry run: no state change
    assert learner.decisions == before and learner.cache.stale_units(entry)
    served = learner.predict(bench.tasks[0].x_test, 0)    # real use triggers the judge
    assert sum(learner.decisions.values()) == 1
    assert torch.equal(view, served) or learner.decisions["repair"] == 1
    assert learner.cache.is_fresh(learner.cache.peek(0))


def test_vcm_keep_decision_retains_pinned_snapshot():
    bench, learner = _update_learner(maintenance="lazy", tolerance=-1.0, repair_steps=0)
    shared = [u for u, users in learner.users.items() if users == {0, 1}]
    if not shared:
        pytest.skip("task 1 did not reuse task 0's units on this seed")
    before = learner.predict(bench.tasks[0].x_test, 0)
    assert learner.decisions["keep"] == 1                 # impossible tolerance: keep snapshot
    assert learner.store.retained_old_snapshots()         # old version stays pinned
    old_payload = learner.cache.peek(0).value
    again = learner.predict(bench.tasks[0].x_test, 0)
    assert torch.equal(before, again) and learner.cache.peek(0).value is old_payload


def test_vcm_eager_checks_right_after_update():
    _, learner = _update_learner(maintenance="eager")
    shared = [u for u, users in learner.users.items() if users == {0, 1}]
    if not shared:
        pytest.skip("task 1 did not reuse task 0's units on this seed")
    assert sum(learner.decisions.values()) == 1


def test_vcm_promotion_creates_concept():
    bench = toy_benchmark(n_tasks=4, task_il=True)
    c = VCMConfig(share_mode="freeze", probe_steps=10, semantic_threshold=0.0, form_threshold=1.1,
                  promotion=True, promote_min_hits=2, promote_min_stable=2, parsimony=1.0)
    learner = VCMLearner(ARCH, True, CFG, seed=0, config=c)
    run(bench, learner, serve_batches=10, serve_size=10, log=lambda s: None)
    if not learner.parts:
        pytest.skip("no chunk was shared by two tasks on this seed")
    concept = next(iter(learner.parts))
    assert learner.store.get(concept).kind.value == "concept"
    assert learner.store.level(concept) == 1
