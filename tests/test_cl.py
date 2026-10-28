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


def test_vcm_dry_run_repair_works_inside_no_grad_and_changes_nothing():
    bench, learner = _update_learner(maintenance="lazy", tolerance=-1.0, repair_steps=100)
    shared = [u for u, users in learner.users.items() if users == {0, 1}]
    if not shared:
        pytest.skip("task 1 did not reuse task 0's units on this seed")
    versions = {u.id: u.version for u in learner.store}
    with torch.no_grad():  # the runner measures accuracy like this
        view = learner.evaluate(bench.tasks[0].x_test, 0)
    assert {u.id: u.version for u in learner.store} == versions
    assert sum(learner.decisions.values()) == 0
    pinned = learner.cache.peek(0).value
    with torch.no_grad():
        assert torch.equal(view, pinned(bench.tasks[0].x_test).argmax(1))  # impossible bar: keep


# -- templates: single-image tasks first, then tasks over two images -------------------
from vcm.cl.models import compose_mlp, template_macs  # noqa: E402

CARCH = compose_mlp(item_dim=20, hidden=32)


def toy_compose(n=600, seed=0) -> Benchmark:
    """Four Gaussian "digits"; task 0 reads one, tasks 1-2 answer sum / equal for a pair."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 2.0, (4, 20))

    def items(y):
        return torch.tensor(centers[y] + rng.normal(0, 1.0, y.shape + (20,)) + 1.0, dtype=torch.float32)

    def unary(m):
        y = rng.integers(0, 4, m)
        return items(y), torch.tensor(y)

    def pairs(m, label):
        a, b = rng.integers(0, 4, m), rng.integers(0, 4, m)
        return items(np.stack([a, b], 1)), torch.tensor(label(a, b))

    s = lambda a, b: a + b
    e = lambda a, b: (a == b).astype(np.int64)
    tasks = [Task(0, "read", *unary(n), *unary(200), *unary(200), classes=list("abcd"),
                  groups=["read"] * 4, template="unary")]
    for t, (name, f, k) in enumerate((("sum", s, 7), ("equal", e, 2)), start=1):
        tasks.append(Task(t, name, *pairs(n, f), *pairs(200, f), *pairs(200, f),
                          classes=[f"{name}{i}" for i in range(k)], groups=[name] * k, template="binary"))
    return Benchmark("toy_compose", tasks, (20,), 7, task_id_at_test=True)


def test_templates_shapes_and_cost():
    net = CARCH.network("binary", 7)
    assert net(torch.zeros(5, 2, 20)).shape == (5, 7)   # one reader, applied to both items
    assert template_macs(CARCH, "unary", 4) == 20 * 32 + 32 * 32 + 32 * 4
    assert template_macs(CARCH, "binary", 7) == 2 * (20 * 32 + 32 * 32) + 64 * 32 + 32 * 7


@pytest.mark.parametrize("cls,kw", [(FineTune, {}), (EWC, {"lam": 10.0}), (Replay, {"per_task": 50}),
                                    (Independent, {})])
def test_baselines_run_on_templates(cls, kw):
    learner = cls(CARCH, True, CFG, seed=0, **kw)
    out = run(toy_compose(), learner, serve_batches=2, serve_size=20, log=lambda s: None)
    assert np.isfinite(out["metrics"]["acc"]) and out["metrics"]["acc"] > 0.3


def _compose_learner(**cfg):
    c = VCMConfig(**{"probe_steps": 10, "parsimony": 1.0, "promote_min_hits": 2, "promote_min_stable": 2, **cfg})
    return VCMLearner(CARCH, True, CFG, seed=0, config=c)


def test_composition_concept_fills_both_reader_slots():
    learner = _compose_learner(share_mode="freeze", promote_compositions=True)
    out = run(toy_compose(), learner, serve_batches=5, serve_size=20, log=lambda s: None)
    assert list(learner.origin.values()) == [0]           # the reading task was promoted ...
    concept = next(iter(learner.origin))
    assert learner.store.get(concept).kind.value == "concept" and learner.store.level(concept) == 1
    spec = learner.cache.peek(1).spec                      # ... and fills the pair task's reader
    assert spec[0].unit == concept and (spec[0].start, spec[0].stop) == (0, 2)
    assert learner.sig[concept] == (20, 4) and 1 in learner.users[concept]
    R = np.array(out["R"])
    assert np.allclose(R[:, 0], R[0, 0])                   # freeze: the reading task never changes


def test_without_promotion_the_head_stays_on_top():
    learner = _compose_learner(share_mode="freeze")
    run(toy_compose(), learner, serve_batches=5, serve_size=20, log=lambda s: None)
    assert not learner.parts
    for k in (1, 2):   # layer-wise reuse may take the reader, never task 0's output layer
        units = [s.unit for s in learner.cache.peek(k).spec]
        assert "t0:out" not in units and units[-1] == f"t{k}:out"
    assert any(s.unit == "t0:in" for s in learner.cache.peek(1).spec)


def test_zero_thresholds_promote_at_commit():
    learner = _compose_learner(share_mode="freeze", promote_compositions=True,
                               promote_min_hits=0, promote_min_stable=0)
    learner.learn(toy_compose().tasks[0])
    assert list(learner.origin.values()) == [0]


@pytest.mark.parametrize("mode,follows", [("cut_deps", False), ("keep_deps", True)])
def test_composition_concept_modes(mode, follows):
    bench = toy_compose()
    learner = _compose_learner(share_mode="freeze", promote_compositions=True, promotion_mode=mode)
    learner.learn(bench.tasks[0])
    for _ in range(3):
        learner.predict(bench.tasks[0].x_test, 0)          # used often and stable -> promoted
    learner.learn(bench.tasks[1])
    concept = next(iter(learner.origin))
    entry = learner.cache.peek(1)
    assert entry.spec[0].unit == concept
    learner.store.update("t0:in", learner.mods["t0:in"].state_dict())
    # keep_deps depends on the parts and goes stale with them; cut_deps is a frozen copy
    assert bool(learner.cache.stale_units(entry)) is follows
    assert (concept in learner.follows) is follows


# -- staged selection: candidates keep training instead of being judged after one probe --
def _staged(selection, **cfg):
    bench = toy_compose()
    c = VCMConfig(**{"probe_steps": 4, "selection": selection, "share_mode": "freeze",
                     "promote_compositions": True, "promote_min_hits": 0, "promote_min_stable": 0, **cfg})
    learner = VCMLearner(CARCH, True, CFG, seed=0, config=c)
    for t in bench.tasks[:2]:
        learner.learn(t)
    return learner, learner.log[-1], learner.n_steps(len(bench.tasks[1].x_train))


def test_halving_schedule_and_cost():
    learner, entry, total = _staged("halving", halving_eta=2)
    steps = sorted(entry["probe_steps"].values())
    assert len(steps) >= 3 and steps[-1] == total         # the survivor trains the whole budget
    alive, budget, expected = len(steps), 4, []           # keep ceil(n / 2), double the budget
    while alive > 1:
        keep = -(-alive // 2)
        expected += [budget] * (alive - keep)
        alive, budget = keep, budget * 2
    assert steps[:-1] == sorted(expected)
    t0, t1 = toy_compose().tasks[:2]
    trained = learner.cost(t0) * learner.n_steps(len(t0.x_train)) + learner.cost(t1) * sum(steps)
    assert learner.meter.counts["train"] == pytest.approx(3 * CFG.batch_size * trained)


def test_finalists_are_the_best_after_the_probe_and_train_fully():
    learner, entry, total = _staged("finalists", finalists=2, curve_every=4, probe_val=100)
    steps = entry["probe_steps"]
    finalists = {s for s, n in steps.items() if n == total}
    rest = set(steps) - finalists
    assert len(finalists) == 2 and rest and {steps[s] for s in rest} == {4}
    at_probe = {s: curve[0][1] for s, curve in entry["curves"].items()}   # selection split, step 4
    assert min(at_probe[s] for s in finalists) >= max(at_probe[s] for s in rest)
    assert entry["plan"] in finalists
    t0, t1 = toy_compose().tasks[:2]
    trained = learner.cost(t0) * learner.n_steps(len(t0.x_train)) + learner.cost(t1) * sum(steps.values())
    assert learner.meter.counts["train"] == pytest.approx(3 * CFG.batch_size * trained)


def test_full_selection_trains_every_candidate_and_logs_curves():
    learner, entry, total = _staged("full", curve_every=5, probe_val=100)   # held-out: val[100:200]
    assert set(entry["probe_steps"].values()) == {total}
    for curve in entry["curves"].values():
        assert [c[0] for c in curve][-1] == total and all(0 <= c[1] <= 1 and 0 <= c[2] <= 1 for c in curve)
    chosen = entry["plan"]
    best = max(c[1] for c in entry["candidates"])
    assert dict((c[0], c[1]) for c in entry["candidates"])[chosen] >= best - learner.c.parsimony


# -- MNTDP-D baseline -----------------------------------------------------------------
from vcm.cl.learners import MNTDP  # noqa: E402
from vcm.cl.learners.mntdp import knn_accuracy  # noqa: E402


def test_knn_accuracy_majority_vote():
    train_x = torch.tensor([[0.0], [0.1], [0.2], [5.0], [5.1]])
    train_y = torch.tensor([0, 0, 1, 1, 1])
    test_x, test_y = torch.tensor([[0.05], [5.05]]), torch.tensor([0, 1])
    assert knn_accuracy(train_x, train_y, test_x, test_y, k=3) == 1.0
    assert knn_accuracy(train_x, train_y, test_x, test_y, k=5) == 0.5   # 3 ones vs 2 zeros everywhere


def test_mntdp_reuses_prefixes_of_the_most_similar_task_only():
    bench = toy_compose()
    learner = MNTDP(CARCH, True, CFG, seed=0)
    out = run(bench, learner, serve_batches=2, serve_size=20, log=lambda s: None)
    R = np.array(out["R"])
    assert np.allclose(R[:, 0], R[0, 0])                   # frozen modules: no forgetting
    for entry in learner.log[1:]:
        k = entry["task"]
        scores = learner.prior[k]
        src = max(reversed(learner.seen[:learner.seen.index(k)]), key=lambda j: scores.get(j, -1.0))
        steps = set(entry["probe_steps"].values())
        assert len(steps) == 1                             # MNTDP-D trains every path fully
        names = [c[0] for c in entry["candidates"]]
        assert "scratch" in names and all(n == "scratch" or n.endswith(f"<-t{src}") for n in names)
        spec = learner.cache.peek(k).spec
        assert spec[-1].unit == f"t{k}:out"                # the output layer is always new
    assert not learner.parts and not learner.origin        # no promotion, no concepts
    # a two-image task can take a digit network's reader layers, never its output layer
    assert "t0:out" not in [s.unit for s in learner.cache.peek(1).spec]
