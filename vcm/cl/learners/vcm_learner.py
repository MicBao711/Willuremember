"""VCM for neural continual learning.

Mapping from the README to networks:

* base unit         one role slot of a network (e.g. the input layer), stored in a
                    `UnitStore` as a state-dict snapshot with a version number
* domain            the task that created the unit
* pointer           reuse of another task's unit, pinned to a snapshot version and
                    annotated with a multi-dimensional similarity profile:
                      form      input-statistics similarity of the two tasks
                      semantic  overlap of label names
                      category  overlap of coarse label groups (e.g. CIFAR-100 superclasses)
                      function  validation accuracy of the composition that reused it
* role binding      a task's network is a sequence of slots, each bound to one unit
                    (never a weighted average of several units)
* conditional var.  the task itself is the hidden variable; at test time without a task
                    label it is inferred from the input (nearest input centroid)
* cache             each task's composed network, recorded with the unit versions it
                    was built from (`CompositionCache`)
* lazy invalidation a shared unit may be updated in place by a later task (version +1).
                    Old tasks are not touched; when one is next used, the version check
                    fails and a judge (accuracy on a small feedback buffer) decides:
                      accept the new versions | repair the task's private units | keep
                    the pinned snapshot
* promotion         a run of shared units that many uses have hit while staying valid is
                    promoted to a concept unit spanning several roles (cut_deps: frozen
                    copy; keep_deps: a named alias that follows its parts); concepts are
                    offered first when composing later tasks
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

from ...cache import CompositionCache, Decision
from ...promotion import PromotionMode
from ...similarity import Pointer, SimilarityProfile
from ...units import UnitKind, UnitStore
from ..data import Task
from .base import Learner, TaskRouter, accuracy, ce, train_steps


def clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in state.items()}


@dataclass
class VCMConfig:
    share_mode: str = "freeze"       # freeze: reused units are frozen | update: trained in place
    maintenance: str = "lazy"        # lazy: check on use | eager: check all right after an update
    shared_lr_scale: float = 0.1
    tolerance: float = 0.02          # allowed accuracy drop on the feedback buffer
    feedback_per_task: int = 200
    repair_steps: int = 100
    form_threshold: float = 0.3
    semantic_threshold: float = 0.5
    probe_steps: int = 50
    probe_val: int = 1000
    max_candidates: int = 5
    parsimony: float = 0.005         # prefer fewer new parameters within this accuracy margin
    promotion: bool = False
    promotion_mode: str = "cut_deps"
    promote_min_hits: int = 5
    promote_min_stable: int = 3
    promote_min_tasks: int = 2


@dataclass(frozen=True)
class Slot:
    start: int                       # first role index covered
    stop: int                        # one past the last role index
    unit: str | None                 # None: a new unit is created for this slot


@dataclass
class Plan:
    slots: tuple[Slot, ...]
    source: str
    prior: float
    sources: dict[str, SimilarityProfile] = field(default_factory=dict)
    concept: str | None = None


class Frozen(nn.Module):
    """Runs a template module with parameters taken from a snapshot (no copy)."""

    def __init__(self, template: nn.Module, payload: dict[str, torch.Tensor]):
        super().__init__()
        self.template = [template]   # list: keep it out of this module's parameters
        self.payload = payload

    def forward(self, x):
        return functional_call(self.template[0], self.payload, (x,))


class ChunkStats:
    def __init__(self):
        self.hits = 0
        self.stable = 0
        self.versions: tuple[int, ...] | None = None
        self.promoted: str | None = None


class VCMLearner(Learner):
    name = "vcm"

    def __init__(self, *args, config: VCMConfig | None = None, **kw):
        super().__init__(*args, **kw)
        self.c = config or VCMConfig()
        self.roles = self.arch.roles
        self.store = UnitStore(freeze_fn=clone_state)
        self.mods: dict[str, nn.Module] = {}          # live module (template) per unit
        self.span: dict[str, tuple[int, int]] = {}
        self.users: dict[str, set[int]] = {}
        self.parts: dict[str, tuple[str, ...]] = {}   # concept -> parts
        self.cache = CompositionCache(
            builder=self._build, version_of=self.store.version,
            on_insert=self._pin, on_remove=self._unpin, judge=self._judge,
        )
        self.router = TaskRouter()
        self.profiles: dict[int, dict] = {}
        self.feedback: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.ref_acc: dict[int, float] = {}
        self.pointers: dict[int, list[Pointer]] = {}
        self.plans: dict[int, str] = {}
        self.chunks: dict[tuple[str, ...], ChunkStats] = {}
        self.log: list[dict] = []
        self.decisions = {"accept": 0, "repair": 0, "keep": 0}
        self._views: dict[tuple, nn.Module] = {}   # memoised dry-run answers for evaluate()

    # -- similarity ---------------------------------------------------------------
    def _profile(self, task: Task) -> dict:
        x = task.x_train[:5000].flatten(1)
        return {"mean": x.mean(0), "std": x.std(0), "classes": set(task.classes),
                "groups": set(task.groups)}

    @staticmethod
    def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
        a, b = a - a.mean(), b - b.mean()
        denom = a.norm() * b.norm()
        return float((a @ b) / denom) if denom > 0 else 0.0

    def similarity(self, p: dict, q: dict) -> SimilarityProfile:
        form = 0.5 * (self._corr(p["mean"], q["mean"]) + self._corr(p["std"], q["std"]))
        jac = lambda a, b: len(a & b) / len(a | b) if a | b else 0.0
        return SimilarityProfile({
            "form": float(np.clip(form, 0.0, 1.0)),
            "semantic": jac(p["classes"], q["classes"]),
            "category": jac(p["groups"], q["groups"]),
        })

    # -- candidate compositions -------------------------------------------------------
    def _candidates(self, task: Task, prof: dict) -> list[Plan]:
        n = len(self.roles)
        new = lambda a, b: tuple(Slot(i, i + 1, None) for i in range(a, b))
        plans = [Plan(new(0, n), "scratch", 0.0)]
        sims = {k: self.similarity(prof, self.profiles[k]) for k in self.seen}
        # Lower roles carry form (input statistics); the head carries meaning (labels).
        # For every cut point, reuse from the task most similar on the relevant dimension.
        for side, dim, threshold in (("bottom", "form", self.c.form_threshold),
                                     ("top", "semantic", self.c.semantic_threshold)):
            for cut in range(1, n):
                best = None
                for k in reversed(self.seen):  # ties go to the most recent task
                    slots = self.cache.peek(k).spec
                    if sims[k][dim] < threshold or cut not in {s.start for s in slots}:
                        continue
                    if best is None or sims[k][dim] > sims[best][dim]:
                        best = k
                if best is None:
                    continue
                slots, sim = self.cache.peek(best).spec, sims[best]
                if side == "bottom":
                    keep = tuple(s for s in slots if s.stop <= cut)
                    slots_, prior = keep + new(cut, n), sim[dim] * cut / n
                else:
                    keep = tuple(s for s in slots if s.start >= cut)
                    slots_, prior = new(0, cut) + keep, sim[dim] * (n - cut) / n
                plans.append(Plan(slots_, f"{side}{cut}<-t{best}", prior, {s.unit: sim for s in keep}))
        for concept, (a, b) in ((c, self.span[c]) for c in self.parts):
            sims = [self.similarity(prof, self.profiles[u]) for u in self._concept_users(concept)]
            sim = max(sims, key=lambda s: s.blended()) if sims else SimilarityProfile()
            needs_form = a == 0 and sim["form"] < self.c.form_threshold
            needs_sem = b == n and sim["semantic"] < self.c.semantic_threshold
            if not (needs_form or needs_sem):
                if concept in self.mods:   # cut_deps: one frozen unit spanning the roles
                    inner = (Slot(a, b, concept),)
                else:                      # keep_deps: resolves to its live parts
                    inner = tuple(Slot(*self.span[u], u) for u in self.parts[concept])
                plans.append(Plan(new(0, a) + inner + new(b, n), f"concept {concept}",
                                  1.0 + self.chunks_of(concept).hits / 100,
                                  {s.unit: sim for s in inner}, concept))
        unique: dict[tuple, Plan] = {}
        for p in plans:
            key = tuple((s.start, s.stop, s.unit) for s in p.slots)
            if key not in unique or unique[key].prior < p.prior:
                unique[key] = p
        ranked = sorted(unique.values(), key=lambda p: -p.prior)
        scratch = [p for p in ranked if p.source == "scratch"]
        others = [p for p in ranked if p.source != "scratch"][: self.c.max_candidates - 1]
        return others + scratch

    def _concept_users(self, concept: str) -> set[int]:
        return self.users.get(concept, set()) | set.intersection(
            *(self.users.get(p, set()) for p in self.parts[concept]))

    def chunks_of(self, concept: str) -> ChunkStats:
        for stats in self.chunks.values():
            if stats.promoted == concept:
                return stats
        return ChunkStats()

    # -- instantiate & train ------------------------------------------------------------
    def _instantiate(self, plan: Plan, task: Task):
        modules, groups_new, groups_shared = [], [], []
        for s in plan.slots:
            if s.unit is None:
                m = self.arch.make(self.roles[s.start])
                groups_new += list(m.parameters())
            else:
                m = copy.deepcopy(self.mods[s.unit])
                trainable = (self.c.share_mode == "update" and s.unit not in self.parts)
                for p in m.parameters():
                    p.requires_grad_(trainable)
                if trainable:
                    groups_shared += list(m.parameters())
            modules.append(m)
        net = nn.Sequential(*modules)
        groups = [{"params": groups_new, "lr": self.cfg.lr},
                  {"params": groups_shared, "lr": self.cfg.lr * self.c.shared_lr_scale}]
        return net, groups

    def _train(self, net, groups, task, steps):
        def loss(idx):
            self.meter.add("train", self.full_macs, len(idx), backward=True)
            return ce(net(task.x_train[idx]), task.y_train[idx])
        net.train()
        train_steps(groups, loss, self.batches(len(task.x_train), steps=steps), lr=groups)
        net.eval()

    def learn(self, task: Task) -> None:
        prof = self._profile(task)
        plans = self._candidates(task, prof)
        total_steps = self.cfg.epochs * int(np.ceil(len(task.x_train) / self.cfg.batch_size))
        probe = min(self.c.probe_steps, total_steps) if len(plans) > 1 else 0
        vx, vy = task.x_val[: self.c.probe_val], task.y_val[: self.c.probe_val]
        results = []
        for plan in plans:
            net, groups = self._instantiate(plan, task)
            if probe:
                self._train(net, groups, task, probe)
                acc = accuracy(lambda x: net(x).argmax(1), vx, vy)
                self.meter.add("probe_eval", self.full_macs, len(vx))
            else:
                acc = float("nan")
            n_new = sum(p.numel() for s, m in zip(plan.slots, net) if s.unit is None for p in m.parameters())
            results.append((acc, n_new, plan, net, groups))
        if probe:
            best_acc = max(r[0] for r in results)
            ok = [r for r in results if r[0] >= best_acc - self.c.parsimony]
            acc, n_new, plan, net, groups = min(ok, key=lambda r: (r[1], -r[0]))
        else:
            acc, n_new, plan, net, groups = results[0]
        self._train(net, groups, task, total_steps - probe)
        self._commit(task, prof, plan, net, results)

    # -- commit -----------------------------------------------------------------------
    def _commit(self, task: Task, prof: dict, plan: Plan, net: nn.Sequential, results) -> None:
        spec, updated = [], []
        for s, m in zip(plan.slots, net):
            if s.unit is None:
                uid = f"t{task.id}:{self.roles[s.start]}"
                self.mods[uid] = m
                self.span[uid] = (s.start, s.stop)
                self.store.add(uid, m.state_dict(), domain=f"task{task.id}")
                self.users[uid] = set()
            else:
                uid = s.unit
                if self.c.share_mode == "update" and uid not in self.parts:
                    self.mods[uid].load_state_dict(m.state_dict())
                    self.store.update(uid, self.mods[uid].state_dict())
                    updated.append(uid)
            self.users[uid].add(task.id)
            spec.append(Slot(s.start, s.stop, uid))
        if plan.concept is not None:
            self.users[plan.concept].add(task.id)
        spec = tuple(spec)
        self.seen.append(task.id)
        self.profiles[task.id] = prof
        self.router.add(task.id, task.x_train)
        idx = torch.randperm(len(task.x_train), generator=self.gen)[: self.c.feedback_per_task]
        self.feedback[task.id] = (task.x_train[idx], task.y_train[idx])
        model = self.cache.get(task.id, spec)
        self.ref_acc[task.id] = self._feedback_acc(task.id, model, count=True)
        func_acc = {id(r[2]): r[0] for r in results}.get(id(plan), float("nan"))
        self.pointers[task.id] = [
            Pointer(s.unit, self.store.version(s.unit), "/".join(self.roles[s.start:s.stop]),
                    plan.sources.get(s.unit, SimilarityProfile()).merged(
                        SimilarityProfile({"function": float(np.nan_to_num(func_acc))})))
            for s, orig in zip(spec, plan.slots) if orig.unit is not None
        ]
        self.plans[task.id] = plan.source
        self.log.append({"task": task.id, "plan": plan.source, "updated": updated,
                         "candidates": [(r[2].source, r[0], r[1]) for r in results]})
        if self.c.maintenance == "eager":
            for k in list(self.seen):
                self.cache.refresh(k)

    # -- composition cache plumbing -----------------------------------------------------
    def _build(self, spec: tuple[Slot, ...]):
        deps, layers = {}, []
        for s in spec:
            deps[s.unit] = self.store.version(s.unit)
            layers.append(Frozen(self.mods[s.unit], self.store.snapshot(s.unit).payload))
        return nn.Sequential(*layers), deps

    def _pin(self, entry):
        for u, v in entry.deps.items():
            self.store.pin(u, v)

    def _unpin(self, entry):
        for u, v in entry.deps.items():
            self.store.unpin(u, v)

    def _feedback_acc(self, task_id, model, count: bool, part: slice = slice(None)) -> float:
        x, y = self.feedback[task_id]
        x, y = x[part], y[part]
        if count:
            self.meter.add("maintenance", self.full_macs, len(x))
        return accuracy(lambda b: model(b).argmax(1), x, y)

    def _private(self, task_id: int, spec) -> list[Slot]:
        return [s for s in spec if s.unit not in self.parts and self.users[s.unit] == {task_id}]

    def _judge(self, entry, stale) -> Decision:
        """External feedback on a stale composition: accept, repair, or keep the snapshot."""
        k = entry.key
        fresh, _ = self._build(entry.spec)
        if self._feedback_acc(k, fresh, count=True) >= self.ref_acc[k] - self.c.tolerance:
            self.decisions["accept"] += 1
            return Decision.RECOMPOSE
        repaired = self._repair(k, entry.spec, commit=True)
        if repaired is not None:
            self.decisions["repair"] += 1
            return Decision.RECOMPOSE
        self.decisions["keep"] += 1
        return Decision.KEEP

    def _repair(self, k: int, spec, commit: bool):
        """Retrain task k's private units against the current shared units."""
        private = self._private(k, spec)
        if not private or self.c.repair_steps <= 0:
            return None
        x, y = self.feedback[k]
        n_fit = int(len(x) * 2 / 3)
        layers, params = [], []
        for s in spec:
            if s in private:
                m = copy.deepcopy(self.mods[s.unit])
                for p in m.parameters():
                    p.requires_grad_(True)
                params += list(m.parameters())
                layers.append(m)
            else:
                layers.append(Frozen(self.mods[s.unit], self.store.snapshot(s.unit).payload))
        net = nn.Sequential(*layers)
        gen_state = self.gen.get_state()

        def loss(idx):
            if commit:
                self.meter.add("maintenance", self.full_macs, len(idx), backward=True)
            return ce(net(x[idx]), y[idx])

        with torch.enable_grad():  # may be called from inside a no_grad evaluation
            train_steps(params, loss, self.batches(n_fit, steps=self.c.repair_steps,
                                                   batch_size=min(64, n_fit)), self.cfg.lr)
        if not commit:
            self.gen.set_state(gen_state)
        acc = self._feedback_acc(k, net, count=commit, part=slice(n_fit, None))
        if acc < self.ref_acc[k] - self.c.tolerance:
            return None
        if commit:
            for s, m in zip(spec, layers):
                if s in private:
                    self.mods[s.unit].load_state_dict(m.state_dict())
                    self.store.update(s.unit, self.mods[s.unit].state_dict())
        return net

    # -- serving ------------------------------------------------------------------------
    def _access(self, task_id: int) -> nn.Module:
        model = self.cache.get(task_id)
        if self.c.promotion:
            self._track_chunks(task_id)
        return model

    def _view(self, task_id: int) -> nn.Module:
        """What the system would answer for this task now, without changing any state."""
        entry = self.cache.peek(task_id)
        current = tuple(self.store.version(u) for u in entry.deps)
        if current == tuple(entry.seen[u] for u in entry.deps):
            return entry.value
        key = (task_id, tuple(entry.deps.items()), current)
        if key not in self._views:
            fresh, _ = self._build(entry.spec)
            if self._feedback_acc(task_id, fresh, count=False) >= self.ref_acc[task_id] - self.c.tolerance:
                self._views[key] = fresh
            else:
                repaired = self._repair(task_id, entry.spec, commit=False)
                self._views[key] = repaired if repaired is not None else entry.value
        return self._views[key]

    def _route(self, x, task_id, get_model):
        ids = torch.full((len(x),), task_id) if task_id is not None else self.router.route(x)
        out = torch.empty(len(x), dtype=torch.long)
        for t in ids.unique().tolist():
            m = ids == t
            model = get_model(t)
            with torch.no_grad():
                out[m] = model(x[m]).argmax(1)
        return out

    def predict(self, x: torch.Tensor, task_id: int | None) -> torch.Tensor:
        return self._route(x, task_id, self._access)

    def evaluate(self, x: torch.Tensor, task_id: int | None) -> torch.Tensor:
        return self._route(x, task_id, self._view)

    # -- promotion ----------------------------------------------------------------------
    def _track_chunks(self, task_id: int) -> None:
        spec = self.cache.peek(task_id).spec
        run: list[Slot] = []
        for s in list(spec) + [None]:
            shared = s is not None and s.unit not in self.parts and \
                len(self.users[s.unit]) >= self.c.promote_min_tasks
            if shared:
                run.append(s)
                continue
            if len(run) >= 2:
                self._hit_chunk(tuple(r.unit for r in run))
            run = []

    def _hit_chunk(self, units: tuple[str, ...]) -> None:
        stats = self.chunks.setdefault(units, ChunkStats())
        versions = tuple(self.store.version(u) for u in units)
        stats.hits += 1
        stats.stable = stats.stable + 1 if versions == stats.versions else 1
        stats.versions = versions
        if (stats.promoted is None and stats.hits >= self.c.promote_min_hits
                and stats.stable >= self.c.promote_min_stable):
            stats.promoted = self._promote(units)

    def _promote(self, units: tuple[str, ...]) -> str:
        mode = PromotionMode(self.c.promotion_mode)
        name = f"concept{len(self.parts)}[{'+'.join(units)}]"
        start, stop = self.span[units[0]][0], self.span[units[-1]][1]
        level = 1 + max(self.store.level(u) for u in units)
        if mode is PromotionMode.CUT_DEPS:
            module = nn.Sequential(*(copy.deepcopy(self.mods[u]) for u in units))
            self.store.add(name, module.state_dict(), domain="concepts", level=level,
                           kind=UnitKind.CONCEPT, meta={"origin": {u: self.store.version(u) for u in units}})
            self.mods[name] = module
            self.users[name] = set()
            self.parts[name] = units
            self.span[name] = (start, stop)
            return name
        # keep_deps: a named span that resolves to its live parts when composed
        self.parts[name] = units
        self.span[name] = (start, stop)
        self.users[name] = set()
        self.store.add(name, {}, domain="concepts", level=level, kind=UnitKind.CONCEPT,
                       deps={u: self.store.version(u) for u in units})
        return name

    # -- reporting ----------------------------------------------------------------------
    def memory(self) -> dict[str, int]:
        params = sum(t.numel() for snap in self.store.snapshots() for t in snap.payload.values())
        other = sum(x.numel() for x, _ in self.feedback.values()) + self.router.floats()
        return {"params": params, "other": other}

    def info(self) -> dict:
        return {
            "plans": dict(self.plans),
            "decisions": dict(self.decisions),
            "cache": self.cache.stats.as_dict(),
            "retained_old_snapshots": len(self.store.retained_old_snapshots()),
            "units": len(self.store),
            "concepts": list(self.parts),
            "pointers": {t: [(p.target, p.version, p.role, p.similarity.to_dict()) for p in ps]
                         for t, ps in self.pointers.items()},
            "log": self.log,
        }
