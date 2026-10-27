"""Cost of keeping compositions consistent: no cache vs. eager vs. lazy (VCM).

A random two-level knowledge graph is built with the VCM facade:
  base units  ->  keep-deps concepts (promoted compositions)  ->  composites
Then a stream of events runs: with probability p a random base unit is updated,
otherwise a composite is read (Zipf-distributed popularity). Every read must return
exactly what a fresh composition from current units would give; this is checked.

Strategies:
  none   recompose on every read (concept fillers recomposed too)
  eager  after every update, immediately rebuild everything that depends on it
  lazy   compare versions on read; rebuild only what is read and stale

    python experiments/cache_cost.py [--events 6000] [--out results/cache_cost]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vcm import VCM, PromotionMode
from vcm.graph import dependents


def build_world(n_base: int, n_concepts: int, n_composites: int, arity: int, dim: int, rng):
    vm = VCM(dim=dim, seed=int(rng.integers(1 << 30)))
    base = [f"b{i}" for i in range(n_base)]
    for b in base:
        vm.add(b)
    concepts = []
    for i in range(n_concepts):
        parts = rng.choice(base, size=arity, replace=False)
        key = f"c{i}"
        vm.compose(key, {f"r{j}": p for j, p in enumerate(parts)})
        concepts.append(vm.promote(key, f"concept:{key}", PromotionMode.KEEP_DEPS))
        vm.cache.invalidate(key)  # the recipe now lives in the concept
    composites = []
    for i in range(n_composites):
        n_c = int(rng.integers(0, 3))
        fillers = list(rng.choice(concepts, size=n_c, replace=False)) + list(
            rng.choice(base, size=arity - n_c, replace=False))
        spec = {f"r{j}": u for j, u in enumerate(fillers)}
        composites.append((f"k{i}", spec))
    return vm, base, concepts, composites


def ground_truth(vm: VCM, spec: dict) -> np.ndarray:
    def value(u):
        unit = vm.store.get(u)
        if unit.spec is not None and unit.deps:
            return vm.composer.compose({r: value(x) for r, x in unit.spec.bindings})
        return unit.payload
    return vm.composer.compose({r: value(u) for r, u in spec.items()})


def run(strategy: str, p_update: float, zipf: float, events: int, seed: int, world_kw) -> dict:
    rng = np.random.default_rng(seed)
    vm, base, concepts, composites = build_world(rng=rng, **world_kw)
    ranks = np.arange(1, len(composites) + 1, dtype=float)
    probs = ranks ** -zipf
    probs /= probs.sum()
    edges = {k: list(spec.values()) for k, spec in composites}
    edges.update({c: list(vm.store.get(c).deps) for c in concepts})

    ops = 0          # compositions computed (composite rebuilds + concept rebuilds)
    checks = 0       # version comparisons
    composes0 = vm.cache.stats.composes
    refresh0 = vm.concept_refreshes
    if strategy != "none":
        for key, spec in composites:
            vm.compose(key, spec)
    warm_ops = vm.cache.stats.composes - composes0 + vm.concept_refreshes - refresh0
    composes0, refresh0 = vm.cache.stats.composes, vm.concept_refreshes
    checks0 = vm.cache.stats.version_checks
    reads = updates = 0
    max_err = 0.0
    spec_of = dict(composites)

    for _ in range(events):
        if rng.random() < p_update:
            u = base[int(rng.integers(len(base)))]
            vm.update(u, vm.vsa.random(f"{u}:{updates}"))
            updates += 1
            if strategy == "eager":
                affected = dependents(edges, u)
                for c in sorted(c for c in affected if c in concepts):
                    vm.fresh_version(c)  # rebuild concept now
                for k in sorted(k for k in affected if k in spec_of):
                    vm.cache.refresh(k)
        else:
            key, spec = composites[int(rng.choice(len(composites), p=probs))]
            reads += 1
            if strategy == "none":
                n_concepts = sum(1 for u in spec.values() if u in concepts)
                ops += 1 + n_concepts
                value = ground_truth(vm, spec)
            else:
                value = vm.lookup(key)
            if reads % 25 == 0:
                max_err = max(max_err, 1 - float(np.dot(value, ground_truth(vm, spec))))

    if strategy != "none":
        ops = vm.cache.stats.composes - composes0 + vm.concept_refreshes - refresh0
        checks = vm.cache.stats.version_checks - checks0
    return dict(strategy=strategy, p_update=p_update, zipf=zipf, reads=reads, updates=updates,
                ops=ops, warmup_ops=warm_ops, version_checks=checks,
                ops_per_event=ops / events, max_consistency_error=max_err)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/cache_cost")
    args = ap.parse_args()
    world_kw = dict(n_base=300, n_concepts=60, n_composites=400, arity=3, dim=256)
    rates = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    rows = []
    for zipf in [0.0, 1.1]:
        for p in rates:
            for strategy in ["none", "eager", "lazy"]:
                for seed in range(args.seeds):
                    r = run(strategy, p, zipf, args.events, seed, world_kw)
                    rows.append(r)
                    print(f"zipf={zipf} p={p:.2f} {strategy:5s} seed={seed} ops/event={r['ops_per_event']:.3f} "
                          f"checks={r['version_checks']} err={r['max_consistency_error']:.1e}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cache_cost.json").write_text(json.dumps(dict(world=world_kw, events=args.events, rows=rows), indent=1))
    plot(rows, out / "cache_cost.png", rates)
    summarize(rows, out / "cache_cost.md", world_kw, args)


def _mean(rows, **match):
    sel = [r for r in rows if all(r[k] == v for k, v in match.items())]
    return float(np.mean([r["ops_per_event"] for r in sel])), float(np.std([r["ops_per_event"] for r in sel]))


def plot(rows, path, rates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"none": "#8a8a8a", "eager": "#d1495b", "lazy": "#00798c"}
    labels = {"none": "no cache", "eager": "eager invalidation", "lazy": "lazy version check (VCM)"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, zipf in zip(axes, [0.0, 1.1]):
        for s in ["none", "eager", "lazy"]:
            m = [_mean(rows, strategy=s, p_update=p, zipf=zipf) for p in rates]
            ax.errorbar(rates, [a for a, _ in m], yerr=[b for _, b in m], marker="o", ms=4,
                        color=colors[s], label=labels[s], capsize=2)
        ax.set_title("uniform reads" if zipf == 0 else f"Zipf reads (s={zipf})")
        ax.set_xlabel("update probability per event")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("compositions computed per event")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def summarize(rows, path, world_kw, args):
    lines = ["# 缓存一致性的计算开销：无缓存 / 立即失效 / 惰性版本检查\n",
             f"世界：{world_kw}；每个设置 {args.events} 个事件 × {args.seeds} 个种子。",
             "指标：每个事件平均需要计算多少次组合（含概念重建）。所有策略读到的结果都与“从当前单元重新组合”一致"
             "（最大一致性误差见 JSON 中 `max_consistency_error`）。\n",
             "| 读取分布 | 更新概率 | 无缓存 | 立即失效 | 惰性 (VCM) | 惰性/立即 |", "|---|---|---|---|---|---|"]
    for zipf in [0.0, 1.1]:
        for p in sorted({r["p_update"] for r in rows}):
            n, _ = _mean(rows, strategy="none", p_update=p, zipf=zipf)
            e, _ = _mean(rows, strategy="eager", p_update=p, zipf=zipf)
            l, _ = _mean(rows, strategy="lazy", p_update=p, zipf=zipf)
            dist = "均匀" if zipf == 0 else f"Zipf {zipf}"
            lines.append(f"| {dist} | {p:.2f} | {n:.3f} | {e:.3f} | {l:.3f} | {l / e:.2f} |")
    err = max(r["max_consistency_error"] for r in rows)
    lines.append(f"\n最大一致性误差（1 − cos）：{err:.1e}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
