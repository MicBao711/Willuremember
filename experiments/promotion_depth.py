"""Does promotion let knowledge be stacked deeper? A chunking experiment.

Random trees are composed bottom-up: each internal node binds its b children to roles
r0..r(b-1). A query walks a random root-to-leaf path by unbinding one role per hop.

* without promotion: intermediate results are not units, so nothing can clean them
  up; noise from every level accumulates and only the final leaf is cleaned up.
* with promotion: every internal composition was promoted to a concept unit (via the
  VCM facade), so each hop is snapped back to a clean vector before the next unbind.

    python experiments/promotion_depth.py [--out results/promotion_depth]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vcm import VCM, ItemMemory, PromotionMode


def role(height: int, i: int) -> str:
    # Roles are level-specific: circular convolution commutes, so with shared roles the
    # paths (r0, r1) and (r1, r0) would unbind to the same key.
    return f"h{height}r{i}"


def build_forest(vm: VCM, leaves: list[str], branching: int, depth: int, n_trees: int, rng):
    """Returns a list of (root_id, children map). Every internal node becomes a concept."""
    children: dict[str, list[str]] = {}
    counter = [0]

    def make(level: int) -> str:
        if level == 0:
            return leaves[int(rng.integers(len(leaves)))]
        kids = [make(level - 1) for _ in range(branching)]
        key = f"n{counter[0]}"
        counter[0] += 1
        vm.compose(key, {role(level, i): k for i, k in enumerate(kids)})
        name = vm.promote(key, f"concept:{key}", PromotionMode.CUT_DEPS)
        vm.cache.invalidate(key)
        children[name] = kids
        return name

    roots = [make(depth) for _ in range(n_trees)]
    return roots, children


def run(dim: int, branching: int, depth: int, n_leaves: int, n_trees: int, n_queries: int, seed: int):
    rng = np.random.default_rng(seed)
    vm = VCM(dim=dim, seed=seed)
    leaves = [f"leaf{i}" for i in range(n_leaves)]
    for leaf in leaves:
        vm.add(leaf)
    roots, children = build_forest(vm, leaves, branching, depth, n_trees, rng)

    leaf_mem = ItemMemory()
    for leaf in leaves:
        leaf_mem.add(leaf, vm.vector(leaf))
    full_mem = ItemMemory()
    for unit in vm.store:
        full_mem.add(unit.id, unit.payload)

    ok_plain = ok_promoted = 0
    for _ in range(n_queries):
        root = node = roots[int(rng.integers(len(roots)))]
        path = []
        while node in children:
            i = int(rng.integers(branching))
            path.append(i)
            node = children[node][i]
        target = node
        v = vm.vector(root)
        for hop, i in enumerate(path):
            v = vm.composer.extract(v, role(depth - hop, i))
        ok_plain += leaf_mem.cleanup(v)[0] == target
        v = vm.vector(root)
        for hop, i in enumerate(path):
            name, _ = full_mem.cleanup(vm.composer.extract(v, role(depth - hop, i)))
            v = vm.vector(name)
        ok_promoted += name == target
    return dict(dim=dim, branching=branching, depth=depth, n_leaves=n_leaves,
                acc_plain=ok_plain / n_queries, acc_promoted=ok_promoted / n_queries,
                memory_items=len(full_mem), concepts=len(children))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/promotion_depth")
    ap.add_argument("--queries", type=int, default=300)
    args = ap.parse_args()
    rows = []
    for dim in [512, 1024]:
        for b in [2, 3, 4]:
            max_depth = {2: 7, 3: 5, 4: 4}[b]
            for depth in range(1, max_depth + 1):
                n_trees = max(4, min(30, 2000 // (b ** depth)))
                r = run(dim, b, depth, n_leaves=500, n_trees=n_trees, n_queries=args.queries, seed=depth * 7 + b)
                rows.append(r)
                print(r, flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "promotion_depth.json").write_text(json.dumps(rows, indent=1))
    plot(rows, out / "promotion_depth.png")
    lines = ["# 晋升与层级深度：逐层解绑的准确率\n",
             "每个内部节点把 b 个子节点绑定到角色 r0..r(b-1)（角色按层区分，避免循环卷积交换律造成路径混淆）。查询沿随机路径从根逐层解绑到叶子（500 个候选叶子）。",
             "无晋升：中间结果不是单元，无法清理，噪声逐层累积；有晋升：每个中间组合都已晋升为概念单元，每一跳先清理再继续。\n",
             "| 维度 d | 分支 b | 深度 | 无晋升 | 有晋升 | 清理记忆条目数 |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['dim']} | {r['branching']} | {r['depth']} | {r['acc_plain']:.3f} | "
                     f"{r['acc_promoted']:.3f} | {r['memory_items']} |")
    (out / "promotion_depth.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dims = sorted({r["dim"] for r in rows})
    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 3.8), sharey=True)
    colors = {2: "#00798c", 3: "#edae49", 4: "#d1495b"}
    for ax, dim in zip(np.atleast_1d(axes), dims):
        for b in [2, 3, 4]:
            sel = [r for r in rows if r["dim"] == dim and r["branching"] == b]
            d = [r["depth"] for r in sel]
            ax.plot(d, [r["acc_promoted"] for r in sel], "-o", ms=4, color=colors[b], label=f"b={b}, promoted")
            ax.plot(d, [r["acc_plain"] for r in sel], "--o", ms=4, mfc="white", color=colors[b], label=f"b={b}, no promotion")
        ax.set_title(f"d = {dim}")
        ax.set_xlabel("composition depth (hops)")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("leaf decoding accuracy")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
