"""Offline study of candidate selection on recorded learning curves.

    python experiments/run_cl.py --benchmark mnist_compose --train-per-task 250 --min-steps 500 \
        --selection full --curve-every 25 --eval-split val --methods vcm_freeze_promo_comp \
        --seeds 0 1 2 --tag curves_n250 --out results/raw/selection
    python experiments/selection_curves.py --raw results/raw/selection --out results/selection

With `--selection full --curve-every 25` the learner trains every candidate composition
for the whole budget and logs, every 25 steps, its accuracy on the selection split
(the first `probe_val` validation samples) and on a disjoint held-out split (the next
`probe_val`). This script replays different selection rules on those curves: each rule
only looks at the selection split up to the steps it would have paid for; the chosen
candidate is scored by its held-out accuracy after the full budget. The "oracle" picks
by that held-out score itself and is the ceiling.

Caveat: in a real run the rule's choice changes which units later tasks can reuse; the
replay keeps the choices of the recorded run, so it scores each task's decision alone.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PARSIMONY = 0.005
TASK_NAMES = ["digit", "parity", "sum", "equal", "greater", "diff"]


def pick(scored):
    """(acc, n_new, name): fewest new parameters within PARSIMONY of the best."""
    best = max(s[0] for s in scored)
    return min((s for s in scored if s[0] >= best - PARSIMONY), key=lambda s: (s[1], -s[0]))[2]


def at(curve, step, col=1):
    """Accuracy at the last checkpoint <= step (col 1: selection split, 2: held-out)."""
    pts = [c for c in curve if c[0] <= step]
    return pts[-1][col]


def extrapolate(curve, k, total):
    """Fit acc(t) = a - b * t^(-c) to the selection-split points up to step k (c on a grid),
    and predict the accuracy at `total`, clipped to [0, 1]."""
    pts = np.array([(c[0], c[1]) for c in curve if c[0] <= k], dtype=float)
    t, y = pts[:, 0], pts[:, 1]
    best = (np.inf, y[-1])
    for c in (0.25, 0.5, 0.75, 1.0, 1.5):
        X = np.c_[np.ones_like(t), -(t ** -c)]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        err = float(((X @ coef - y) ** 2).sum())
        if coef[1] >= 0 and err < best[0]:       # rising curves only
            best = (err, float(coef[0] - coef[1] * total ** -c))
    return float(np.clip(best[1], 0.0, 1.0))


RULES = {}


def rule(name):
    def deco(f):
        RULES[name] = f
        return f
    return deco


for k in (50, 100, 200):
    RULES[f"probe {k}"] = (lambda k: lambda cands, total: (
        pick([(at(c["curve"], k), c["n_new"], n) for n, c in cands.items()]),
        k * len(cands) + total - k))(k)
    RULES[f"extrapolate {k}"] = (lambda k: lambda cands, total: (
        pick([(extrapolate(c["curve"], k, total), c["n_new"], n) for n, c in cands.items()]),
        k * len(cands) + total - k))(k)


def halving(start, eta):
    def f(cands, total):
        alive, budget, cost = list(cands), min(start, total), 0
        spent = {n: 0 for n in cands}
        while len(alive) > 1:
            for n in alive:
                cost += budget - spent[n]
                spent[n] = budget
            scored = sorted(((at(cands[n]["curve"], budget), cands[n]["n_new"], n) for n in alive), key=lambda s: -s[0])
            keep = max(1, -(-len(alive) // eta))
            alive = [pick(scored)] if keep == 1 or budget >= total else [s[2] for s in scored[:keep]]
            budget = min(total, budget * eta)
        return alive[0], cost + total - spent[alive[0]]
    return f


RULES["halving 25 x2"] = halving(25, 2)
RULES["halving 50 x2"] = halving(50, 2)
RULES["halving 50 x3"] = halving(50, 3)
RULES["full"] = lambda cands, total: (pick([(at(c["curve"], total), c["n_new"], n) for n, c in cands.items()]),
                                      total * len(cands))
RULES["oracle (held-out)"] = lambda cands, total: (max(cands, key=lambda n: at(cands[n]["curve"], total, 2)),
                                                   total * len(cands))


def decisions(raw: Path):
    """Yield (n_samples, seed, task, total, {candidate: {curve, n_new}}) for every logged choice."""
    for d in sorted(raw.glob("*curves_n*")):
        n = int(d.name.split("curves_n")[1])
        for f in sorted(d.glob("*.json")):
            run = json.loads(f.read_text())
            for entry in run["info"]["log"]:
                curves = entry.get("curves", {})
                if len(curves) < 2:
                    continue
                n_new = {c[0]: c[2] for c in entry["candidates"]}
                total = max(c[-1][0] for c in curves.values())
                cands = {name: {"curve": [tuple(p) for p in curve], "n_new": n_new[name]} for name, curve in curves.items()}
                yield n, run["seed"], entry["task"], total, cands


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    o = ap.parse_args()
    out = Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = defaultdict(list)            # (rule, n) -> [(held-out acc, regret, cost ratio, concept?)]
    flips = defaultdict(list)           # n -> step after which the selection-split leader is final
    per_task = defaultdict(lambda: defaultdict(list))
    examples = {}
    for n, seed, task, total, cands in decisions(Path(o.raw)):
        ceiling = max(at(c["curve"], total, 2) for c in cands.values())
        base_cost = 100 * len(cands) + total - 100
        for name, f in RULES.items():
            chosen, cost = f(cands, total)
            acc = at(cands[chosen]["curve"], total, 2)
            rows[(name, n)].append((acc, ceiling - acc, cost / base_cost, chosen.startswith("concept"), task >= 2))
            per_task[TASK_NAMES[task]][(name, n)].append((acc, chosen.startswith("concept")))
        final = max(cands, key=lambda c: at(cands[c]["curve"], total))
        steps = sorted({p[0] for c in cands.values() for p in c["curve"]})
        settled = next(s for s in steps if all(max(cands, key=lambda c: at(cands[c]["curve"], t)) == final
                                                 for t in steps if t >= s))
        flips[n].append(settled)
        if task == 2 and seed == 0:
            examples[n] = cands
    sizes = sorted({k[1] for k in rows})
    lines = ["# 候选选择：在记录下的学习曲线上重放各种选择规则\n",
             "每个学习器决策（parity + 四个两图任务）都把所有候选训满 500 步，每 25 步记录选择集与留出集上的准确率。"
             "各规则只看选择集、只看它付过费的那些步；被选中的候选按 500 步后在留出集上的准确率计分。"
             "“遗憾” = 天花板（按留出集挑）减去所选候选的留出集准确率。计算 = 候选训练步数之和，相对“探测 100 步”。\n",
             "## 两图任务（sum / equal / greater / diff）\n",
             "| 规则 | " + " | ".join(f"{n} 个样本：准确率 / 遗憾 / 选概念" for n in sizes) + " | 相对计算 |",
             "|---" * (len(sizes) + 2) + "|"]
    for name in RULES:
        cells, costs = [], []
        for n in sizes:
            r = np.array([x for x in rows[(name, n)] if x[4]], dtype=float)
            cells.append(f"{r[:, 0].mean():.3f} / {r[:, 1].mean():.3f} / {r[:, 3].mean():.0%}")
            costs.append(r[:, 2].mean())
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {np.mean(costs):.2f} |")
    tasks = sorted({k for k in per_task})
    lines += ["", "## 按任务：所选候选的留出集准确率（括号里是选中概念的比例）\n",
              "| 规则 | 样本数 | " + " | ".join(t for t in TASK_NAMES if t in tasks) + " |",
              "|---" * (len([t for t in TASK_NAMES if t in tasks]) + 2) + "|"]
    for name in ("probe 100", "halving 50 x2", "full", "oracle (held-out)"):
        for n in sizes:
            cells = []
            for t in (t for t in TASK_NAMES if t in tasks):
                r = np.array(per_task[t][(name, n)], dtype=float)
                cells.append(f"{r[:, 0].mean():.3f}（{r[:, 1].mean():.0%}）")
            lines.append(f"| {name} | {n} | " + " | ".join(cells) + " |")
    lines += ["", "## 选择集上的领先者从第几步起不再变\n",
              "| 样本数 | 中位数 | 75% 分位 | 最大值 |", "|---|---|---|---|"]
    for n in sizes:
        f = np.array(flips[n])
        lines.append(f"| {n} | {np.median(f):.0f} | {np.percentile(f, 75):.0f} | {f.max():.0f} |")
    lines.append("\n![curves](curves.png)\n")
    (out / "curves.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    plot(examples, out / "curves.png")


def plot(examples, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sizes = sorted(examples)
    fig, axes = plt.subplots(1, len(sizes), figsize=(4 * len(sizes), 3.4), sharey=False)
    for ax, n in zip(np.atleast_1d(axes), sizes):
        for name, c in examples[n].items():
            t = [p[0] for p in c["curve"]]
            ax.plot(t, [p[1] for p in c["curve"]], lw=1.6, label=name.replace("concept ", ""))
        ax.set_title(f"sum, {n} samples (seed 0)")
        ax.set_xlabel("training steps")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("selection-split accuracy")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
