"""Candidate selection on Split CIFAR: can a cheaper rule buy what full training buys?

On Split CIFAR, VCM (freeze) with every candidate trained fully ties MNTDP-D (v0.4), at
about three times the compute of the default 50-step probe. Three parts:

1. curves (validation only). Every candidate trains the whole budget and logs, every 25
   steps, its accuracy on the selection split and on a disjoint held-out split. CIFAR's
   validation pools are small (1000 / 500 images per task), so these runs hold out 20% of
   each training pool instead of 10%: the selection split keeps the size the learner uses
   (1000 images on CIFAR-10, all 500 on CIFAR-100) and the rest scores the choice. Rules are
   replayed on the curves as in selection_curves.py.

    python experiments/run_cl.py --benchmark split_cifar10 --feedback-source train --repair-steps 100 \
        --val-fraction 0.2 --selection full --curve-every 25 --eval-split val \
        --methods vcm_freeze --seeds 0 1 2 3 4 5 6 7 --tag curves
    (split_cifar100: the same plus --probe-val 500)

2. learner (test). The rules fixed on the curves (`--probe-steps 25` with `--selection halving`
   or `--selection finalists`), run inside the learner and paired by seed with the probe and
   full runs of `run_suite.sh cifar10 | cifar100 | cifar_full`.

    bash experiments/run_suite.sh cifar_curves   # part 1 (also writes the report)
    bash experiments/run_suite.sh cifar_select   # part 2
    python experiments/selection_cifar.py        # every part, from whatever raw data exists

3. seeds (test, v0.6). Seeds 3-9 for every arm of part 2, analysed as pre-registered in
   results/selection_cifar/seeds_plan.md (paired differences, t intervals, sign-flip tests).

    bash experiments/run_suite.sh cifar_seeds
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_cl import load  # noqa: E402
from selection_curves import at, halving, pick  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = [("split_cifar10", "Split CIFAR-10"), ("split_cifar100", "Split CIFAR-100")]
BASE = "probe 50"   # the CIFAR default (probe_steps = 50)


def probe(k, parsimony=True):
    def f(cands, total):
        scored = [(at(c["curve"], k), c["n_new"], n) for n, c in cands.items()]
        chosen = pick(scored) if parsimony else max(scored, key=lambda s: s[0])[2]
        return chosen, k * len(cands) + total - k
    return f


def race(k, margin):
    """Probe k steps; every candidate within `margin` of the leader trains on to the full
    budget, and the best of those at the end is kept."""
    def f(cands, total):
        scored = [(at(c["curve"], k), c["n_new"], n) for n, c in cands.items()]
        best = max(s[0] for s in scored)
        alive = [s[2] for s in scored if s[0] >= best - margin]
        cost = k * len(cands) + (total - k) * len(alive)
        return pick([(at(cands[n]["curve"], total), cands[n]["n_new"], n) for n in alive]), cost
    return f


def top(k, m):
    """Probe k steps; the best m train on to the full budget, the best of those is kept."""
    def f(cands, total):
        scored = sorted(((at(c["curve"], k), c["n_new"], n) for n, c in cands.items()), key=lambda s: -s[0])
        alive = [s[2] for s in scored[:m]]
        cost = k * len(cands) + (total - k) * len(alive)
        return pick([(at(cands[n]["curve"], total), cands[n]["n_new"], n) for n in alive]), cost
    return f


def full(parsimony=True):
    def f(cands, total):
        scored = [(at(c["curve"], total), c["n_new"], n) for n, c in cands.items()]
        return (pick(scored) if parsimony else max(scored, key=lambda s: s[0])[2]), total * len(cands)
    return f


RULES = {
    "probe 25": probe(25),
    "probe 50": probe(50),
    "probe 100": probe(100),
    "probe 50, no parsimony": probe(50, parsimony=False),
    "halving 25 x2": halving(25, 2),
    "halving 25 x3": halving(25, 3),
    "halving 50 x2": halving(50, 2),
    "race 25, margin 0.02": race(25, 0.02),
    "race 50, margin 0.01": race(50, 0.01),
    "race 50, margin 0.02": race(50, 0.02),
    "top-2 after 25": top(25, 2),
    "top-2 after 50": top(50, 2),
    "top-3 after 25 (post hoc)": top(25, 3),
    "full": full(),
    "full, no parsimony": full(parsimony=False),
    "oracle (held-out)": lambda cands, total: (max(cands, key=lambda n: at(cands[n]["curve"], total, 2)),
                                               total * len(cands)),
}
FAMILY = {"probe": "#2a78d6", "halving": "#eb6834", "top-2": "#1baf7a"}   # categorical slots 1-3
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DEPTH = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]         # ordinal: scratch .. bottom4


def kind(source: str) -> str:
    return source.split("<-")[0]


def decisions(raw: Path):
    """Yield (seed, task, total, {candidate: {curve, n_new}}) for every logged choice."""
    for run in (r for rs in load(raw).values() for r in rs):
        for entry in run["info"]["log"]:
            curves = entry.get("curves", {})
            if len(curves) < 2:
                continue
            n_new = {c[0]: c[2] for c in entry["candidates"]}
            total = max(c[-1][0] for c in curves.values())
            yield run["seed"], entry["task"], total, {
                name: {"curve": [tuple(p) for p in curve], "n_new": n_new[name]} for name, curve in curves.items()}


def replay(raw: Path):
    """Per rule: list of (seed, held-out acc of the chosen candidate, regret, cost, kind);
    per decision, the rank on the selection split at each step of `full`'s choice; and the
    pairs of candidates that lead after 25 steps (the finalists of `top-2 after 25`)."""
    rows, ranks, pairs, totals, example = defaultdict(list), [], Counter(), set(), None
    for seed, task, total, cands in decisions(raw):
        totals.add(total)
        ceiling = max(at(c["curve"], total, 2) for c in cands.values())
        for name, f in RULES.items():
            chosen, cost = f(cands, total)
            acc = at(cands[chosen]["curve"], total, 2)
            rows[name].append((seed, acc, ceiling - acc, cost, kind(chosen)))
        lead = sorted(cands, key=lambda n: -at(cands[n]["curve"], 25))[:2]
        pairs[" + ".join(sorted(kind(n) for n in lead))] += 1
        winner = RULES["full"](cands, total)[0]
        steps = sorted({p[0] for c in cands.values() for p in c["curve"]})
        ranks.append({t: 1 + sum(at(c["curve"], t) > at(cands[winner]["curve"], t) for c in cands.values())
                      for t in steps})
        if example is None or (seed, task) == (0, 2):
            example = (task, total, cands)
    return rows, ranks, pairs, totals, example


def curves_part(raw_root: Path, out: Path) -> dict:
    found = {}
    for bench, title in BENCHMARKS:
        raw = raw_root / f"{bench}_curves"
        if raw.exists() and (res := replay(raw))[0]:
            found[bench] = (title, *res)
    if not found:
        return found
    lines = ["# Split CIFAR 上的候选选择：在学习曲线上重放（验证集）\n",
             "VCM（freeze）每个决策的所有候选都训满，每 25 步记录选择集与留出集准确率（诊断运行留出训练池的 20%：选择集大小与学习器实际用的相同，"
             "CIFAR-10 1000 张、CIFAR-100 500 张，其余做留出集）。各规则只看选择集、只看它付过费的步数；"
             "所选候选按训满后在留出集上的准确率计分。“遗憾” = 天花板（按留出集挑）− 所选；“计算” = 候选训练步数之和，相对“探测 50 步”（CIFAR 的默认）；"
             "“对探测 50”= 逐决策配对差的均值，括号里是按种子平均后为正 / 为负的种子数；“补回” = (规则 − 探测 50) / (训满 − 探测 50)。"
             "测试集上跑哪两条规则，是看完本页之前定好的标准选的：与探测 50 同价的逐步淘汰（所问的问题本身），"
             "以及计算不超过 1.5 倍的规则里两个基准平均增益最大的一条（top-2 after 25）。"
             "标 (post hoc) 的规则是看过测试集结果之后才加的，只在这里看，没有上测试集。\n"]
    for bench, (title, rows, ranks, pairs, totals, _) in found.items():
        n_dec, seeds = len(rows[BASE]), sorted({r[0] for r in rows[BASE]})
        gap = np.mean([r[1] for r in rows["full"]]) - np.mean([r[1] for r in rows[BASE]])
        lines += [f"## {title}（{len(seeds)} 个种子，{n_dec} 个决策）\n",
                  "| 规则 | 留出集准确率 | 遗憾 | 计算 | 对探测 50 | 补回 | 所选方案 |", "|---|---|---|---|---|---|---|"]
        base_cost = np.mean([r[3] for r in rows[BASE]])
        for name in RULES:
            r = rows[name]
            acc = np.array([x[1] for x in r])
            d = acc - np.array([x[1] for x in rows[BASE]])
            per_seed = np.array([d[[i for i, x in enumerate(r) if x[0] == s]].mean() for s in seeds])
            share = f"{(acc.mean() - np.mean([x[1] for x in rows[BASE]])) / gap:.0%}" if abs(gap) > 1e-9 else "–"
            kinds = Counter(x[4] for x in r)
            mix = ", ".join(f"{k}×{v}" for k, v in sorted(kinds.items()))
            lines.append(f"| {name} | {acc.mean():.4f} | {np.mean([x[2] for x in r]):.4f} | "
                         f"{np.mean([x[3] for x in r]) / base_cost:.2f} | {d.mean():+.4f}（{int((per_seed > 1e-9).sum())}+ / "
                         f"{int((per_seed < -1e-9).sum())}−） | {share} | {mix} |")
        steps = [t for t in (25, 50, 100, 200) if t < min(totals)]
        lines += ["", f"“全部训满”最终选中的候选，在第 k 步时于选择集上排第几（{len(ranks)} 个决策；"
                      f"总预算 {'/'.join(str(t) for t in sorted(totals))} 步）：\n",
                  "| 步数 | 第 1 | 前 2 | 前 3 |", "|---|---|---|---|"]
        for t in steps:
            r = np.array([d[t] for d in ranks])
            lines.append(f"| {t} | {(r <= 1).mean():.0%} | {(r <= 2).mean():.0%} | {(r <= 3).mean():.0%} |")
        lines += ["", "第 25 步时选择集上的前两名（即“top-2 after 25”的决赛名单）：" +
                  "，".join(f"{k} {v} 次" for k, v in pairs.most_common()) + "。\n"]
    lines.append("![curves](curves.png)\n\n![pareto](pareto.png)\n")
    (out / "curves.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    plot_curves(found, out / "curves.png")
    plot_pareto(found, out / "pareto.png")
    return found


def _style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def plot_curves(found, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(found), figsize=(5.2 * len(found), 3.6))
    order = ["scratch", "bottom1", "bottom2", "bottom3", "bottom4"]
    for ax, (bench, (title, _, _, _, _, (task, total, cands))) in zip(np.atleast_1d(axes), found.items()):
        names = sorted((n for n in cands if kind(n) != "bottom4"), key=lambda n: order.index(kind(n)))
        for n in names:
            c = cands[n]["curve"]
            ax.plot([p[0] for p in c], [p[1] for p in c], lw=2, color=DEPTH[order.index(kind(n))], label=kind(n))
        for k, ls in ((25, ":"), (50, "--")):
            ax.axvline(k, color=INK2, lw=0.8, ls=ls)
        ax.set_title(f"{title}, task {task} (seed 0); bottom4 (new head only) is far below", fontsize=9, color=INK)
        ax.set_xlabel("training steps", fontsize=8, color=INK2)
        _style(ax)
    first = np.atleast_1d(axes)[0]
    first.set_ylabel("selection-split accuracy", fontsize=8, color=INK2)
    first.legend(frameon=False, fontsize=7, title="reused layers", title_fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def plot_pareto(found, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    show = ["probe 25", "probe 50", "probe 100", "halving 25 x2", "halving 50 x2", "top-2 after 25", "full",
            "oracle (held-out)"]   # the rest are in the table
    fig, axes = plt.subplots(1, len(found), figsize=(5.2 * len(found), 3.8))
    for ax, (bench, (title, rows, _, _, _, _)) in zip(np.atleast_1d(axes), found.items()):
        base_cost = np.mean([r[3] for r in rows[BASE]])
        for name in show:
            x = np.mean([r[3] for r in rows[name]]) / base_cost
            y = np.mean([r[1] for r in rows[name]])
            if name.startswith("oracle"):
                ax.axhline(y, color=INK2, lw=1, ls="--")
                ax.annotate("ceiling (pick by held-out)", (1.0, y), xytext=(0, 3), textcoords="offset points",
                            fontsize=7, color=INK2)
                continue
            ax.plot(x, y, "o", ms=8, color=FAMILY.get(name.split()[0], INK), mec="#fcfcfb", mew=2)
            ax.annotate(name, (x, y), xytext=(6, -3), textcoords="offset points", fontsize=7, color=INK)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("selection cost (candidate steps, relative to probe 50)", fontsize=8, color=INK2)
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("held-out accuracy of the chosen candidate", fontsize=8, color=INK2)
    handles = [Line2D([], [], ls="", marker="o", ms=7, color=c, label=lab) for c, lab in
               ((FAMILY["probe"], "probe k steps"), (FAMILY["halving"], "successive halving"),
                (FAMILY["top-2"], "probe, then the top 2 train fully"), (INK, "every candidate trains fully"))]
    np.atleast_1d(axes)[0].legend(handles=handles, frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


# -- part 2: the learner on test -----------------------------------------------------------
ARMS = [("", "探测 50 步（默认）"), ("_halving", "逐步淘汰（25 步起，每轮减半；与探测同价）"),
        ("_finalists", "探测 25 步，前 2 名训满"), ("_full", "全部训满再挑")]
METHODS = [("vcm_freeze", "VCM（freeze）"), ("vcm_lazy", "VCM（update + 惰性）")]


def plans(run) -> list[str]:
    return [kind(p) for k, p in sorted(run["info"].get("plans", {}).items(), key=lambda kv: int(kv[0])) if int(k) > 0]


def by_seed(runs) -> dict[int, dict]:
    return {r["seed"]: r for r in runs}


def paired(a, b) -> str:
    a, b = by_seed(a), by_seed(b)
    d = np.array([a[s]["metrics"]["acc"] - b[s]["metrics"]["acc"] for s in sorted(set(a) & set(b))])
    if not len(d):
        return "–"
    return f"{d.mean():+.4f}（{int((d > 0).sum())}/{len(d)}）"


def learner_part(raw_root: Path, out: Path, arms) -> None:
    lines = ["# 学习器里的候选选择：Split CIFAR（测试集）\n",
             "与 `run_suite.sh cifar10 | cifar100 | cifar_full` 同样的设置（v0.1 参数），只改 `--selection`；同一方法、同一种子配对。"
             "计算是整个任务流的总乘加次数；“方案”是第 2 个任务起被选中的组合（bottomk = 复用前 k 层，scratch = 从零）。"
             "要跑的两条新规则在验证集曲线上事先选定（见 curves.md）；括号里是差值为正的种子数。\n"]
    for bench, title in BENCHMARKS:
        runs = {v: load(raw_root / f"{bench}{v}") for v, _ in arms if (raw_root / f"{bench}{v}").exists()}
        if len(runs) < 2:
            continue
        ref = runs[""]
        lines += [f"## {title}\n",
                  "| 方法 | 选择方式 | 种子数 | 最终 ACC | 计算 (TMAC) | 对探测 50 步（配对） | 对全部训满（配对） | 方案 |",
                  "|---|---|---|---|---|---|---|---|"]
        for m, mlabel in METHODS:
            for v, label in arms:
                rs = runs.get(v, {}).get(m)
                if not rs:
                    continue
                acc = np.array([r["metrics"]["acc"] for r in rs])
                comp = np.mean([r["compute"]["total"] for r in rs]) / 1e12
                mix = Counter(p for r in rs for p in plans(r))
                full_runs = runs.get("_full", {}).get(m)
                lines.append(f"| {mlabel} | {label} | {len(rs)} | {acc.mean():.4f} ± {acc.std():.4f} | {comp:.1f} | "
                             f"{paired(rs, ref.get(m, [])) if v else '–'} | "
                             f"{paired(rs, full_runs) if full_runs and v != '_full' else '–'} | "
                             f"{', '.join(f'{k}×{n}' for k, n in sorted(mix.items()))} |")
        for m, mlabel in (("mntdp", "MNTDP-D"), ("independent", "Independent")):
            rs = ref.get(m)
            if rs:
                acc = np.array([r["metrics"]["acc"] for r in rs])
                lines.append(f"| {mlabel} | – | {len(rs)} | {acc.mean():.4f} ± {acc.std():.4f} | "
                             f"{np.mean([r['compute']['total'] for r in rs]) / 1e12:.1f} | – | – | – |")
        lines.append("")
    (out / "learner.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# -- part 3: more test seeds (v0.6), analysed as pre-registered in seeds_plan.md -------------
GROUPS = [("0–2（v0.5）", (0, 1, 2)), ("3–9（新）", (3, 4, 5, 6, 7, 8, 9)), ("0–9（全部）", tuple(range(10)))]
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
       11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093}


def accs(raw_root: Path, bench: str, suffix: str, method: str) -> dict[int, float]:
    d = raw_root / f"{bench}{suffix}"
    return {r["seed"]: r["metrics"]["acc"] for r in load(d).get(method, [])} if d.exists() else {}


def diffs(a: dict, b: dict, seeds) -> dict[int, float]:
    return {s: a[s] - b[s] for s in seeds if s in a and s in b}


def stats(d) -> dict:
    """Paired differences: mean, standard error, 95% t interval, seeds above zero, and the
    two-sided p-value of the exact sign-flip permutation test."""
    d = np.asarray(list(d), dtype=float)
    n = len(d)
    if n < 2:
        return dict(n=n, mean=d.mean() if n else np.nan, se=np.nan, lo=np.nan, hi=np.nan, pos=int((d > 0).sum()), p=np.nan)
    se = d.std(ddof=1) / np.sqrt(n)
    half = T95.get(n - 1, 1.96) * se
    flips = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    p = float(np.mean(np.abs(flips @ d) >= abs(d.sum()) - 1e-12))
    return dict(n=n, mean=d.mean(), se=se, lo=d.mean() - half, hi=d.mean() + half, pos=int((d > 0).sum()), p=p)


def stat_cells(s: dict) -> str:
    pp = 100   # percentage points
    if s["n"] < 2:
        return f"{s['n']} | {s['mean'] * pp:+.2f} | – | – | {s['pos']}/{s['n']} | – |"
    return (f"{s['n']} | {s['mean'] * pp:+.2f} | {s['se'] * pp:.2f} | [{s['lo'] * pp:+.2f}, {s['hi'] * pp:+.2f}] | "
            f"{s['pos']}/{s['n']} | {s['p']:.3f} |")


def verdict(new: dict, every: dict) -> str:
    """The pre-registered rule (seeds_plan.md)."""
    if new["n"] == 0:
        return "还没有新种子"
    tail = "" if every["n"] == 10 else f"（种子不全：{every['n']}/10，判定是暂时的）"
    if new["mean"] <= 0:
        return "没有重复出来" + tail
    return ("坐实" if every["lo"] > 0 else "方向一致、证据仍不足") + tail


def seeds_part(raw_root: Path, out: Path) -> None:
    arm = {(b, v, m): accs(raw_root, b, v, m) for b, _ in BENCHMARKS
           for v, m in (("", "vcm_freeze"), ("_finalists", "vcm_freeze"), ("_full", "vcm_freeze"),
                        ("_halving", "vcm_freeze"), ("", "mntdp"), ("", "vcm_lazy"), ("_finalists", "vcm_lazy"))}
    main_d = {b: diffs(arm[b, "_finalists", "vcm_freeze"], arm[b, "", "vcm_freeze"], range(10)) for b, _ in BENCHMARKS}
    if not any(s >= 3 for d in main_d.values() for s in d):
        return
    both = {s: np.mean([main_d[b][s] for b, _ in BENCHMARKS]) for s in range(10) if all(s in main_d[b] for b, _ in BENCHMARKS)}
    head = "| 种子 | n | 均值 | 标准误 | 95% 置信区间 | 为正 | 置换 p |"
    lines = ["# 补种子：finalists 在测试集上的复核（v0.6）\n",
             "按 [`seeds_plan.md`](seeds_plan.md) 事先登记的方案：新种子 3–9 上跑与 v0.5 完全相同的各臂（`run_suite.sh cifar_seeds`），"
             "规则、步数、决赛名额都不动。差值单位是百分点（测试集最终 ACC，逐种子配对）；置信区间用 t 分布；"
             "置换 p 是双侧的精确符号翻转检验。\n",
             "## 主终点：VCM（freeze）探测 25 步 + 前 2 名训满 − 探测 50 步\n"]
    for b, title in BENCHMARKS:
        g = {name: stats([main_d[b][s] for s in seeds if s in main_d[b]]) for name, seeds in GROUPS}
        lines += [f"### {title}\n", head, "|---|---|---|---|---|---|---|",
                  *[f"| {name} | {stat_cells(g[name])}" for name, _ in GROUPS],
                  "", f"判定（事先定的规则）：**{verdict(g['3–9（新）'], g['0–9（全部）'])}**。\n"]
    g = {name: stats([both[s] for s in seeds if s in both]) for name, seeds in GROUPS}
    lines += ["### 两个基准平均（每个种子先对两个基准取平均；只做汇总，不判定）\n", head, "|---|---|---|---|---|---|---|",
              *[f"| {name} | {stat_cells(g[name])}" for name, _ in GROUPS], ""]

    lines += ["## 次要比较（全部种子，只描述）\n",
              "| 基准 | 比较 | n | 均值 | 标准误 | 95% 置信区间 | 为正 | 置换 p |", "|---|---|---|---|---|---|---|---|"]
    rows = [("训满 − 探测 50（freeze）", ("_full", "vcm_freeze"), ("", "vcm_freeze")),
            ("finalists − 训满（freeze）", ("_finalists", "vcm_freeze"), ("_full", "vcm_freeze")),
            ("finalists − MNTDP-D", ("_finalists", "vcm_freeze"), ("", "mntdp")),
            ("逐步淘汰 − 探测 50（freeze）", ("_halving", "vcm_freeze"), ("", "vcm_freeze")),
            ("finalists − 探测 50（update + 惰性）", ("_finalists", "vcm_lazy"), ("", "vcm_lazy"))]
    share = {}
    for b, title in BENCHMARKS:
        for label, x, y in rows:
            d = diffs(arm[(b, *x)], arm[(b, *y)], range(10))
            lines.append(f"| {title} | {label} | {stat_cells(stats(d.values()))}")
        common = [s for s in range(10) if all(s in arm[b, v, "vcm_freeze"] for v in ("", "_finalists", "_full"))]
        gain = lambda v: np.mean([arm[b, v, "vcm_freeze"][s] - arm[b, "", "vcm_freeze"][s] for s in common])  # noqa: E731
        share[title] = (gain("_finalists") / gain("_full"), len(common)) if common and abs(gain("_full")) > 1e-12 else None
    lines += ["", "补回比例 = (finalists − 探测 50) / (训满 − 探测 50)，用三臂都有的种子的均值：" +
              "；".join(f"{t} {v[0]:.0%}（{v[1]} 个种子）" for t, v in share.items() if v) + "。\n"]

    lines += ["## 逐种子明细（百分点）\n",
              "| 种子 | " + " | ".join(f"{t}：finalists − 探测 50 | {t}：训满 − 探测 50" for _, t in BENCHMARKS) + " |",
              "|---|" + "---|---|" * len(BENCHMARKS)]
    for s in range(10):
        cells = []
        for b, _ in BENCHMARKS:
            for v in ("_finalists", "_full"):
                d = diffs(arm[b, v, "vcm_freeze"], arm[b, "", "vcm_freeze"], [s])
                cells.append(f"{d[s] * 100:+.2f}" if d else "–")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines.append("\n各臂的准确率、计算与所选方案见 [`learner.md`](learner.md)。\n")
    (out / "seeds.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "results" / "selection_cifar"))
    ap.add_argument("--arms", nargs="*", default=None,
                    help="learner arms as suffix=label pairs (default: probe, halving, finalists, full)")
    o = ap.parse_args()
    out = Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    curves_part(Path(o.raw), out)
    arms = ARMS if o.arms is None else [("", ARMS[0][1])] + [tuple(a.split("=", 1)) for a in o.arms]
    learner_part(Path(o.raw), out, arms)
    seeds_part(Path(o.raw), out)


if __name__ == "__main__":
    main()
