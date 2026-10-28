"""Falsification test: does a function-space judge beat DEN-style weight-space judges?

When a shared unit is updated in place, VCM lets each old task decide, when it is next
queried, whether to take the new version or stay on its pinned old snapshot. The judge
that decides is the only thing that differs between the configurations here:

  function       accuracy drop on the task's 200-sample feedback buffer <= tolerance
  weight_rel     relative weight change of the stale unit(s) <= threshold      (no data)
  weight_neuron  largest L2 change of one neuron's incoming weights <= threshold (DEN's drift)

plus two shared end points, "keep" (never take a new version, = freezing for old tasks)
and "accept" (always take it). Each judge sweeps 5 thresholds at the 10/30/50/70/90%
quantiles of its own statistic in a pilot run (validation split, always-accept, seed 0),
so every family spans the range from keeping almost everything to accepting almost
everything. What is compared is the trade-off between forgetting and the old snapshots a
judge keeps alive (memory), measured after all pending lazy checks are settled.

    python experiments/judge_study.py grid --pilot <dir with pilot JSON>   # thresholds (tsv)
    python experiments/judge_study.py report                               # tables, figure

The commands behind the results are `run_suite.sh judge_pilot` and `run_suite.sh judge`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = [("function", "drop_feedback", "函数空间（反馈样本准确率降幅）"),
            ("weight_rel", "rel", "权重空间：整层相对变化"),
            ("weight_neuron", "neuron", "权重空间：单个神经元的最大变化（DEN）")]
QUANTILES = (10, 30, 50, 70, 90)
SETTINGS = [("pmnist_stress", "Permuted MNIST（10 任务，共享层以全学习率原地更新）"),
            ("cifar100_stress", "Split CIFAR-100（10 任务，共享层以全学习率原地更新）"),
            ("cifar100", "Split CIFAR-100（默认：共享层以 0.1 倍学习率更新）")]
PLOT_TITLES = {"pmnist_stress": "Permuted MNIST, shared units at full LR",
               "cifar100_stress": "Split CIFAR-100, shared units at full LR",
               "cifar100": "Split CIFAR-100, shared units at 0.1x LR (default)"}
PRIMARY = ("pmnist_stress", "cifar100_stress")   # the decision rule uses these two
HARM = 0.02                                       # "real damage": held-out drop above this
COLORS = {"function": "#2a78d6", "weight_rel": "#eb6834", "weight_neuron": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# -- thresholds from the pilot ------------------------------------------------------------
def grid(pilot: Path) -> None:
    """Print `setting config judge flag value` lines for run_suite.sh, primary settings first."""
    lines = {}
    for f in sorted(pilot.rglob("*.json")):
        run = json.loads(f.read_text())
        setting = run["args"]["tag"].replace("pilot_", "").replace("pilot", "")
        setting = {"stress": f"{_short(run['benchmark'])}_stress", "": _short(run["benchmark"])}[setting]
        log = run["info"]["judge_log"]
        out = [f"{setting}\tkeep\tfunction\t--tolerance\t-1", f"{setting}\taccept\tfunction\t--tolerance\t1"]
        for judge, stat, _ in FAMILIES:
            values = np.array([r[stat] for r in log])
            for q in QUANTILES:
                v = float(np.percentile(values, q))
                flag = "--tolerance" if judge == "function" else "--drift-threshold"
                out.append(f"{setting}\t{judge}_q{q}\t{judge}\t{flag}\t{v:.6g}")
        lines[setting] = out
    for setting, _ in SETTINGS:
        print("\n".join(lines.get(setting, [])))


def _short(benchmark: str) -> str:
    return {"pmnist": "pmnist", "split_cifar100": "cifar100"}[benchmark]


# -- report -------------------------------------------------------------------------------
def load(setting: Path) -> dict[str, dict[int, dict]]:
    runs = defaultdict(dict)
    for f in sorted(setting.rglob("*.json")):
        run = json.loads(f.read_text())
        runs[run["args"]["tag"]][run["seed"]] = run
    return dict(runs)


def point(run) -> dict:
    m = run["metrics"]
    settled = run["settled"]["retained_old_floats"]
    return {"acc": m["acc"], "forgetting": m["forgetting"], "mem": settled,
            "params": run["memory"]["params"] - run["info"]["retained_old_floats"] + settled,
            "maint": run["compute"].get("maintenance", 0.0), "decisions": run["settled"]["decisions"]}


def frontier_area(points: list[tuple[float, float]], lower_is_better: bool = True) -> float:
    """Area under the Pareto frontier of (normalised memory, y) over memory in [0, 1]:
    keep the non-dominated points, join them linearly, integrate (trapezoids)."""
    pts = sorted(points)
    better = (lambda a, b: a < b) if lower_is_better else (lambda a, b: a > b)
    front = []
    for x, y in pts:   # increasing memory: keep a point only if it improves on everything cheaper
        if not front or better(y, front[-1][1]):
            front.append((x, y))
    xs, ys = np.array([p[0] for p in front]), np.array([p[1] for p in front])
    if xs[-1] < 1.0:   # flat to the right: more memory cannot do worse than the best point
        xs, ys = np.append(xs, 1.0), np.append(ys, ys[-1])
    return float(np.trapezoid(ys, xs))


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """P(score of a harmful update > score of a harmless one), ties counted half."""
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)


def family(runs, judge: str) -> list[str]:
    """Configurations of one judge: the shared end points, its pre-registered quantile
    thresholds (`_q`) and any post-hoc thresholds (`_t`)."""
    return [c for c in ("accept", "keep") if c in runs] + \
        [c for c in runs if c.startswith((f"{judge}_q", f"{judge}_t"))]


def curve(runs, s: int, keep_mem: float, configs, metric: str) -> list[tuple[float, float]]:
    return [(point(runs[c][s])["mem"] / keep_mem if keep_mem else 0.0, point(runs[c][s])[metric])
            for c in configs if c in runs and s in runs[c]]


def at_memory(points, m: float) -> float:
    """Forgetting of the Pareto frontier (non-dominated points, joined linearly) at memory m."""
    front = []
    for x, y in sorted(points):
        if not front or y < front[-1][1]:
            front.append((x, y))
    xs, ys = [p[0] for p in front], [p[1] for p in front]
    return float(np.interp(m, xs, ys))


def report(raw: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# 证伪实验：函数空间裁判 vs DEN 式权重空间裁判\n",
             "VCM（update + 惰性），只换“旧任务是否接受共享单元的新版本”的裁判；其余设置一律相同（反馈缓冲区每任务 200 个验证样本、不修复、探测选组合）。"
             "“保留内存”= 跑完后把所有待检查的旧任务结算一遍、仍被钉住的旧快照的浮点数，按同种子“永远退回”归一化。"
             "遗忘、最终 ACC 在测试集上测；均值 ± 标准差跨种子。事先登记的判据见 `plan.md`。\n"]
    summary = {}
    for setting, title in SETTINGS:
        runs = load(raw / setting)
        if not runs:
            continue
        seeds = sorted(set.intersection(*(set(r) for r in runs.values())))
        keep_mem = {s: point(runs["keep"][s])["mem"] for s in seeds}
        lines += [f"## {title}（{len(seeds)} 个种子）\n",
                  "| 裁判 | 阈值 | 最终 ACC | 遗忘 | 保留内存（归一化） | 参数总量 (M) | 接受 / 保留 | 维护计算 (GMAC) |",
                  "|---|---|---|---|---|---|---|---|"]
        configs = ["accept"] + [f"{j}_q{q}" for j, _, _ in FAMILIES for q in QUANTILES] + ["keep"]
        for c in configs:
            if c not in runs:
                continue
            pts = [point(runs[c][s]) for s in seeds]
            a = np.array([p["acc"] for p in pts])
            f = np.array([p["forgetting"] for p in pts])
            mem = np.array([p["mem"] / keep_mem[s] if keep_mem[s] else 0.0 for p, s in zip(pts, seeds)])
            dec = np.mean([[p["decisions"]["accept"], p["decisions"]["keep"]] for p in pts], axis=0)
            args = runs[c][seeds[0]]["args"]
            thr = args["tolerance"] if args.get("judge", "function") == "function" else args["drift_threshold"]
            lines.append(f"| {c} | {thr:.4g} | {a.mean():.4f} ± {a.std():.4f} | {f.mean():.4f} ± {f.std():.4f} | "
                         f"{mem.mean():.2f} | {np.mean([p['params'] for p in pts]) / 1e6:.2f} | {dec[0]:.1f} / {dec[1]:.1f} | "
                         f"{np.mean([p['maint'] for p in pts]) / 1e9:.1f} |")
        # frontier areas per seed
        areas = {}
        for j, _, _ in FAMILIES:
            for metric, low in (("forgetting", True), ("acc", False)):
                areas[(j, metric)] = np.array([frontier_area(curve(runs, s, keep_mem[s], family(runs, j), metric),
                                                             lower_is_better=low) for s in seeds])
        lines += ["", "帕累托前沿下的面积（遗忘：越小越好；ACC：越大越好），逐种子：\n",
                  "| 裁判 | 遗忘面积 | ACC 面积 |", "|---|---|---|"]
        for j, _, label in FAMILIES:
            fa, aa = areas[(j, "forgetting")], areas[(j, "acc")]
            lines.append(f"| {label} | {fa.mean():.4f}（{', '.join(f'{x:.4f}' for x in fa)}） | "
                         f"{aa.mean():.4f}（{', '.join(f'{x:.4f}' for x in aa)}） |")
        verdict = {}
        for j in ("weight_rel", "weight_neuron"):
            d = areas[(j, "forgetting")] - areas[("function", "forgetting")]   # > 0: function better
            rel = d.mean() / areas[(j, "forgetting")].mean() if areas[(j, "forgetting")].mean() > 0 else 0.0
            verdict[j] = bool((d > 0).all() and rel >= 0.2)
            lines.append(f"\n函数空间对 {j}：遗忘面积减少 {d.mean():+.4f}（相对 {rel:.0%}），{int((d > 0).sum())}/{len(d)} 个种子更好 → "
                         f"{'函数空间更好' if verdict[j] else '未达到判据'}")
        summary[setting] = verdict
        # does each statistic predict real damage?
        log = [r for c in runs for s in seeds for r in runs[c][s]["info"].get("judge_log", [])]
        if log:
            harm = np.array([r["drop_heldout"] > HARM for r in log])
            lines += ["", f"每种统计量能否预测“真实损伤”（留出样本上准确率下降 > {HARM}）：全部 {len(log)} 次裁判调用，"
                          f"其中 {int(harm.sum())} 次有真实损伤。AUROC：\n",
                      "| 统计量 | AUROC |", "|---|---|"]
            for j, stat, label in FAMILIES:
                lines.append(f"| {label} | {auroc(np.array([r[stat] for r in log]), harm):.3f} |")
        lines.append("")
    if summary:
        ok = all(all(summary[s].values()) for s in PRIMARY if s in summary)
        lines += ["## 判定\n",
                  f"事先登记的判据（两个压力设置上，函数空间的遗忘面积都比两种权重空间裁判小，3/3 个种子、相对至少 20%）："
                  f"**{'成立' if ok and all(s in summary for s in PRIMARY) else '不成立'}**。\n",
                  "![frontier](frontier.png)\n"]
    post = raw.parent / "judge_posthoc" / "cifar100_stress"
    if load(post):
        lines += posthoc(raw / "cifar100_stress", post, out / "frontier_posthoc.png")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    plot(raw, out / "frontier.png")


def posthoc(pre: Path, post: Path, path: Path) -> list[str]:
    """CIFAR-100 stress with the post-hoc points added (see results/judge/posthoc.md)."""
    runs = {**load(pre), **load(post)}
    seeds = sorted(set.intersection(*(set(r) for r in runs.values())))
    keep_mem = {s: point(runs["keep"][s])["mem"] for s in seeds}
    lines = ["## 事后补点：CIFAR-100 压力设置（不改变上面的判定）\n",
             "预登记的网格在这个设置上让两族各空了一段内存区间（函数空间缺 0.66–1.0，权重空间缺 0–0.72），面积比较主要在比插值。"
             "这里对称地补了点（动机与阈值见 `posthoc.md`，在运行前提交），合并后重算。\n",
             "| 补的配置 | 最终 ACC | 遗忘 | 保留内存（归一化） |", "|---|---|---|---|"]
    for c in sorted(c for c in load(post)):
        pts = [point(runs[c][s]) for s in seeds]
        mem = np.mean([p["mem"] / keep_mem[s] if keep_mem[s] else 0.0 for p, s in zip(pts, seeds)])
        lines.append(f"| {c} | {np.mean([p['acc'] for p in pts]):.4f} | {np.mean([p['forgetting'] for p in pts]):.4f} | {mem:.2f} |")
    area = {j: np.array([frontier_area(curve(runs, s, keep_mem[s], family(runs, j), "forgetting")) for s in seeds])
            for j, _, _ in FAMILIES}
    lines += ["", "| 裁判 | 遗忘面积（合并后，逐种子） |", "|---|---|"]
    for j, _, label in FAMILIES:
        lines.append(f"| {label} | {area[j].mean():.4f}（{', '.join(f'{x:.4f}' for x in area[j])}） |")
    for j in ("weight_rel", "weight_neuron"):
        d = area[j] - area["function"]
        lines.append(f"\n函数空间对 {j}：遗忘面积减少 {d.mean():+.4f}（相对 {d.mean() / area[j].mean():.0%}），"
                     f"{int((d > 0).sum())}/{len(d)} 个种子更好")
    levels = (0.2, 0.4, 0.6, 0.8, 0.9)
    lines += ["", "同样内存下的遗忘（各种子的前沿在该内存处插值后取平均）：\n",
              "| 裁判 | " + " | ".join(f"内存 {m}" for m in levels) + " |", "|---" * (len(levels) + 1) + "|"]
    for j, _, label in FAMILIES:
        cells = [np.mean([at_memory(curve(runs, s, keep_mem[s], family(runs, j), "forgetting"), m) for s in seeds])
                 for m in levels]
        lines.append(f"| {label} | " + " | ".join(f"{v:.4f}" for v in cells) + " |")
    lines += ["", "![frontier, post hoc](frontier_posthoc.png)\n"]
    plot_posthoc(runs, set(load(post)), seeds, keep_mem, path)
    return lines


def plot_posthoc(runs, extra: set, seeds, keep_mem, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    mean = lambda c, k: np.mean([point(runs[c][s])[k] / (keep_mem[s] or 1.0 if k == "mem" else 1.0) for s in seeds])
    for j, _, _ in FAMILIES:
        xy = sorted((mean(c, "mem"), mean(c, "forgetting"), c in extra) for c in family(runs, j))
        ax.plot([p[0] for p in xy], [p[1] for p in xy], "-", lw=2, color=COLORS[j])
        for x, y, new in xy:
            ax.plot(x, y, "o", ms=6, color="#fcfcfb" if new else COLORS[j], mec=COLORS[j], mew=1.8)
    ax.set_title("Split CIFAR-100, shared units at full LR (hollow: post-hoc points)", fontsize=9, color=INK)
    ax.set_xlabel("retained old snapshots (relative to always-keep)", fontsize=8, color=INK2)
    ax.set_ylabel("forgetting (test)", fontsize=8, color=INK2)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=COLORS[j], lw=2, marker="o", label=lab) for j, lab in
                       (("function", "function space (feedback accuracy)"),
                        ("weight_rel", "weight space: relative change"),
                        ("weight_neuron", "weight space: max neuron drift (DEN)"))],
              frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def plot(raw: Path, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    settings = [(s, t) for s, t in SETTINGS if load(raw / s)]
    if not settings:
        return
    fig, axes = plt.subplots(1, len(settings), figsize=(5 * len(settings), 3.8), squeeze=False)
    for ax, (setting, title) in zip(axes[0], settings):
        runs = load(raw / setting)
        seeds = sorted(set.intersection(*(set(r) for r in runs.values())))
        keep_mem = {s: point(runs["keep"][s])["mem"] for s in seeds}
        mean = lambda c, k: np.mean([point(runs[c][s])[k] / (keep_mem[s] or 1.0 if k == "mem" else 1.0)
                                     for s in seeds])
        for j, _, label in FAMILIES:
            xy = sorted((mean(c, "mem"), mean(c, "forgetting")) for c in family(runs, j))
            ax.plot([p[0] for p in xy], [p[1] for p in xy], "-o", lw=2, ms=6, color=COLORS[j],
                    mec="#fcfcfb", mew=1.5, label=label)
        ax.set_title(PLOT_TITLES[setting], fontsize=9, color=INK)
        ax.set_xlabel("retained old snapshots (relative to always-keep)", fontsize=8, color=INK2)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
    axes[0][0].set_ylabel("forgetting (test)", fontsize=8, color=INK2)
    handles = [Line2D([], [], color=COLORS[j], lw=2, marker="o", label=lab) for j, lab in
               (("function", "function space (feedback accuracy)"), ("weight_rel", "weight space: relative change"),
                ("weight_neuron", "weight space: max neuron drift (DEN)"))]
    axes[0][-1].legend(handles=handles, frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["grid", "report"])
    ap.add_argument("--pilot", default=str(ROOT / "results" / "raw" / "judge_pilot"))
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw" / "judge"))
    ap.add_argument("--out", default=str(ROOT / "results" / "judge"))
    o = ap.parse_args()
    if o.part == "grid":
        grid(Path(o.pilot))
    else:
        report(Path(o.raw), Path(o.out))


if __name__ == "__main__":
    main()
