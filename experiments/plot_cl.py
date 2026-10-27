"""Aggregate continual-learning runs into tables and figures.

    python experiments/plot_cl.py --raw results/raw/pmnist --out results/pmnist

Figures:
  curves.png   per method: accuracy on the task just learned (solid) and mean accuracy
               on all earlier tasks (dashed), on the same axes
  pareto.png   final average accuracy vs. total compute and vs. stored floats
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

LABELS = {
    "finetune": "Fine-tune", "ewc": "EWC", "er": "ER (replay)", "grace": "GRACE-CL",
    "independent": "Independent", "vcm_freeze": "VCM (freeze)", "vcm_lazy": "VCM (update, lazy)",
    "vcm_eager": "VCM (update, eager)", "vcm_lazy_norepair": "VCM (lazy, no repair)",
    "vcm_nojudge": "VCM (update, no judge)",
    "vcm_lazy_promo": "VCM (lazy + promotion, cut)", "vcm_lazy_promo_keep": "VCM (lazy + promotion, keep)",
    "vcm_freeze_promo": "VCM (freeze + promotion)",
}
COLORS = {
    "finetune": "#8a8a8a", "ewc": "#edae49", "er": "#66a182", "grace": "#9d4edd",
    "independent": "#2e4057", "vcm_freeze": "#00798c", "vcm_lazy": "#d1495b",
    "vcm_eager": "#f08a4b", "vcm_lazy_norepair": "#b56576", "vcm_nojudge": "#5c5c5c", "vcm_lazy_promo": "#3a86ff",
    "vcm_lazy_promo_keep": "#8ecae6", "vcm_freeze_promo": "#43aa8b",
}
ORDER = list(LABELS)


def load(raw: Path) -> dict[str, list[dict]]:
    runs = defaultdict(list)
    for f in sorted(raw.glob("*.json")):
        d = json.loads(f.read_text())
        runs[d["method"]].append(d)
    return dict(sorted(runs.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99))


def ms(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    return float(np.mean(v)), float(np.std(v)) if len(v) > 1 else 0.0


def fmt(values, scale=1.0, digits=3) -> str:
    m, s = ms(np.asarray(values) * scale)
    return f"{m:.{digits}f} ± {s:.{digits}f}" if s else f"{m:.{digits}f}"


def forward_transfer(rs, reference) -> list[float]:
    """Mean over tasks >= 1 of R_ii(method) - R_ii(reference), matched by seed."""
    ref = {r["seed"]: np.diag(np.array(r["R"], dtype=float)) for r in reference}
    out = []
    for r in rs:
        if r["seed"] in ref:
            d = np.diag(np.array(r["R"], dtype=float))
            out.append(float(np.mean(d[1:] - ref[r["seed"]][1:])))
    return out


def table(runs, title: str, reference=None) -> list[str]:
    ref = runs.get("independent") or reference
    lines = [f"## {title}\n",
             "| 方法 | 种子数 | 最终平均准确率 ACC | 学习准确率 LA | 前向迁移 FWT | BWT | 遗忘 | 计算量 (GMAC) | 参数 (M) | 缓冲等 (M) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m, rs in runs.items():
        mt = [r["metrics"] for r in rs]
        fwt = forward_transfer(rs, ref) if ref else []
        lines.append(
            f"| {LABELS.get(m, m)} | {len(rs)} | {fmt([x['acc'] for x in mt])} | "
            f"{fmt([x['learning_acc'] for x in mt])} | {fmt(fwt) if fwt else '–'} | "
            f"{fmt([x.get('bwt', 0) for x in mt])} | "
            f"{fmt([x.get('forgetting', 0) for x in mt])} | "
            f"{fmt([r['compute']['total'] for r in rs], 1e-9, 1)} | "
            f"{fmt([r['memory']['params'] for r in rs], 1e-6, 2)} | "
            f"{fmt([r['memory']['other'] for r in rs], 1e-6, 2)} |")
    return lines


def vcm_details(runs) -> list[str]:
    lines = []
    for m, rs in runs.items():
        if not m.startswith("vcm"):
            continue
        dec = defaultdict(list)
        for r in rs:
            for k, v in r["info"]["decisions"].items():
                dec[k].append(v)
            dec["maintenance_gmac"].append(r["compute"].get("maintenance", 0) / 1e9)
            dec["probe_gmac"].append((r["compute"].get("probe_eval", 0)) / 1e9)
            dec["retained_snapshots"].append(r["info"]["retained_old_snapshots"])
            dec["concepts"].append(len(r["info"]["concepts"]))
            if "router_acc" in r:
                dec["router_acc"].append(r["router_acc"])
        plans = defaultdict(int)
        for r in rs:
            for p in r["info"]["plans"].values():
                plans[p.split("<-")[0].split(" ")[0]] += 1
        parts = [f"{k}={fmt(v, digits=2)}" for k, v in dec.items()]
        lines.append(f"- **{LABELS.get(m, m)}**: " + ", ".join(parts)
                     + "；方案分布 " + ", ".join(f"{k}×{v}" for k, v in sorted(plans.items())))
    return lines


MAIN = ["finetune", "ewc", "er", "grace", "independent", "vcm_freeze", "vcm_lazy"]


def _curves(ax, runs, methods) -> list[float]:
    values = []
    for m in methods:
        rs = runs[m]
        new = np.array([r["metrics"]["new_task_curve"] for r in rs]).mean(0)
        old = np.array([r["metrics"]["old_task_curve"] for r in rs]).mean(0)
        T = len(new)
        c = COLORS.get(m, None)
        ax.plot(range(1, T + 1), new, "-", color=c, lw=1.8, label=LABELS.get(m, m))
        ax.plot(range(2, T + 1), old, "--", color=c, lw=1.8)
        values += list(new) + list(old)
    ax.set_xlabel("number of tasks learned")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.grid(alpha=0.3)
    return values


def plot_curves(runs, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main = [m for m in MAIN if m in runs]
    variants = [m for m in runs if m.startswith("vcm") or m == "independent"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1]})
    _curves(axes[0], runs, main)
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("baselines vs. VCM")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    vals = _curves(axes[1], runs, variants)
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    pad = max(0.003, 0.08 * (hi - lo))
    axes[1].set_ylim(lo - pad, hi + pad)
    axes[1].set_title("VCM variants (zoomed)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle(f"{title}  —  solid: task just learned (new), dashed: mean over earlier tasks (old)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pareto(runs, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for m, rs in runs.items():
        acc = [r["metrics"]["acc"] for r in rs]
        comp = [r["compute"]["total"] / 1e9 for r in rs]
        mem = [sum(r["memory"].values()) / 1e6 for r in rs]
        c = COLORS.get(m, None)
        axes[0].errorbar(np.mean(comp), np.mean(acc), yerr=np.std(acc), fmt="o", color=c, label=LABELS.get(m, m))
        axes[1].errorbar(np.mean(mem), np.mean(acc), yerr=np.std(acc), fmt="o", color=c)
    axes[0].set_xlabel("total compute (GMAC)")
    axes[1].set_xlabel("stored floats (millions)")
    axes[0].set_ylabel("final average accuracy")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--reference", default=None,
                    help="raw dir whose 'independent' runs serve as the FWT reference")
    o = ap.parse_args()
    runs = load(Path(o.raw))
    out = Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    title = o.title or Path(o.raw).name
    plot_curves(runs, out / "curves.png", title)
    plot_pareto(runs, out / "pareto.png", title)
    any_run = next(iter(runs.values()))[0]
    lines = [f"# {title}\n",
             f"参数：`{ {k: v for k, v in any_run['args'].items() if k not in ('out', 'methods', 'seeds')} }`，"
             f"每任务 epoch = {any_run['epochs']}。\n",
             *table(runs, "汇总（均值 ± 标准差，跨种子）",
                    load(Path(o.reference)).get("independent") if o.reference else None), "",
             "- ACC：学完全部任务后在所有任务上的平均准确率；LA：每个任务刚学完时的准确率均值；"
             "FWT：第 2 个任务起，刚学完时的准确率减去同种子 Independent（每任务一个全新网络）的对应值，再取均值；"
             "BWT：最终准确率减去刚学完时准确率的均值（负数 = 遗忘）；遗忘：各旧任务历史最高准确率与最终准确率之差的均值。",
             "- 计算量按乘加次数估计（一次训练样本 ≈ 3 次前向），包括训练、探测、维护（版本检查后的验证与修复）、服务查询。",
             "- 参数：所有保留的参数（VCM 含被钉住的旧快照与晋升出的概念副本）；缓冲等：回放/反馈缓冲区、Fisher 与锚点、GRACE 码本、任务质心等，单位百万个浮点数。\n",
             "## VCM 内部统计\n", *vcm_details(runs), "",
             "![curves](curves.png)\n", "![pareto](pareto.png)\n"]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
