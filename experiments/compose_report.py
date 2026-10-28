"""Tables and figures for the compositional MNIST suite (promotion -> forward transfer?).

    python experiments/compose_report.py --sizes 100 250 1000 4000 --out results/compose

Reads results/raw/mnist_compose_n<N>/ (and _n<N>_p50/ for the short-probe ablation, if
present) and results/compose_readers/readers.json (forced readers, if present).
Writes summary.md and scaling.png into --out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_cl import COLORS, LABELS, load  # noqa: E402

from vcm.cl.data import COMPOSE_TASKS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BINARY = slice(2, None)   # tasks 2.. (sum, equal, greater, diff) read two images
PAIRS = [("vcm_freeze_promo_comp", "vcm_freeze", "整条组合晋升 − 不晋升（freeze）"),
         ("vcm_freeze", "mntdp", "VCM 逐层复用（freeze）− MNTDP-D"),
         ("vcm_freeze_promo_comp", "mntdp", "VCM 整条组合晋升（freeze）− MNTDP-D"),
         ("vcm_freeze_promo", "vcm_freeze", "v0.1 共享块晋升 − 不晋升（freeze）"),
         ("vcm_freeze_promo_comp0", "vcm_freeze_promo_comp", "立即晋升（无门槛）− 按门槛晋升"),
         ("vcm_lazy_promo_comp", "vcm_lazy", "整条组合晋升 − 不晋升（update + 惰性）"),
         ("vcm_lazy_promo_comp_keep", "vcm_lazy_promo_comp", "保持依赖 − 切断依赖（update）")]
CURVE = ["independent", "finetune", "er", "mntdp", "vcm_freeze", "vcm_freeze_promo", "vcm_freeze_promo_comp",
         "vcm_lazy", "vcm_lazy_promo_comp"]


def diag(run) -> np.ndarray:
    return np.diag(np.array(run["R"], dtype=float))


def two_image(run) -> float:
    return float(diag(run)[BINARY].mean())


def by_seed(runs) -> dict[int, dict]:
    return {r["seed"]: r for r in runs}


def ms(v, digits=3) -> str:
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.{digits}f} ± {v.std():.{digits}f}" if len(v) > 1 else f"{v.mean():.{digits}f}"


def concept_choices(run) -> int:
    """Two-image tasks whose chosen plan put a promoted composition into the reader slots."""
    chosen = {f"concept {c}" for c in run["info"].get("composition_concepts", {})}
    return sum(p in chosen for k, p in run["info"].get("plans", {}).items() if int(k) >= 2)


def paired(a_runs, b_runs) -> np.ndarray:
    a, b = by_seed(a_runs), by_seed(b_runs)
    return np.array([two_image(a[s]) - two_image(b[s]) for s in sorted(set(a) & set(b))])


def main_table(runs, n) -> list[str]:
    ref = by_seed(runs.get("independent", []))
    lines = [f"### 每个后续任务 {n} 个训练样本\n",
             "| 方法 | 两图任务学习准确率 | 两图 FWT（对 Independent） | parity 学习准确率 | 最终平均 ACC | 遗忘 | 计算 (GMAC) | 参数 (M) | 两图任务选用概念 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m, rs in runs.items():
        fwt = [float((diag(r)[BINARY] - diag(ref[r["seed"]])[BINARY]).mean()) for r in rs if r["seed"] in ref]
        uses = [concept_choices(r) for r in rs] if m.startswith("vcm") else []
        lines.append(
            f"| {LABELS.get(m, m)} | {ms([two_image(r) for r in rs])} | {ms(fwt) if fwt else '–'} | "
            f"{ms([diag(r)[1] for r in rs])} | {ms([r['metrics']['acc'] for r in rs])} | "
            f"{ms([r['metrics'].get('forgetting', 0) for r in rs])} | "
            f"{np.mean([r['compute']['total'] for r in rs]) / 1e9:.0f} | "
            f"{np.mean([r['memory']['params'] for r in rs]) / 1e6:.2f} | "
            + (f"{np.mean(uses):.1f} / 4" if uses else "–") + " |")
    return lines


def per_task(runs, n, methods) -> list[str]:
    names = list(COMPOSE_TASKS)
    lines = [f"### 每个任务刚学完时的准确率（{n} 个样本）\n",
             "| 方法 | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for m in methods:
        if m in runs:
            D = np.array([diag(r) for r in runs[m]])
            lines.append(f"| {LABELS.get(m, m)} | " + " | ".join(ms(D[:, j]) for j in range(D.shape[1])) + " |")
    return lines


def plot(all_runs, sizes, readers, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for m in CURVE:
        pts = [(n, [two_image(r) for r in all_runs[n][m]]) for n in sizes if m in all_runs[n]]
        if pts:
            axes[0].errorbar([p[0] for p in pts], [np.mean(p[1]) for p in pts], yerr=[np.std(p[1]) for p in pts],
                             marker="o", ms=3, lw=1.6, capsize=2, color=COLORS.get(m), label=LABELS.get(m, m))
    axes[0].set_xscale("log")
    axes[0].set_xlabel("training samples per later task")
    axes[0].set_ylabel("two-image tasks: accuracy right after learning")
    axes[0].set_title("continual learners")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=7)
    series = [("vcm_freeze_promo_comp", "vcm_freeze", "learner: promoted compositions − none (freeze)", "#3d348b"),
              ("vcm_lazy_promo_comp", "vcm_lazy", "learner: promoted compositions − none (update)", "#7209b7")]
    for a, b, label, c in series:
        pts = [(n, paired(all_runs[n][a], all_runs[n][b])) for n in sizes if a in all_runs[n] and b in all_runs[n]]
        if pts:
            axes[1].errorbar([p[0] for p in pts], [p[1].mean() for p in pts], yerr=[p[1].std() for p in pts],
                             marker="o", ms=3, lw=1.6, capsize=2, color=c, label=label)
    if readers:
        rs = readers["rows"]
        ns = sorted({r["n"] for r in rs})
        seeds = sorted({r["seed"] for r in rs})

        def seed_mean(n, kind, s):
            return np.mean([r["acc"] for r in rs if r["n"] == n and r["reader"] == kind and r["seed"] == s])

        d = [np.array([seed_mean(n, "concept", s) - seed_mean(n, "layerwise", s) for s in seeds]) for n in ns]
        axes[1].errorbar(ns, [x.mean() for x in d], yerr=[x.std() for x in d], marker="s", ms=3, lw=1.4,
                         ls="--", capsize=2, color="#ff006e", label="forced: concept − layer-wise reader")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("training samples per later task")
    axes[1].set_ylabel("difference in accuracy (paired by seed)")
    axes[1].set_title("what promotion adds over layer-wise reuse")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[100, 250, 1000, 4000])
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw"))
    ap.add_argument("--readers", default=str(ROOT / "results" / "compose_readers" / "readers.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "compose"))
    o = ap.parse_args()
    out = Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = Path(o.raw)
    all_runs = {n: load(raw / f"mnist_compose_n{n}") for n in o.sizes}
    short = {n: load(raw / f"mnist_compose_n{n}_p50") for n in o.sizes if (raw / f"mnist_compose_n{n}_p50").exists()}
    readers = json.loads(Path(o.readers).read_text()) if Path(o.readers).exists() else None
    any_run = next(iter(all_runs[o.sizes[0]].values()))[0]
    lines = ["# 组合式 MNIST：晋升能否带来正向迁移\n",
             f"参数：`{ {k: v for k, v in any_run['args'].items() if k not in ('out', 'methods', 'seeds', 'tag', 'train_per_task')} }`，"
             f"每任务 epoch = {any_run['epochs']}（训练样本少时按 `min_steps` 补足步数）。"
             "任务顺序：digit（读数字，27000 张图）→ parity → sum → equal → greater → diff；后 5 个任务的样本数见各表。\n",
             "## 晋升 − 不晋升：两图任务学习准确率的配对差（同一种子）\n",
             "| 比较 | " + " | ".join(f"{n} 个样本" for n in o.sizes) + " |", "|---" * (len(o.sizes) + 1) + "|"]
    for a, b, label in PAIRS:
        cells = []
        for n in o.sizes:
            if a in all_runs[n] and b in all_runs[n]:
                d = paired(all_runs[n][a], all_runs[n][b])
                cells.append(f"{d.mean():+.3f} ± {d.std():.3f}（{int((d > 0).sum())}/{len(d)}）")
            else:
                cells.append("–")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    if short:
        cells = []
        for n in o.sizes:
            s = short.get(n, {})
            if "vcm_freeze_promo_comp" in s and "vcm_freeze" in s:
                d = paired(s["vcm_freeze_promo_comp"], s["vcm_freeze"])
                cells.append(f"{d.mean():+.3f} ± {d.std():.3f}（{int((d > 0).sum())}/{len(d)}）")
            else:
                cells.append("–")
        lines.append("| 同上（freeze），但探测只有 50 步（v0.1 默认） | " + " | ".join(cells) + " |")
    lines += ["", "括号里是差值为正的种子数。“两图任务学习准确率”= sum / equal / greater / diff 四个任务刚学完时的测试准确率均值。\n",
              "## 各方法\n"]
    for n in o.sizes:
        lines += main_table(all_runs[n], n) + [""]
    for n in o.sizes:
        lines += per_task(all_runs[n], n, ["independent", "finetune", "er", "mntdp", "vcm_freeze", "vcm_freeze_promo_comp",
                                           "vcm_lazy", "vcm_lazy_promo_comp"]) + [""]
    lines += ["- 两图 FWT：两图任务刚学完时的准确率减去同种子 Independent（每任务一个全新网络）的对应值，再取均值。",
              "- 两图任务选用概念：4 个两图任务里，学习器最终选了“晋升出的整条组合”填读图槽的次数（种子平均）。",
              "- 计算量按乘加次数估计（一次训练样本 ≈ 3 次前向），包括训练、候选探测、维护与服务查询；参数含被钉住的旧快照与概念副本。\n",
              "![scaling](scaling.png)\n"]
    plot(all_runs, o.sizes, readers, out / "scaling.png")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
