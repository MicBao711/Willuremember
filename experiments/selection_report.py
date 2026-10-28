"""Candidate selection inside the continual learner: probe vs successive halving vs full.

    python experiments/selection_report.py --out results/selection

Reads results/raw/mnist_compose_n<N>/ (probe, 100 steps), _n<N>_probewarm/ (the same probe,
but the chosen candidate keeps its optimizer: `--selection halving --halving-eta 10`),
_n<N>_halving/ and _n<N>_full/, pairs runs by seed and writes learner.md into --out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compose_report import concept_choices, paired, two_image  # noqa: E402
from plot_cl import LABELS, load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [("", "探测 100 步（v0.2）"), ("_probewarm", "探测 100 步，不重置优化器"),
            ("_halving", "逐步淘汰（50 步起，每轮减半）"), ("_full", "全部训满再挑")]
METHODS = ["vcm_freeze", "vcm_freeze_promo_comp", "vcm_lazy", "vcm_lazy_promo_comp"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[100, 250, 1000, 4000])
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "results" / "selection"))
    o = ap.parse_args()
    raw, out = Path(o.raw), Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {(v, n): load(raw / f"mnist_compose_n{n}{v}") for v, _ in VARIANTS for n in o.sizes
            if (raw / f"mnist_compose_n{n}{v}").exists()}
    lines = ["# 持续学习器里的候选选择：探测 / 逐步淘汰 / 全部训满\n",
             "组合式 MNIST，5 个种子，测试集。“两图”= sum / equal / greater / diff 刚学完时的准确率均值；"
             "“选概念”= 4 个两图任务里选了晋升出的整条组合的平均次数；计算是整个任务流的总乘加次数。\n",
             "## 两图任务准确率\n",
             "| 方法 | 选择方式 | " + " | ".join(f"{n} 个样本" for n in o.sizes) + " | 选概念（各样本数） | 计算 (GMAC，1000 个样本) |",
             "|---" * (len(o.sizes) + 4) + "|"]
    for m in METHODS:
        for v, label in VARIANTS:
            cells, uses = [], []
            for n in o.sizes:
                rs = runs.get((v, n), {}).get(m)
                if not rs:
                    cells.append("–")
                    uses.append("–")
                    continue
                a = np.array([two_image(r) for r in rs])
                cells.append(f"{a.mean():.3f} ± {a.std():.3f}")
                uses.append(f"{np.mean([concept_choices(r) for r in rs]):.1f}" if "promo_comp" in m else "–")
            rs = runs.get((v, 1000), {}).get(m)
            comp = f"{np.mean([r['compute']['total'] for r in rs]) / 1e9:.0f}" if rs else "–"
            lines.append(f"| {LABELS.get(m, m)} | {label} | " + " | ".join(cells) + f" | {' / '.join(uses)} | {comp} |")
    lines += ["", "## 与“探测 100 步”的配对差（同一方法、同一种子；括号里是差值为正的种子数）\n",
              "| 方法 | 选择方式 | " + " | ".join(f"{n} 个样本" for n in o.sizes) + " |",
              "|---" * (len(o.sizes) + 2) + "|"]
    for m in METHODS:
        for v, label in VARIANTS[1:]:
            cells = []
            for n in o.sizes:
                a, b = runs.get((v, n), {}).get(m), runs.get(("", n), {}).get(m)
                if a and b:
                    d = paired(a, b)
                    cells.append(f"{d.mean():+.3f} ± {d.std():.3f}（{int((d > 0).sum())}/{len(d)}）")
                else:
                    cells.append("–")
            lines.append(f"| {LABELS.get(m, m)} | {label} | " + " | ".join(cells) + " |")
    lines += ["", "## 晋升的效果在不同选择方式下（整条组合晋升 − 不晋升，配对）\n",
              "| 模式 | 选择方式 | " + " | ".join(f"{n} 个样本" for n in o.sizes) + " |",
              "|---" * (len(o.sizes) + 2) + "|"]
    for a_m, b_m, mode in (("vcm_freeze_promo_comp", "vcm_freeze", "freeze"),
                           ("vcm_lazy_promo_comp", "vcm_lazy", "update + 惰性")):
        for v, label in VARIANTS:
            cells = []
            for n in o.sizes:
                a, b = runs.get((v, n), {}).get(a_m), runs.get((v, n), {}).get(b_m)
                if a and b:
                    d = paired(a, b)
                    cells.append(f"{d.mean():+.3f} ± {d.std():.3f}（{int((d > 0).sum())}/{len(d)}）")
                else:
                    cells.append("–")
            lines.append(f"| {mode} | {label} | " + " | ".join(cells) + " |")
    (out / "learner.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
