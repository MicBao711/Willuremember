"""Validation selections for compositional MNIST (see `run_suite.sh compose_val`).

    python experiments/compose_val.py

Reads results/raw/sweeps/mnist_compose_* and writes results/sweeps/compose.md. Scores are
the mean validation accuracy, right after learning, of the two-image tasks (sum, equal,
greater, diff), except for EWC, which is selected on the final average accuracy like
the other benchmarks.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SW = ROOT / "results" / "raw" / "sweeps"


def runs(pattern: str):
    for d in sorted(SW.glob(pattern)):
        for f in sorted(d.glob("*.json")):
            yield d.name.replace("mnist_compose_", ""), json.loads(f.read_text())


def two_image(r) -> float:
    return float(np.diag(np.array(r["R"], dtype=float))[2:].mean())


def main() -> None:
    lines = ["# 组合式 MNIST：验证集上的选择\n",
             "所有数字都在验证集上测得（`--eval-split val`）；分数 = 两图任务（sum / equal / greater / diff）刚学完时的平均准确率。\n",
             "## 1. 每任务最少训练步数 `--min-steps`（种子 0）\n",
             "| 后续任务样本数 | min_steps | Independent | Fine-tune | VCM (freeze) | VCM (freeze + 组合晋升) | 均值 |",
             "|---|---|---|---|---|---|---|"]
    table = defaultdict(dict)
    for tag, r in runs("mnist_compose_n*_s*"):
        n, s = tag.split("_")
        table[(int(n[1:]), int(s[1:]))][r["method"]] = two_image(r)
    order = ["independent", "finetune", "vcm_freeze", "vcm_freeze_promo_comp"]
    for key in sorted(table):
        row = table[key]
        lines.append(f"| {key[0]} | {key[1]} | " + " | ".join(f"{row[m]:.3f}" for m in order)
                     + f" | {np.mean([row[m] for m in order]):.3f} |")
    means = defaultdict(list)
    for (n, s), row in table.items():
        means[s].append(np.mean([row[m] for m in order]))
    lines += ["", "按 min_steps 平均：" + "，".join(f"{s}: {np.mean(v):.3f}" for s, v in sorted(means.items()))
              + "。三者相差不到 0.5 个百分点（与噪声同级），固定为中间值 **500**，之后的探测步数选择也在 500 步下做。"
              "（若套用下一节的规则“离最好不到 0.5 个百分点里取最省的”，会选 250；这个规则是在固定 500 之后才定下的，照实记录。）\n",
              "## 2. 候选探测步数 `--probe-steps`（种子 0、1）\n",
              "| 后续任务样本数 | 探测步数 | VCM (freeze) | VCM (freeze + 组合晋升) | 差 | 计算 (GMAC，两者平均) |",
              "|---|---|---|---|---|---|"]
    probe = defaultdict(lambda: defaultdict(list))
    for tag, r in runs("mnist_compose_n*_p*"):
        n, p = tag.split("_")
        probe[(int(n[1:]), int(p[1:]))][r["method"]].append((two_image(r), r["compute"]["total"] / 1e9))
    score = defaultdict(list)
    for key in sorted(probe):
        a = np.array(probe[key]["vcm_freeze"])
        b = np.array(probe[key]["vcm_freeze_promo_comp"])
        lines.append(f"| {key[0]} | {key[1]} | {a[:, 0].mean():.3f} | {b[:, 0].mean():.3f} | "
                     f"{b[:, 0].mean() - a[:, 0].mean():+.3f} | {np.r_[a[:, 1], b[:, 1]].mean():.0f} |")
        score[key[1]].append((a[:, 0].mean() + b[:, 0].mean()) / 2)
    best = max(np.mean(v) for v in score.values())
    chosen = min(p for p, v in score.items() if np.mean(v) >= best - 0.005)
    lines += ["", "按探测步数平均：" + "，".join(f"{p}: {np.mean(v):.3f}" for p, v in sorted(score.items()))
              + f"。规则：离最好不到 0.5 个百分点的里面取最省的 → **{chosen}**。\n",
              "## 3. EWC 的 λ（种子 0，后续任务 1000 个样本；按最终平均准确率）\n",
              "| λ | ACC | 学习准确率 | BWT | 两图任务 |", "|---|---|---|---|---|"]
    ewc = sorted(((r["args"]["ewc_lambda"], r) for _, r in runs("mnist_compose_ewc*")), key=lambda t: t[0])
    for lam, r in ewc:
        m = r["metrics"]
        lines.append(f"| {lam:g} | {m['acc']:.4f} | {m['learning_acc']:.4f} | {m['bwt']:+.4f} | {two_image(r):.3f} |")
    lam = max(ewc, key=lambda t: t[1]["metrics"]["acc"])[0]
    lines += ["", f"选定 λ = **{lam:g}**。λ 这么大时，共享的读图层学完第一个任务后几乎不再动，"
              "EWC 实际上变成“冻结读图器 + 每任务新头”。\n"]
    out = ROOT / "results" / "sweeps" / "compose.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
