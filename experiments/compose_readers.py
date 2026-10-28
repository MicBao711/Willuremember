"""Forced readers on the two-image tasks of compositional MNIST.

    python experiments/compose_readers.py --seeds 0 1 2 3 4

The continual learner decides by itself which composition to use, so its numbers mix
two questions: is a promoted concept a better building block, and does the learner pick
it? This script answers the first one alone. A digit network is trained on the first
task (as in the learner), then every two-image task is learned with the reader fixed:

  concept    the whole digit network, output layer included (what promotion offers)
  layerwise  its first two layers, frozen (what positional layer-wise reuse can take)
  first      its first layer only, then a new per-image layer (layer-wise reuse when the
             new template's reader does not repeat the old network's roles)
  finetune   its first two layers as initialisation, trained further (ordinary transfer)
  scratch    a new network (as Independent)

Every variant trains the same number of steps (as `--min-steps` in run_cl.py) and is
measured on the test pairs. Writes results/compose_readers/{readers.json, readers.md, readers.png}.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from vcm.cl.data import mnist_compose
from vcm.cl.models import compose_mlp

ROOT = Path(__file__).resolve().parents[1]
READERS = ["concept", "layerwise", "first", "finetune", "scratch"]
LABELS = {"concept": "concept (whole digit net)", "layerwise": "layer-wise (in+hidden, frozen)",
          "first": "first layer only", "finetune": "fine-tune in+hidden", "scratch": "scratch"}
COLORS = {"concept": "#3a86ff", "layerwise": "#00798c", "first": "#8ecae6",
          "finetune": "#d1495b", "scratch": "#2e4057"}


def train(net: nn.Module, params, x, y, steps: int, seed: int, lr: float = 1e-3, bs: int = 128) -> nn.Module:
    """Adam on random batches (full passes over x, dropping a partial last batch)."""
    gen = torch.Generator().manual_seed(seed)
    params = [p for p in params if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    done = 0
    net.train()
    while done < steps:
        perm = torch.randperm(len(x), generator=gen)
        for i in range(0, max(len(x) - bs + 1, 1), bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(net(x[idx]), y[idx]).backward()
            opt.step()
            done += 1
            if done >= steps:
                break
    return net.eval()


def accuracy(net: nn.Module, x, y) -> float:
    with torch.no_grad():
        return float(torch.cat([net(x[i:i + 2000]).argmax(1) for i in range(0, len(x), 2000)]).eq(y).float().mean())


def frozen(m: nn.Module) -> nn.Module:
    return m.requires_grad_(False)


def make(kind: str, arch, digit: nn.Sequential, n_out: int) -> nn.Module:
    rest = lambda roles, d: nn.Sequential(*arch.build(roles, n_out, d_in=d, arity=2)[0])
    if kind == "concept":
        return nn.Sequential(frozen(digit), rest(("mix", "out"), 10))
    if kind == "layerwise":
        return nn.Sequential(frozen(digit[:2]), rest(("mix", "out"), 256))
    if kind == "first":
        return nn.Sequential(frozen(digit[:1]), rest(("hidden", "mix", "out"), 256))
    if kind == "finetune":
        return nn.Sequential(copy.deepcopy(digit[:2]).requires_grad_(True), rest(("mix", "out"), 256))
    return arch.network("binary", n_out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--sizes", nargs="+", type=int, default=[50, 100, 250, 1000, 4000, 16000])
    ap.add_argument("--steps", type=int, default=500, help="steps per two-image task (as --min-steps)")
    ap.add_argument("--digit-epochs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results" / "compose_readers"))
    o = ap.parse_args()
    torch.set_num_threads(o.threads)
    arch = compose_mlp()
    out = Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in o.seeds:
        bench = mnist_compose(seed=seed, train_per_task=max(o.sizes))
        t0 = bench.tasks[0]
        torch.manual_seed(seed)
        digit = arch.network("unary", 10)
        train(digit, digit.parameters(), t0.x_train, t0.y_train,
              o.digit_epochs * int(np.ceil(len(t0.x_train) / 128)), seed)
        digit_acc = accuracy(digit, t0.x_test, t0.y_test)
        print(f"seed {seed}: digit reader test accuracy {digit_acc:.4f}", flush=True)
        for t in bench.tasks[1:]:
            if t.template != "binary":
                continue
            for n in o.sizes:
                sub = torch.randperm(len(t.x_train), generator=torch.Generator().manual_seed(seed))[:n]
                for kind in READERS:
                    torch.manual_seed(1000 * seed + n)
                    net = make(kind, arch, digit, t.n_classes)
                    train(net, net.parameters(), t.x_train[sub], t.y_train[sub], o.steps, seed)
                    acc = accuracy(net, t.x_test, t.y_test)
                    rows.append({"seed": seed, "task": t.name, "n": n, "reader": kind, "acc": acc,
                                 "digit_acc": digit_acc})
                print(f"  {t.name:8s} n={n:6d} " + " ".join(
                    f"{r['reader']}={r['acc']:.3f}" for r in rows[-len(READERS):]), flush=True)
    (out / "readers.json").write_text(json.dumps({"args": vars(o), "rows": rows}))
    report(rows, o, out)


def report(rows, o, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = list(dict.fromkeys(r["task"] for r in rows))
    get = lambda task, n, kind: [r["acc"] for r in rows if r["n"] == n and r["reader"] == kind
                                 and (task is None or r["task"] == task)]
    per_seed_mean = lambda n, kind: [np.mean([r["acc"] for r in rows if r["n"] == n and r["reader"] == kind
                                              and r["seed"] == s]) for s in o.seeds]
    lines = ["# 强制读图器：晋升出的概念作为积木本身值多少\n",
             f"种子 {o.seeds}；每个两图任务训练 {o.steps} 步；测试集准确率，均值 ± 标准差（跨种子）。"
             f"读数字网络在测试集上的准确率：{np.mean([r['digit_acc'] for r in rows if r['n'] == o.sizes[0]]):.4f}。\n",
             "## 四个两图任务的平均\n",
             "| 训练样本数 | " + " | ".join(LABELS[k] for k in READERS) + " | 概念 − 逐层（配对差，跨种子） |",
             "|---" * (len(READERS) + 2) + "|"]
    for n in o.sizes:
        cells = [f"{np.mean(per_seed_mean(n, k)):.3f} ± {np.std(per_seed_mean(n, k)):.3f}" for k in READERS]
        diff = np.array(per_seed_mean(n, "concept")) - np.array(per_seed_mean(n, "layerwise"))
        wins = int((diff > 0).sum())
        lines.append(f"| {n} | " + " | ".join(cells) + f" | {diff.mean():+.3f} ± {diff.std():.3f}（{wins}/{len(diff)} 个种子为正） |")
    for task in tasks:
        lines += [f"\n## {task}\n", "| 训练样本数 | " + " | ".join(READERS) + " |", "|---" * (len(READERS) + 1) + "|"]
        for n in o.sizes:
            lines.append(f"| {n} | " + " | ".join(f"{np.mean(get(task, n, k)):.3f} ± {np.std(get(task, n, k)):.3f}"
                                                   for k in READERS) + " |")
    lines.append("\n![readers](readers.png)\n")
    (out / "readers.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, len(tasks) + 1, figsize=(4 * (len(tasks) + 1), 3.6), sharex=True)
    for ax, task in zip(axes, [None] + tasks):
        for kind in READERS:
            vals = [per_seed_mean(n, kind) if task is None else get(task, n, kind) for n in o.sizes]
            ax.errorbar(o.sizes, [np.mean(v) for v in vals], yerr=[np.std(v) for v in vals],
                        marker="o", ms=3, lw=1.6, capsize=2, color=COLORS[kind], label=LABELS[kind])
        ax.set_xscale("log")
        ax.set_title("mean of two-image tasks" if task is None else task)
        ax.set_xlabel("training samples")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test accuracy")
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Forced readers on compositional MNIST (reader trained on digits first)")
    fig.tight_layout()
    fig.savefig(out / "readers.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
