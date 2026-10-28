"""Run continual-learning benchmarks.

    python experiments/run_cl.py --benchmark pmnist --tasks 10 --methods finetune ewc vcm_lazy --seeds 0 1 2
    python experiments/run_cl.py --benchmark mnist_compose --train-per-task 1000 --methods vcm_freeze vcm_freeze_promo_comp

Each (benchmark, method, seed) is written to results/raw/<benchmark>/<method>_s<seed>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from vcm.cl.data import make_benchmark
from vcm.cl.learners import (EWC, MNTDP, FineTune, GraceCL, Independent, Replay, TrainConfig,
                             VCMConfig, VCMLearner)
from vcm.cl.models import make_architecture
from vcm.cl.runner import run

ROOT = Path(__file__).resolve().parents[1]


def vcm(o, **cfg):
    cfg = {"feedback_per_task": o.buffer, "shared_lr_scale": o.shared_lr_scale,
           "tolerance": o.tolerance, "judge": o.judge, "drift_threshold": o.drift_threshold,
           "judge_log": o.judge_log, "feedback_source": o.feedback_source,
           "repair_steps": o.repair_steps, "probe_steps": o.probe_steps, "probe_val": o.probe_val,
           "selection": o.selection, "halving_eta": o.halving_eta, "finalists": o.finalists,
           "curve_every": o.curve_every, **cfg}
    return lambda *a, **kw: VCMLearner(*a, config=VCMConfig(**cfg), **kw)


METHODS = {
    "finetune": lambda o: FineTune,
    "ewc": lambda o: (lambda *a, **kw: EWC(*a, lam=o.ewc_lambda, **kw)),
    "er": lambda o: (lambda *a, **kw: Replay(*a, per_task=o.buffer, **kw)),
    "independent": lambda o: Independent,
    "grace": lambda o: (lambda *a, **kw: GraceCL(*a, eps_scale=o.grace_eps, max_keys=o.grace_keys, **kw)),
    "mntdp": lambda o: MNTDP,   # MNTDP-D (Veniat et al., ICLR 2021), see vcm/cl/learners/mntdp.py
    # VCM variants
    "vcm_freeze": lambda o: vcm(o, share_mode="freeze"),
    "vcm_lazy": lambda o: vcm(o, share_mode="update", maintenance="lazy"),
    "vcm_eager": lambda o: vcm(o, share_mode="update", maintenance="eager"),
    "vcm_lazy_norepair": lambda o: vcm(o, share_mode="update", maintenance="lazy", repair_steps=0),
    # control: shared units updated in place, every stale composition accepted unchecked
    "vcm_nojudge": lambda o: vcm(o, share_mode="update", maintenance="lazy", tolerance=1.0),
    "vcm_lazy_promo": lambda o: vcm(o, share_mode="update", maintenance="lazy", promotion=True,
                                    promotion_mode="cut_deps"),
    "vcm_lazy_promo_keep": lambda o: vcm(o, share_mode="update", maintenance="lazy", promotion=True,
                                         promotion_mode="keep_deps"),
    "vcm_freeze_promo": lambda o: vcm(o, share_mode="freeze", promotion=True, promotion_mode="cut_deps"),
    # promotion of whole task compositions (cache entries), which can fill reader slots of templates
    "vcm_freeze_promo_comp": lambda o: vcm(o, share_mode="freeze", promote_compositions=True),
    # control: every composition is promoted at once (no frequency / stability requirement)
    "vcm_freeze_promo_comp0": lambda o: vcm(o, share_mode="freeze", promote_compositions=True,
                                            promote_min_hits=0, promote_min_stable=0),
    "vcm_lazy_promo_comp": lambda o: vcm(o, share_mode="update", maintenance="lazy", promote_compositions=True),
    "vcm_lazy_promo_comp_keep": lambda o: vcm(o, share_mode="update", maintenance="lazy",
                                              promote_compositions=True, promotion_mode="keep_deps"),
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="pmnist",
                    choices=["pmnist", "split_cifar10", "split_cifar100", "mnist_compose"])
    ap.add_argument("--tasks", type=int, default=None)
    ap.add_argument("--train-per-task", type=int, default=None)
    ap.add_argument("--keep-first-full", action="store_true",
                    help="limit training data only for tasks after the first (forward-transfer regime)")
    ap.add_argument("--eval-split", default="test", choices=["test", "val"])
    ap.add_argument("--val-fraction", type=float, default=None,
                    help="share of each task's training pool held out as validation (default: the benchmark's)")
    ap.add_argument("--methods", nargs="+", default=["finetune", "ewc", "er", "grace", "independent", "vcm_lazy"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--min-steps", type=int, default=0,
                    help="train every task for at least this many steps (small training sets)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--buffer", type=int, default=200, help="replay / feedback samples per task")
    ap.add_argument("--ewc-lambda", type=float, default=100.0)
    ap.add_argument("--grace-eps", type=float, default=1.0)
    ap.add_argument("--grace-keys", type=int, default=2000)
    ap.add_argument("--shared-lr-scale", type=float, default=0.1,
                    help="VCM update mode: learning-rate factor for reused units trained in place")
    ap.add_argument("--tolerance", type=float, default=0.02, help="VCM judge: allowed accuracy drop")
    ap.add_argument("--judge", default="function", choices=["function", "weight_rel", "weight_neuron"],
                    help="VCM: what decides whether an old task takes updated shared units "
                         "(accuracy on its feedback buffer, or DEN-style weight drift without data)")
    ap.add_argument("--drift-threshold", type=float, default=0.1,
                    help="VCM weight judges: largest drift at which the new version is accepted")
    ap.add_argument("--judge-log", action="store_true",
                    help="VCM diagnostics: log every judge call with all statistics (not counted)")
    ap.add_argument("--settle", action="store_true",
                    help="after the last measurement, run all pending lazy checks and record what is retained")
    ap.add_argument("--feedback-source", default="val", choices=["train", "val"],
                    help="VCM judge buffer: held-out validation samples or training samples")
    ap.add_argument("--repair-steps", type=int, default=0,
                    help="VCM judge: steps spent retraining private units of a failing task (0: keep old snapshot)")
    ap.add_argument("--probe-steps", type=int, default=50,
                    help="VCM: training steps each candidate composition gets before the choice")
    ap.add_argument("--probe-val", type=int, default=1000,
                    help="VCM: validation samples the choice looks at (diagnostics score on the next as many)")
    ap.add_argument("--selection", default="probe", choices=["probe", "halving", "finalists", "full"],
                    help="VCM: how candidate compositions are chosen (see VCMLearner._learn_staged)")
    ap.add_argument("--halving-eta", type=int, default=2)
    ap.add_argument("--finalists", type=int, default=2,
                    help="VCM --selection finalists: candidates trained fully after the probe")
    ap.add_argument("--curve-every", type=int, default=0,
                    help="VCM diagnostics: log every candidate's validation curve at this step interval")
    ap.add_argument("--serve-batches", type=int, default=20)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(ROOT / "results" / "raw"))
    o = ap.parse_args(argv)
    if o.threads:
        torch.set_num_threads(o.threads)
    if o.eval_split == "val" and o.feedback_source == "val":
        # the judge must not see the samples we measure on
        print("note: --eval-split val, so the VCM judge uses training samples", flush=True)
        o.feedback_source = "train"
    epochs = o.epochs or (1 if o.benchmark == "pmnist" else 5)
    out_dir = Path(o.out) / (o.benchmark + (f"_{o.tag}" if o.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in o.seeds:
        kw = {"seed": seed, "train_per_task": o.train_per_task, "keep_first_full": o.keep_first_full}
        if o.benchmark == "pmnist":
            kw["num_tasks"] = o.tasks or 10
        elif o.benchmark == "mnist_compose":  # the first task always trains on its whole pool
            kw = {"seed": seed, "train_per_task": o.train_per_task, "num_tasks": o.tasks}
        else:
            kw["num_tasks"] = o.tasks
        if o.val_fraction is not None:
            kw["val_fraction"] = o.val_fraction
        cfg = TrainConfig(epochs=epochs, batch_size=o.batch_size, lr=o.lr, min_steps=o.min_steps)
        for name in o.methods:
            # Fresh benchmark per method: the runner releases training sets once learned.
            bench = make_benchmark(o.benchmark, **kw)
            arch = make_architecture(bench.name, bench.n_outputs)
            print(f"== {bench.name} | {name} | seed {seed}", flush=True)
            learner = METHODS[name](o)(arch, bench.task_id_at_test, cfg, seed=seed)
            learner.name = name
            result = run(bench, learner, serve_batches=o.serve_batches, seed=seed,
                         log=lambda s: print(s, flush=True), eval_split=o.eval_split, settle=o.settle)
            result.update(method=name, seed=seed, benchmark=bench.name, args=vars(o), epochs=epochs)
            path = out_dir / f"{name}_s{seed}.json"
            path.write_text(json.dumps(result, default=float))
            m = result["metrics"]
            print(f"   ACC={m['acc']:.4f} LA={m['learning_acc']:.4f} BWT={m.get('bwt', 0):.4f} "
                  f"compute={result['compute']['total'] / 1e9:.1f}G time={result['time']:.0f}s -> {path}",
                  flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
