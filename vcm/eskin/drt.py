"""Reference implementation of the DRT test method (docs/eskin_drt.md).

DRT (decoder retention test) asks how much of a fixed reference decoder's accuracy a
tactile array keeps as it ages, with and without recalibration on the reference presses.
The decoder, the recalibration and the splits are fixed by the method, so the numbers
compare specimens (materials), not algorithms. Everything is closed-form or seeded.

Session file (one per recording): `<name>.npz` plus `<name>.json` with the metadata.
Section numbers in comments refer to docs/eskin_drt.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPEC_VERSION = "0.1"

H = W = 4                                   # 6.1 standard specimen: 4 x 4 taxels
N_POS = H * W
FORCE_LEVELS_N = (0.1, 0.3, 1.0, 3.0)       # 5.3 force levels, index = force_level
N_SHAPES = 6                                # 5.2 shape indenters S1..S6
BLOCK = {"ref": 0, "pos": 1, "force": 2, "shape": 3, "warmup": 8, "other": 9}
TASKS = {"pos": ("pos", N_POS), "force": ("force_level", len(FORCE_LEVELS_N)),
         "shape": ("shape", N_SHAPES)}      # task -> (label field, number of classes)

# Appendix A (normative): reference press order, (position, force_level), 6 rounds of 16.
_REF_ROUNDS = [
    ([11, 12, 6, 1, 2, 0, 9, 14, 8, 13, 3, 5, 15, 7, 4, 10], 2),
    ([10, 9, 7, 3, 15, 12, 11, 5, 6, 14, 0, 13, 8, 4, 1, 2], 3),
    ([2, 0, 9, 12, 13, 11, 14, 3, 4, 8, 6, 1, 15, 5, 7, 10], 1),
    ([14, 15, 12, 7, 10, 0, 3, 11, 4, 1, 6, 8, 9, 5, 2, 13], 2),
    ([1, 14, 3, 10, 4, 15, 12, 2, 13, 7, 0, 11, 9, 5, 8, 6], 3),
    ([7, 3, 13, 14, 1, 12, 5, 0, 8, 9, 10, 2, 15, 11, 6, 4], 1),
]
REF_ORDER = np.array([(p, lvl) for ps, lvl in _REF_ROUNDS for p in ps], dtype=np.int64)
DRIFT_LEVEL = 2                             # 9.6 signal drift is read at 1 N

RECAL_N = (0, 16, 32, 48, 96)               # 9.3 recalibration budgets (prefixes of REF_ORDER)
SPLIT_SEEDS = (0, 1, 2, 3, 4)               # 9.1 train / test splits
SHRINK = 0.1                                # 9.4 LDA shrinkage
RIDGE = 1.0                                 # 9.3 recalibration ridge towards identity
RETAIN = 0.95                               # 9.5 threshold for the recovery cost
MIN_DECODABLE = 0.5                         # 9.5 day-0 chance-normalised accuracy needed

ARRAYS = {"x": np.floating, "block": np.integer, "pos": np.integer, "force_level": np.integer,
          "shape": np.integer, "force": np.floating, "t0": np.floating, "baseline": np.floating}
META = ("spec_version", "specimen_id", "material", "batch", "fab_date", "session_date", "day",
        "temp_C", "RH", "freqs_Hz", "sample_rate_Hz", "timeline_s", "readout", "stage")


@dataclass
class Session:
    path: Path
    meta: dict
    a: dict[str, np.ndarray]

    @property
    def day(self) -> float:
        return float(self.meta["day"])

    def rows(self, block: str) -> np.ndarray:
        return np.flatnonzero(self.a["block"] == BLOCK[block])


def load_session(path: str | Path) -> Session:
    path = Path(path).with_suffix(".npz")
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return Session(path, meta, a)


def load_specimen(folder: str | Path) -> list[Session]:
    """All sessions of one specimen, ordered by `day`; the first one is S0."""
    return sorted((load_session(p) for p in Path(folder).glob("*.npz")), key=lambda s: s.day)


def windows(meta: dict) -> dict[str, tuple[int, int]]:
    """7.2 press timeline -> sample index ranges of the pre / ramp / hold / release / post windows."""
    edges = np.rint(np.cumsum([0.0, *meta["timeline_s"]]) * meta["sample_rate_Hz"]).astype(int)
    names = ("pre", "ramp", "hold", "release", "post")
    return {n: (int(edges[i]), int(edges[i + 1])) for i, n in enumerate(names)}


def validate(s: Session) -> list[str]:
    """Clause 8 conformance; returns a list of problems (empty = conforming)."""
    missing = [f"missing metadata: {k}" for k in META if k not in s.meta]
    errs = [f"missing array: {k}" for k in ARRAYS if k not in s.a]
    errs += [f"{k} should be {t.__name__}" for k, t in ARRAYS.items()
             if k in s.a and not np.issubdtype(s.a[k].dtype, t)]
    if errs or any(k not in s.meta for k in ("spec_version", "freqs_Hz", "sample_rate_Hz", "timeline_s")):
        return missing + errs
    x, n = s.a["x"], len(s.a["x"])
    if x.ndim != 5 or x.shape[2:4] != (H, W):
        return missing + [f"x should be [N, F, {H}, {W}, T], got {list(x.shape)}"]
    if s.meta["spec_version"] != SPEC_VERSION:
        errs.append(f"spec_version {s.meta['spec_version']!r}, this checker is {SPEC_VERSION}")
    if len(s.meta["freqs_Hz"]) != x.shape[1]:
        errs.append("freqs_Hz must list one frequency per channel of x")
    if len(s.meta["timeline_s"]) != 5:
        errs.append("timeline_s must give pre, ramp, hold, release, post (s)")
    elif windows(s.meta)["post"][1] != x.shape[-1]:
        errs.append(f"timeline_s x sample_rate_Hz gives {windows(s.meta)['post'][1]} samples, x has {x.shape[-1]}")
    for k in ("block", "pos", "force_level", "shape", "t0"):
        if s.a[k].shape != (n,):
            errs.append(f"{k} should have shape [{n}]")
    if s.a["force"].shape != (n, x.shape[-1]):
        errs.append(f"force should have shape [{n}, {x.shape[-1]}]")
    if s.a["baseline"].shape[:3] != x.shape[1:4]:
        errs.append("baseline should be [F, H, W, T0]")
    if errs:
        return missing + errs
    if not np.isin(s.a["block"], list(BLOCK.values())).all():
        errs.append("unknown block code")
    if np.any(np.diff(s.a["t0"]) <= 0):
        errs.append("presses must be stored in time order (t0 increasing)")
    ref = s.rows("ref")
    got = np.stack([s.a["pos"][ref], s.a["force_level"][ref]], 1)
    if len(ref) != len(REF_ORDER) or not (got == REF_ORDER).all():
        errs.append("reference presses must follow Appendix A exactly (96 presses, same order)")
    elif ref[-1] > min(s.rows(t).min(initial=n) for t in TASKS):
        errs.append("the reference block must come before the task presses (7.3)")
    for task, (field, k) in TASKS.items():
        lab = s.a[field][s.rows(task)]
        if len(lab) == 0:
            errs.append(f"no presses in task block {task!r}")
        elif lab.min() < 0 or lab.max() >= k:
            errs.append(f"{field} labels in block {task!r} must lie in 0..{k - 1}")
        elif np.bincount(lab, minlength=k).min() < 4:
            errs.append(f"block {task!r} needs at least 4 presses of every class of {field}")
    pre = windows(s.meta)["pre"]
    if (x[..., pre[0]:pre[1]].mean(-1) <= 0).any():
        errs.append("x must be a positive quantity (capacitance or |Z|), see 8.1")
    return missing + errs


def features(s: Session) -> np.ndarray:
    """9.2: relative change against each press's own pre-contact level, averaged over the
    ramp, hold and post windows -> [N, F * H * W * 3]."""
    x = s.a["x"].astype(np.float64)
    w = windows(s.meta)
    base = x[..., slice(*w["pre"])].mean(-1, keepdims=True)
    r = (x - base) / base
    f = np.stack([r[..., slice(*w[k])].mean(-1) for k in ("ramp", "hold", "post")], -1)
    return f.reshape(len(x), -1)


def split(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """9.1: stratified half / half split of positions into `labels` (train, test)."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        tr.append(idx[: len(idx) // 2])
        te.append(idx[len(idx) // 2:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(te))


class Standardizer:
    def __init__(self, X: np.ndarray):
        sd = X.std(0)
        self.mu, self.sd = X.mean(0), np.where(sd > 0, sd, 1.0)    # a dead channel stays at 0

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd


class ShrinkageLDA:
    """9.4: linear discriminant with shrinkage towards a scaled identity; equal priors."""

    def __init__(self, X: np.ndarray, y: np.ndarray, n_classes: int, shrink: float = SHRINK):
        mu = np.stack([X[y == c].mean(0) for c in range(n_classes)])
        Xc = X - mu[y]
        S = Xc.T @ Xc / max(len(X) - n_classes, 1)
        S = (1 - shrink) * S + shrink * np.trace(S) / len(S) * np.eye(len(S))
        self.W = np.linalg.solve(S, mu.T)
        self.c = -0.5 * np.sum(mu.T * self.W, 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(X @ self.W + self.c, 1)


def recal_map(u: np.ndarray, v: np.ndarray, ridge: float = RIDGE) -> tuple[np.ndarray, np.ndarray]:
    """9.3: per feature, (g, b) minimising sum (g u + b - v)^2 + ridge ((g - 1)^2 + b^2).
    With no presses this is the identity."""
    n = len(u)
    a11, a12, a22 = (u * u).sum(0) + ridge, u.sum(0), n + ridge
    r1, r2 = (u * v).sum(0) + ridge, v.sum(0)
    det = a11 * a22 - a12 ** 2
    return (r1 * a22 - a12 * r2) / det, (a11 * r2 - a12 * r1) / det


def ref_targets(s: Session, F: np.ndarray) -> np.ndarray:
    """Mean feature of each (position, force_level) over its repeats in the reference block,
    looked up for every entry of REF_ORDER -> [96, d]."""
    ref = s.rows("ref")
    keys = REF_ORDER[:, 0] * len(FORCE_LEVELS_N) + REF_ORDER[:, 1]
    mean = {k: F[ref[keys == k]].mean(0) for k in np.unique(keys)}
    return np.stack([mean[k] for k in keys])


def chance_norm(acc: float, k: int) -> float:
    return (acc - 1 / k) / (1 - 1 / k)


def _decode(src: Session, Fs: np.ndarray, dst: Session, Fd: np.ndarray, seed: int) -> dict:
    """Decoder trained on `src` (train split), tested on `dst` (test split) for every task and
    recalibration budget; recalibration maps dst's reference presses onto src's."""
    out = {}
    tgt = ref_targets(src, Fs)
    dref = Fd[dst.rows("ref")]
    for task, (field, k) in TASKS.items():
        rs, rd = src.rows(task), dst.rows(task)
        tr, _ = split(src.a[field][rs], seed)
        _, te = split(dst.a[field][rd], seed)
        z = Standardizer(Fs[rs[tr]])
        lda = ShrinkageLDA(z(Fs[rs[tr]]), src.a[field][rs[tr]], k)
        Zd, yd = z(Fd[rd[te]]), dst.a[field][rd[te]]
        out[task] = {}
        for n in RECAL_N:
            g, b = recal_map(z(dref[:n]), z(tgt[:n]))
            out[task][n] = chance_norm(float(np.mean(lda.predict(g * Zd + b) == yd)), k)
    return out


def signal_drift(s0: Session, F0: np.ndarray, s: Session, Fs: np.ndarray) -> float:
    """9.6: median relative change of the hold-window response of each taxel to a 1 N press
    on itself, all frequencies (the classical signal-space view, for comparison)."""
    def own(sess, F):
        f = ref_targets(sess, F).reshape(len(REF_ORDER), -1, H * W, 3)[..., 1]
        sel = REF_ORDER[:, 1] == DRIFT_LEVEL
        return f[sel, :, REF_ORDER[sel, 0]]            # [presses, F]
    a, b = own(s0, F0), own(s, Fs)
    return float(np.median(np.abs(b - a) / np.abs(a)))


def evaluate_specimen(sessions: list[Session]) -> dict:
    """Clause 9 for one specimen: S0 decodability, retention R(s|n), recovery cost n*(s),
    signal drift D(s). Means over SPLIT_SEEDS."""
    s0, F0 = sessions[0], features(sessions[0])
    Fs = [features(s) for s in sessions]
    per = [[_decode(s0, F0, s, F, seed) for s, F in zip(sessions, Fs)] for seed in SPLIT_SEEDS]
    mean = lambda t, i, n: float(np.mean([p[i][t][n] for p in per]))
    a0 = {t: mean(t, 0, 0) for t in TASKS}
    rep = {"specimen_id": s0.meta["specimen_id"], "material": s0.meta["material"],
           "a0": a0, "decodable": all(v >= MIN_DECODABLE for v in a0.values()), "sessions": []}
    for i, s in enumerate(sessions[1:], 1):
        R = {n: {t: mean(t, i, n) / a0[t] for t in TASKS} for n in RECAL_N}
        Rbar = {n: float(np.mean(list(R[n].values()))) for n in RECAL_N}
        # n*: the smallest budget from which every larger budget also reaches RETAIN
        ok = [n for i_n, n in enumerate(RECAL_N) if all(Rbar[m] >= RETAIN for m in RECAL_N[i_n:])]
        rep["sessions"].append({
            "day": s.day, "temp_C": s.meta["temp_C"], "RH": s.meta["RH"],
            "R_task": R, "R": Rbar, "n_star": ok[0] if ok else None,
            "D": signal_drift(s0, F0, s, Fs[i])})
    return rep


def evaluate_cross(specimens: list[list[Session]]) -> dict:
    """9.7 (optional): decoder from specimen A's S0, applied to specimen B's S0 after
    recalibration on B's reference presses; R averaged over ordered pairs and seeds."""
    S0 = [sp[0] for sp in specimens]
    F0 = [features(s) for s in S0]
    R = {n: [] for n in RECAL_N}
    for i, a in enumerate(S0):
        for j, b in enumerate(S0):
            if i == j:
                continue
            for seed in SPLIT_SEEDS:
                own, cross = _decode(a, F0[i], a, F0[i], seed), _decode(a, F0[i], b, F0[j], seed)
                for n in RECAL_N:
                    R[n].append(np.mean([cross[t][n] / own[t][0] for t in TASKS]))
    return {"pairs": len(S0) * (len(S0) - 1), "R": {n: float(np.mean(v)) for n, v in R.items()}}


def report_md(reps: list[dict], cross: dict | None = None) -> str:
    """Clause 10 summary table."""
    lines = [f"# DRT {SPEC_VERSION} 结果", "",
             "| 试样 | 材料 | 天 | 温度 ℃ | RH % | ã₀ (pos / force / shape) | R(s\\|0) | R(s\\|16) | R(s\\|96) | n* | D |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in reps:
        a0 = " / ".join(f"{r['a0'][t]:.2f}" for t in TASKS)
        if not r["decodable"]:
            lines.append(f"| {r['specimen_id']} | {r['material']} | 0 | | | {a0} | 不可解码，不报告 R | | | | |")
            continue
        for s in r["sessions"]:
            ns = s["n_star"] if s["n_star"] is not None else f"> {RECAL_N[-1]}"
            lines.append(f"| {r['specimen_id']} | {r['material']} | {s['day']:g} | {s['temp_C']:g} | {s['RH']:g} | {a0} | "
                         f"{s['R'][0]:.3f} | {s['R'][16]:.3f} | {s['R'][96]:.3f} | {ns} | {s['D']:.3f} |")
    if cross:
        lines += ["", f"试样间迁移（{cross['pairs']} 个有序对）：" +
                  "，".join(f"R(n={n}) = {v:.3f}" for n, v in cross["R"].items())]
    return "\n".join(lines) + "\n"
