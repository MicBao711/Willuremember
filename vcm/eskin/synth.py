"""Synthetic DRT sessions: a 4 x 4 capacitive array under the clause-7 protocol.

Only for testing the reference implementation and the file format; the numbers it gives
say nothing about real materials. Response of taxel c to the mean pressure P_c on its
electrode: r = g_c * log(1 + P_c / P0_c) with creep during hold, a decaying residual after
release and nearest-neighbour crosstalk; raw value x = C0_c * (1 + r) + noise.

Drift modes (as a function of day d, f = d / 30):
  none       nothing changes between sessions
  gain       per-taxel sensitivity loss and baseline drift (a per-feature affine change)
  nonlinear  gain loss plus shifted saturation (P0) and growing crosstalk (not affine)
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .drt import BLOCK, FORCE_LEVELS_N, H, N_POS, N_SHAPES, REF_ORDER, SPEC_VERSION, W

PITCH, ELECTRODE, GRID = 5.0, 3.0, 0.25          # mm (6.1 standard specimen)
TIMELINE = (0.2, 0.3, 1.0, 0.3, 0.7)              # s (7.2)
FS = 50.0                                         # Hz
_n = int(round(H * PITCH / GRID))
_xy = (np.arange(_n) + 0.5) * GRID
GX, GY = np.meshgrid(_xy, _xy, indexing="xy")
CENTERS = np.array([(PITCH * (j + 0.5), PITCH * (i + 0.5)) for i in range(H) for j in range(W)])
ELECTRODES = np.stack([(np.abs(GX - cx) <= ELECTRODE / 2) & (np.abs(GY - cy) <= ELECTRODE / 2)
                       for cx, cy in CENTERS])


def indenter(kind: str, cx: float, cy: float) -> np.ndarray:
    """5.2 indenters: 'probe' (round, 3 mm) and shapes S1..S6."""
    dx, dy = GX - cx, GY - cy
    r = np.hypot(dx, dy)
    if kind == "probe":
        return r <= 1.5
    if kind == "S1":                                   # round, 10 mm
        return r <= 5.0
    if kind == "S2":                                   # bar 2 x 16 mm, horizontal
        return (np.abs(dx) <= 8) & (np.abs(dy) <= 1)
    if kind == "S3":                                   # the same bar, vertical
        return (np.abs(dx) <= 1) & (np.abs(dy) <= 8)
    if kind == "S4":                                   # ring, 14 / 10 mm
        return (r <= 7) & (r >= 5)
    if kind == "S5":                                   # cross, 14 mm, arms 3 mm wide
        return ((np.abs(dx) <= 7) & (np.abs(dy) <= 1.5)) | ((np.abs(dx) <= 1.5) & (np.abs(dy) <= 7))
    if kind == "S6":                                   # equilateral triangle, side 14 mm
        h = 14 * np.sqrt(3) / 2
        y = dy + h / 3
        return (y >= 0) & (y <= h) & (np.abs(dx) <= (h - y) / np.sqrt(3))
    raise ValueError(kind)


def taxel_pressure(mask: np.ndarray, force_n: float) -> np.ndarray:
    """Mean contact pressure (kPa) over each electrode for a uniform load on `mask`."""
    p = force_n / (mask.sum() * GRID ** 2) * 1e3
    return p * (mask[None] & ELECTRODES).sum((1, 2)) / ELECTRODES.sum((1, 2))


def _profile(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load fraction, creep factor and post-release mask over the record."""
    e = np.cumsum([0.0, *TIMELINE])
    load = np.interp(t, [e[1], e[2], e[3], e[4]], [0, 1, 1, 0], left=0, right=0)
    creep = 1 + 0.05 * (1 - np.exp(-np.clip(t - e[2], 0, None) / 0.3))
    return load, creep, t >= e[4]


def make_session(specimen: dict, day: float, drift: str, rng: np.random.Generator) -> tuple[dict, dict]:
    if drift not in ("none", "gain", "nonlinear"):
        raise ValueError(drift)
    f = day / 30
    g, c0 = specimen["gain"].copy(), specimen["c0"].copy()
    p0, kappa = np.full(N_POS, 20.0), 0.05
    if drift in ("gain", "nonlinear"):
        g *= 1 - 0.3 * f + 0.15 * f * specimen["g_noise"]
        c0 *= 1 - 0.2 * f
    if drift == "nonlinear":
        p0 *= 1 + 2.0 * f * specimen["p0_noise"]
        kappa += 0.3 * f

    # 7.3 press list: warm-up, reference block (Appendix A), then the tasks interleaved
    presses = [("warmup", "probe", int(rng.integers(N_POS)), 2, -1) for _ in range(20)]
    presses += [("ref", "probe", int(p), int(l), -1) for p, l in REF_ORDER]
    tasks = [("pos", "probe", p, l, -1) for p in range(N_POS) for l in range(4) for _ in range(4)]
    tasks += [("force", "probe", int(rng.integers(N_POS)), l, -1) for l in range(4) for _ in range(64)]
    tasks += [("shape", f"S{s + 1}", -1, int(rng.integers(2, 4)), s) for s in range(N_SHAPES) for _ in range(40)]
    presses += [tasks[i] for i in rng.permutation(len(tasks))]

    t = np.arange(int(round(sum(TIMELINE) * FS))) / FS
    load, creep, post = _profile(t)
    t_post = t - np.cumsum(TIMELINE)[3]
    nbr = [[k for k in (c - W, c + W, c - 1, c + 1) if 0 <= k < N_POS and (k // W == c // W or k % W == c % W)]
           for c in range(N_POS)]
    X, F = [], []
    for block, kind, pos, lvl, shape in presses:
        if kind == "probe":
            cx, cy = CENTERS[pos] + rng.normal(0, 0.2, 2)
        else:
            cx, cy = (H * PITCH / 2, H * PITCH / 2) + rng.uniform(-1, 1, 2)
        force = FORCE_LEVELS_N[lvl] * (1 + rng.normal(0, 0.02))
        pc = taxel_pressure(indenter(kind, cx, cy), force)
        r = g[:, None] * np.log1p(pc[:, None] * load[None] / p0[:, None]) * creep[None]
        hold_end = g * np.log1p(pc / p0) * creep[~post][-1]
        r[:, post] = 0.1 * hold_end[:, None] * np.exp(-t_post[post] / 0.2)[None]
        r = r + kappa * np.stack([r[k].mean(0) for k in nbr])
        x = c0[:, None] * (1 + r) + rng.normal(0, 0.003, r.shape) * c0[:, None]
        X.append(x.reshape(1, H, W, -1))
        F.append(force * load + rng.normal(0, 0.005, len(t)))
    n = len(presses)
    arrays = {
        "x": np.asarray(X, dtype=np.float32),
        "block": np.array([BLOCK[p[0]] for p in presses], dtype=np.int8),
        "pos": np.array([p[2] for p in presses], dtype=np.int16),
        "force_level": np.array([p[3] for p in presses], dtype=np.int8),
        "shape": np.array([p[4] for p in presses], dtype=np.int8),
        "force": np.asarray(F, dtype=np.float32),
        "t0": np.arange(n) * (sum(TIMELINE) + 1.0),
        "baseline": (c0[:, None] * (1 + rng.normal(0, 0.003, (N_POS, 50)))).reshape(1, H, W, 50).astype(np.float32),
    }
    fab = date(2026, 1, 5)
    meta = {
        "spec_version": SPEC_VERSION, "specimen_id": specimen["id"], "material": specimen["material"],
        "batch": "synthetic", "fab_date": fab.isoformat(),
        "session_date": (fab + timedelta(days=day)).isoformat(), "day": day,
        "temp_C": 23.0, "RH": 50.0, "freqs_Hz": [1000.0], "sample_rate_Hz": FS,
        "timeline_s": list(TIMELINE), "readout": "synthetic (vcm.eskin.synth)",
        "stage": {"probe": "round 3 mm", "shapes": ["S1", "S2", "S3", "S4", "S5", "S6"]},
        "notes": f"synthetic, drift={drift}",
    }
    return arrays, meta


def write_session(folder: str | Path, arrays: dict, meta: dict) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{meta['specimen_id']}_{meta['session_date'].replace('-', '')}_1"
    np.savez_compressed(folder / f"{stem}.npz", **arrays)
    (folder / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return folder / f"{stem}.npz"


def make_specimen(folder: str | Path, specimen_id: str, days=(0, 7, 30), drift: str = "gain",
                  seed: int = 0, material: str = "synthetic") -> list[Path]:
    rng = np.random.default_rng(seed)
    spec = {"id": specimen_id, "material": material,
            "gain": 1 + 0.1 * rng.normal(size=N_POS), "c0": 10 * (1 + 0.05 * rng.normal(size=N_POS)),
            "g_noise": rng.normal(size=N_POS), "p0_noise": rng.uniform(-0.5, 1.0, N_POS)}
    return [write_session(folder, *make_session(spec, d, drift, rng)) for d in days]
