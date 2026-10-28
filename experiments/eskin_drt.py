"""DRT (decoder retention test, docs/eskin_drt.md): check files, evaluate specimens, make demo data.

    python experiments/eskin_drt.py validate data/eskin/A1/*.npz
    python experiments/eskin_drt.py evaluate data/eskin/A1 data/eskin/A2 data/eskin/A3 --cross --out results/eskin/A
    python experiments/eskin_drt.py synth --out data/eskin_synth      # synthetic demo (not material data)

`evaluate` takes one folder per specimen (all sessions of that specimen inside; S0 = smallest `day`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vcm.eskin import drt, synth


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("files", nargs="+")
    e = sub.add_parser("evaluate")
    e.add_argument("specimens", nargs="+")
    e.add_argument("--cross", action="store_true", help="also 9.7 specimen-to-specimen transfer")
    e.add_argument("--out", default="results/eskin/drt")
    s = sub.add_parser("synth")
    s.add_argument("--out", default="data/eskin_synth")
    args = ap.parse_args()

    if args.cmd == "validate":
        bad = 0
        for f in args.files:
            errs = drt.validate(drt.load_session(f))
            bad += bool(errs)
            print(f"{f}: " + ("OK" if not errs else "\n  " + "\n  ".join(errs)))
        return int(bad > 0)

    if args.cmd == "synth":
        for k, mode in enumerate(("none", "gain", "nonlinear")):
            synth.make_specimen(Path(args.out) / f"syn-{mode}", f"syn-{mode}", days=(0, 7, 30),
                                drift=mode, seed=k, material=f"synthetic ({mode})")
        print(f"wrote 3 synthetic specimens x 3 sessions to {args.out}")
        return 0

    specimens = [drt.load_specimen(p) for p in args.specimens]
    for sp in specimens:
        for sess in sp:
            if errs := drt.validate(sess):
                print(f"{sess.path} does not conform:\n  " + "\n  ".join(errs), file=sys.stderr)
                return 1
    reps = [drt.evaluate_specimen(sp) for sp in specimens]
    cross = drt.evaluate_cross(specimens) if args.cross and len(specimens) > 1 else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "drt.json").write_text(json.dumps({"spec_version": drt.SPEC_VERSION, "specimens": reps,
                                              "cross": cross}, ensure_ascii=False, indent=1))
    (out / "drt.md").write_text(drt.report_md(reps, cross), encoding="utf-8")
    print(drt.report_md(reps, cross))
    return 0


if __name__ == "__main__":
    sys.exit(main())
