"""Pick the hyperparameter value with the best validation ACC from sweep directories.

    python experiments/select_hparam.py results/raw/sweeps/split_cifar10_ewc* --key ewc_lambda
Prints the best value (for shell use) and writes a markdown table next to the sweeps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--key", required=True)
    ap.add_argument("--md", default=None)
    o = ap.parse_args()
    rows = []
    for d in o.dirs:
        for f in Path(d).glob("*.json"):
            r = json.loads(f.read_text())
            rows.append((r["args"][o.key], r["metrics"]["acc"], r["metrics"]["learning_acc"],
                         r["metrics"].get("bwt", 0.0), r["method"], r["benchmark"]))
    rows.sort(key=lambda r: r[0])
    best = max(rows, key=lambda r: r[1])
    if o.md:
        lines = [f"| {rows[0][5]} / {rows[0][4]} | {o.key} | ACC (val) | LA (val) | BWT (val) |",
                 "|---|---|---|---|---|"]
        lines += [f"| | {v:g} | {a:.4f} | {la:.4f} | {b:+.4f} |" for v, a, la, b, _, _ in rows]
        lines.append(f"\n选定 {o.key} = {best[0]:g}\n")
        with open(o.md, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    print(f"{best[0]:g}")


if __name__ == "__main__":
    main()
