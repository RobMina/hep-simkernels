"""Where the exit angle and displacement differ: quantiles of Geant4 (halves A and B) and eslab for some cases.

    python validation/g4slab/diag_quantiles.py RUN_DIR --cases a,b --events 200000 --output diag.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze import numpy_arm, read_half

Q = [0.5, 0.9, 0.99, 0.999]


def summ(x):
    x = x[np.isfinite(x)]
    return {"n": int(x.size), "rms": float(np.sqrt(np.mean(x * x))), "q": [float(v) for v in np.quantile(x, Q)],
            "rms_err": float(np.sqrt(np.mean(x * x)) / np.sqrt(2 * max(x.size, 1)))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--events", type=int, default=200000)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)
    camp = json.loads((a.run / "campaign.json").read_text())
    out = {"quantiles": Q}
    for ci, case in enumerate(camp["cases"]):
        if case["case"] not in a.cases.split(","):
            continue
        A, B = read_half(a.run, case, "A"), read_half(a.run, case, "B")
        K = numpy_arm(case, a.events, A["tcut"], A["kcut"], 7919 + ci)
        row = {}
        for k in ("theta", "r"):
            row[k] = {arm: summ(s[k][s["trans"]]) for arm, s in (("g4_A", A), ("g4_B", B), ("kernel", K))}
            row[k]["g4_A_frac_zero"] = float((A[k][A["trans"]] == 0).mean())
        out[case["case"]] = row
        t = row["theta"]
        print(case["case"], "theta rms mrad: G4 A %.4f B %.4f kernel %.4f; q50/q90/q99 G4 A %s kernel %s" % (
            1e3 * t["g4_A"]["rms"], 1e3 * t["g4_B"]["rms"], 1e3 * t["kernel"]["rms"],
            [round(1e3 * v, 3) for v in t["g4_A"]["q"][:3]], [round(1e3 * v, 3) for v in t["kernel"]["q"][:3]]), flush=True)
        r = row["r"]
        print("   r rms um: G4 A %.3f B %.3f kernel %.3f; G4 r==0 %.3f" % (
            1e3 * r["g4_A"]["rms"], 1e3 * r["g4_B"]["rms"], 1e3 * r["kernel"]["rms"], r["g4_A_frac_zero"]), flush=True)
    a.output.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
