"""Three-way comparison of muon energy loss: hep-simkernels (NumPy), Geant4 (G4EmCalculator, emcalc.cc) and PDG.

    python validation/g4emcalc/energies.py > energies.txt            # the PDG grid, 10 MeV-1 TeV
    emcalc FTFP_BERT_EMZ energies.txt emcalc.json G4_W G4_Au ...      # Geant4 (build: CMakeLists.txt)
    python validation/g4emcalc/compare.py emcalc.json --output record.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "estar_pdg"))
from compare import b_loss  # noqa: E402  (the radiative integral of the ESTAR/PDG comparison)

PROC = {"muIoni": "ionization", "muPairProd": "pair", "muBrems": "brems"}
KERNEL = {"muPairProd": "pair", "muBrems": "brems"}


def worst(T, r, lo, hi):
    m = (T >= lo) & (T <= hi) & np.isfinite(r)
    i = np.flatnonzero(m)[np.argmax(np.abs(r[m]))]
    return {"max_abs_rel": float(abs(r[i])), "signed": float(r[i]), "at_T_MeV": float(T[i])}


def main(argv=None) -> int:
    from hepsimkernels import eloss, materials, muradiative
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("emcalc", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)
    crit_bytes = (HERE / "criteria.json").read_bytes()
    crit = json.loads(crit_bytes)
    g4 = json.loads(a.emcalc.read_text())
    pdg = json.loads((HERE.parent / "tables" / "pdg_muon.json").read_text())["materials"]
    rows = g4["rows"]
    rec = {"criteria_sha256": hashlib.sha256(crit_bytes).hexdigest(), "geant4": g4["geant4"],
           "physics_list": g4["physics_list"], "materials": {}}
    print(f"{'material':18s} {'process':10s} {'kernel/G4 (mu-)':>22s} {'G4 mu-/PDG':>22s} {'kernel/PDG':>22s}  class")
    for name in crit["materials"]:
        mat = materials.get(name)
        q = pdg[name]
        Tq = np.array(q["T_MeV"])
        m = {}
        for proc, lim in crit["kernel_vs_geant4"].items():
            g = {pt: np.array([[r[3], r[4]] for r in rows if r[0] == name and r[1] == pt and r[2] == proc])
                 for pt in ("mu-", "mu+")}
            T = g["mu-"][:, 0]
            if proc == "muIoni":
                p = np.sqrt(T * (T + 2 * eloss.MMU))
                k = eloss.dedx_restricted(mat, p, 1e12) * 10.0 / mat.rho
            else:
                k = np.full(T.shape, np.nan)
                sel = T >= 1e3
                k[sel] = b_loss(muradiative, mat, T[sel], "nuclear" if proc == "muonNuclear" else KERNEL[proc])
            tab = np.interp(np.log(T), np.log(Tq), np.array(q[PROC[proc]]))
            lo, hi = lim["range_MeV"]
            kg = worst(T, k / g["mu-"][:, 1] - 1, lo, hi)
            kgp = worst(T, k / g["mu+"][:, 1] - 1, lo, hi)
            gp = worst(T, g["mu-"][:, 1] / tab - 1, lo, hi)
            kp = worst(T, k / tab - 1, lo, hi)
            kernel_ok = kg["max_abs_rel"] <= lim["max_rel"]
            pdg_ok = kp["max_abs_rel"] <= lim["max_rel"]
            cls = ("agrees with both" if kernel_ok and pdg_ok else "Geant4-vs-PDG difference" if kernel_ok
                   else "kernel departs from Geant4")
            m[proc] = {"kernel_vs_g4_mu-": kg, "kernel_vs_g4_mu+": kgp, "g4_vs_pdg": gp, "kernel_vs_pdg": kp,
                       "kernel_vs_g4_pass": bool(kernel_ok), "class": cls,
                       "profile": {"T_MeV": T.tolist(), "kernel": k.tolist(), "g4_mu-": g["mu-"][:, 1].tolist(),
                                   "g4_mu+": g["mu+"][:, 1].tolist(), "pdg": tab.tolist()}}
            f = lambda w: f"{100 * w['signed']:+.2f} % ({w['at_T_MeV']:.3g})"
            print(f"{name:18s} {proc:10s} {f(kg):>22s} {f(gp):>22s} {f(kp):>22s}  {cls}")
        rec["materials"][name] = m
    a.output.write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
