"""Compare the hep-simkernels stopping powers with NIST ESTAR (electrons) and the PDG muon tables, on public tables
only (validation/tables, written by fetch_tables.py).

    HEPSIMK_DEVICE_FLOAT=float64 python validation/estar_pdg/compare.py --out record_float64.json
    HEPSIMK_DEVICE_FLOAT=float32 python validation/estar_pdg/compare.py --out record_float32.json
    python validation/estar_pdg/compare.py --report record_float64.json record_float32.json   # Markdown tables

Electrons: eioni (G4MollerBhabhaModel) and ebrem (G4SeltzerBergerModel) unrestricted losses against ESTAR's collision
and radiative stopping powers. Muons: eloss (G4MuBetheBlochModel without corrections) and its JAX twin in the device
float against the PDG ionization column, and muradiative's brems, pair and photonuclear losses (b E) against the PDG
columns. Criteria: criteria.json, declared before the first comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "tables"


def rel(kernel, table):
    kernel, table = np.asarray(kernel, float), np.asarray(table, float)
    return np.where(table > 0, kernel / np.where(table > 0, table, 1.0) - 1.0, np.nan)


def window(T, r, lo, hi):
    m = (T >= lo) & (T <= hi) & np.isfinite(r)
    if not m.any():
        return None
    i = np.flatnonzero(m)[np.argmax(np.abs(r[m]))]
    return {"max_abs_rel": float(abs(r[i])), "at_T_MeV": float(T[i]), "signed": float(r[i]), "n": int(m.sum())}


def b_loss(R, mat, T, process):
    """Energy loss b E [MeV cm2/g] of one radiative process, by integration of v dSigma/dv as the unit tests do."""
    E = T + R.MMU
    lv = np.linspace(np.log(1e-7), np.log(1 - 1e-9), 4000)
    v = np.exp(lv)
    return np.array([np.trapezoid(v * v * R.dsigma_dv_per_mm(mat, e, v, processes=(process,)), lv) * 10.0 / mat.rho * e
                     for e in E])


def run(out: Path) -> int:
    from hepsimkernels import ebrem, eioni, eloss, eloss_jax, materials, muradiative, precision
    crit_bytes = (HERE / "criteria.json").read_bytes()
    crit = json.loads(crit_bytes)
    estar = json.loads((TABLES / "estar.json").read_text())["materials"]
    pdg = json.loads((TABLES / "pdg_muon.json").read_text())["materials"]
    dtype = precision.FLOAT.name
    rec = {"dtype": dtype, "criteria_sha256": hashlib.sha256(crit_bytes).hexdigest(),
           "tables_sha256": {k: hashlib.sha256((TABLES / k).read_bytes()).hexdigest()
                             for k in ("estar.json", "pdg_muon.json")}, "materials": {}, "checks": {}}
    ok = True
    for name in estar:
        mat = materials.get(name)
        m = {"rho": mat.rho, "I_eV": mat.I_eV}
        e = estar[name]
        T = np.array(e["T_MeV"])
        match_e = abs(e["density_g_cm3"] / mat.rho - 1) <= 0.01 and abs(e["I_eV"] / mat.I_eV - 1) <= 0.01
        coll = eioni.dedx_restricted(mat, T, 1e9) * 10.0 / mat.rho
        rad = ebrem.dedx_restricted(mat, T, 1e9) * 10.0 / mat.rho
        m["estar"] = {"material_match": bool(match_e),
                      "collision": window(T, rel(coll, e["collision_MeV_cm2_g"]), *crit["estar_collision"]["range"]),
                      "collision_below": window(T, rel(coll, e["collision_MeV_cm2_g"]), 0.0, 0.1),
                      "radiative": window(T, rel(rad, e["radiative_MeV_cm2_g"]), *crit["estar_radiative"]["range"]),
                      "radiative_below": window(T, rel(rad, e["radiative_MeV_cm2_g"]), 0.0, 1.0)}
        q = pdg[name]
        Tm = np.array(q["T_MeV"])
        sel = (Tm >= 10.0) & (Tm <= 1e6)
        Tm = Tm[sel]
        p = np.sqrt(Tm * (Tm + 2 * eloss.MMU))
        dens = q["density_g_cm3"] if q["density_g_cm3"] else mat.rho
        match_q = abs(dens / mat.rho - 1) <= 0.01 and abs(q["I_eV"] / mat.I_eV - 1) <= 0.01
        ion_np = eloss.dedx_restricted(mat, p, 1e12) * 10.0 / mat.rho
        ion_jx = np.asarray(eloss_jax.dedx_restricted(mat, p.astype(dtype), 1e12), float) * 10.0 / mat.rho
        jx = float(np.max(np.abs(ion_jx / ion_np - 1)))
        tab = {k: np.array(q[k])[sel] for k in ("ionization", "brems", "pair", "photonuc", "radloss")}
        rad_T = Tm[Tm >= 1e3]
        rk = {"brems": b_loss(muradiative, mat, rad_T, "brems"), "pair": b_loss(muradiative, mat, rad_T, "pair"),
              "photonuc": b_loss(muradiative, mat, rad_T, "nuclear")}
        rk["radloss"] = rk["brems"] + rk["pair"] + rk["photonuc"]
        hi = Tm >= 1e3
        m["pdg"] = {"material_match": bool(match_q), "table_density": q["density_g_cm3"], "table_I_eV": q["I_eV"],
                    "ionization_numpy": window(Tm, rel(ion_np, tab["ionization"]), *crit["pdg_ionization"]["range"]),
                    "ionization_device": window(Tm, rel(ion_jx, tab["ionization"]), *crit["pdg_ionization"]["range"]),
                    "ionization_10_100MeV": window(Tm, rel(ion_np, tab["ionization"]), 10.0, 100.0),
                    "ionization_100GeV_1TeV": window(Tm, rel(ion_np, tab["ionization"]), 1e5, 1e6),
                    "device_vs_numpy": jx}
        for k in ("brems", "pair", "photonuc", "radloss"):
            m["pdg"][k] = window(rad_T, rel(rk[k], tab[k][hi]), *crit[f"pdg_{k}"]["range"])
            m["pdg"][k + "_1_10GeV"] = window(rad_T, rel(rk[k], tab[k][hi]), 1e3, 1e4)
        checks = {}
        if match_e:
            checks["estar_collision"] = m["estar"]["collision"]["max_abs_rel"] <= crit["estar_collision"]["max_rel"]
        checks["estar_radiative"] = m["estar"]["radiative"]["max_abs_rel"] <= crit["estar_radiative"]["max_rel"]
        if match_q:
            checks["pdg_ionization"] = (m["pdg"]["ionization_numpy"]["max_abs_rel"] <= crit["pdg_ionization"]["max_rel"]
                                        and m["pdg"]["ionization_device"]["max_abs_rel"] <= crit["pdg_ionization"]["max_rel"])
        for k in ("brems", "pair", "photonuc", "radloss"):
            checks[f"pdg_{k}"] = m["pdg"][k]["max_abs_rel"] <= crit[f"pdg_{k}"]["max_rel"]
        checks["jax_vs_numpy"] = jx <= crit["jax_vs_numpy"][dtype]
        m["checks"] = {k: bool(v) for k, v in checks.items()}
        ok &= all(checks.values())
        rec["materials"][name] = m
        print(f"{name}: " + ", ".join(f"{k} {'pass' if v else 'FAIL'}" for k, v in checks.items()))
    rec["all_pass"] = bool(ok)
    out.write_text(json.dumps(rec, indent=1))
    print("all pass" if ok else "some checks fail", f"({dtype})")
    return 0


def report(paths: list[Path]) -> int:
    recs = [json.loads(p.read_text()) for p in paths]
    r0 = recs[0]
    f = lambda w: "-" if not w else f"{100 * w['signed']:+.2f} % ({w['at_T_MeV']:.3g})"
    print("| Material | ESTAR collision 0.1-1000 MeV | ESTAR radiative 1-1000 MeV | PDG ionization 0.1-100 GeV | "
          "brems 10 GeV-1 TeV | pair | photonuclear | total radiative |")
    print("|---|---|---|---|---|---|---|---|")
    for name, m in r0["materials"].items():
        e, q = m["estar"], m["pdg"]
        tag = lambda ok: "" if ok else " (material differs)"
        print(f"| {name} | {f(e['collision'])}{tag(e['material_match'])} | {f(e['radiative'])} | "
              f"{f(q['ionization_numpy'])}{tag(q['material_match'])} | {f(q['brems'])} | {f(q['pair'])} | "
              f"{f(q['photonuc'])} | {f(q['radloss'])} |")
    for r in recs:
        dev = max(m["pdg"]["device_vs_numpy"] for m in r["materials"].values())
        fails = [f"{n}:{k}" for n, m in r["materials"].items() for k, v in m["checks"].items() if not v]
        print(f"\n{r['dtype']}: device vs NumPy ionization max {dev:.2e}; failed checks: {', '.join(fails) or 'none'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--report", type=Path, nargs="+")
    a = ap.parse_args(argv)
    if a.report:
        return report(a.report)
    return run(a.out or Path(f"record.json"))


if __name__ == "__main__":
    sys.exit(main())
