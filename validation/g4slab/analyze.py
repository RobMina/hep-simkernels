"""Compare hep-simkernels' slab propagation with the Geant4 slab captures of campaign.py.

    python validation/g4slab/analyze.py RUN_DIR --output record.json [--criteria validation/g4slab/criteria.json]
        [--cases a,b] [--device-events N]

Per case it reads every capture of halves A and B (hepsimlib.capture.tables, "tracks" table) and the production
thresholds in energy from the shards' Geant4 couple tables. It then builds the arms of criteria.json: Geant4 B vs A
(null), eslab (NumPy, float64) and eslab_jax in float64 and float32 (device; both run in this process, the device
float being set per call). Observables: the primary's fate (transmitted, reflected, absorbed = stopped or annihilated),
the transmitted primaries' energy loss, exit polar angle and lateral displacement on the exit face, the reflected
primaries' exit energy, and the primary's direct photons (eBrem) and delta rays (eIoni) per primary and their energy
spectra (NumPy arm only: the device arm returns counts per track).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy import stats

ME = 0.51099895
UNIT = {"eV": 1e-6, "keV": 1e-3, "MeV": 1.0, "GeV": 1e3}


def cuts_from_log(text: str, material: str) -> tuple[float, float]:
    """(gamma, e-) production thresholds [MeV] of `material` in a Geant4 /run/verbose 2 couple table."""
    for block in text.split("Index :")[1:]:
        if re.search(rf"Material\s*:\s*{re.escape(material)}\b", block):
            m = re.search(r"Energy thresholds\s*:\s*gamma\s+([\d.eE+-]+)\s*(\w+)\s+e-\s+([\d.eE+-]+)\s*(\w+)", block)
            if m:
                return float(m[1]) * UNIT[m[2]], float(m[3]) * UNIT[m[4]]
    raise ValueError(f"no couple for {material}")


def read_half(run: Path, case: dict, half: str) -> dict:
    """Per-primary and per-child arrays of one Geant4 half."""
    from hepsimlib.capture import tables
    t2 = case["thickness_mm"] / 2
    parts, n_events, cuts = [], 0, set()
    for f in sorted(run.glob(f"{case['case']}_{half}_*.root")):
        tr = tables.read(f, names=("tracks",))["tracks"]
        cuts.add(cuts_from_log(f.with_suffix(".log").read_text(), case["material"]))
        ev = tr["subrun"].astype(np.int64) * 10_000_000 + tr["event"].astype(np.int64)
        prim = tr["parent"] == 0
        if np.unique(ev[prim]).size != prim.sum():
            raise ValueError(f"{f.name}: not one primary per event")
        order = np.argsort(ev[prim])
        P = {k: np.asarray(v)[prim][order] for k, v in tr.items()}
        child = tr["parent"] == 1
        parts.append({"P": P, "child_ev": ev[child], "child_pdg": tr["pdg"][child],
                      "child_creator": np.asarray(tr["creator"])[child], "child_T": tr["begin_kinetic"][child]})
        n_events += int(prim.sum())
    if not parts:
        raise FileNotFoundError(f"no captures for {case['case']} {half}")
    if len(cuts) != 1:
        raise ValueError(f"{case['case']}: shards disagree on the production thresholds {cuts}")
    cat = lambda key: np.concatenate([p["P"][key] for p in parts])
    reason, stop_proc = cat("reason"), cat("stopProcess")
    lz, Tend = cat("end_lz"), cat("end_kinetic")
    exited = reason == 2
    refl = exited & (lz < 0)
    trans = exited & ~refl
    annih = (reason == 1) & (stop_proc == "annihil") & (Tend > 0)
    stop = (reason == 1) & ~annih
    d = np.stack([cat("end_lpx"), cat("end_lpy"), cat("end_lpz")], axis=1)
    d = d / np.maximum(np.linalg.norm(d, axis=1), 1e-300)[:, None]
    theta = np.arctan2(np.hypot(d[:, 0], d[:, 1]), d[:, 2])
    r = np.hypot(cat("end_lx"), cat("end_ly"))
    cc = np.concatenate([p["child_creator"] for p in parts])
    cp = np.concatenate([p["child_pdg"] for p in parts])
    cT = np.concatenate([p["child_T"] for p in parts])
    kcut, tcut = cuts.pop()
    return {"n": n_events, "T": np.where(exited, Tend, 0.0), "trans": trans, "refl": refl, "stop": stop,
            "annih": annih, "theta": theta, "r": r, "kcut": kcut, "tcut": tcut,
            "photons": cT[(cc == "eBrem") & (cp == 22)], "deltas": cT[(cc == "eIoni") & (cp == 11)]}


def merge(a: dict, b: dict) -> dict:
    out = {"n": a["n"] + b["n"], "kcut": a["kcut"], "tcut": a["tcut"]}
    for k in ("T", "trans", "refl", "stop", "annih", "theta", "r", "photons", "deltas"):
        out[k] = np.concatenate([a[k], b[k]])
    return out


def theta_of(d):
    return np.arctan2(np.hypot(d[:, 0], d[:, 1]), d[:, 2])


def numpy_arm(case, n, tcut, kcut, seed):
    from hepsimkernels import eslab, materials
    mat = materials.get(case["material"])
    r = eslab.electron_slab(np.random.default_rng(seed), mat, np.full(n, case["T_MeV"]), np.tile([0.0, 0.0, 1.0], (n, 1)),
                            case["thickness_mm"], tcut, kcut, case["pdg"] == -11, msc=True, annihilation=True)
    return {"n": n, "T": r.T, "trans": ~r.stopped & ~r.reflected, "refl": r.reflected,
            "stop": r.stopped & ~r.annihilated, "annih": r.annihilated, "theta": theta_of(r.direction),
            "r": np.hypot(r.position[:, 0], r.position[:, 1]), "photons": r.photons.energy, "deltas": r.deltas.energy}


def device_arm(case, n, tcut, kcut, seed, dtype):
    import jax
    import jax.numpy as jnp
    from hepsimkernels import eslab_jax, materials
    mat = materials.get(case["material"])
    T = jnp.full(n, case["T_MeV"], dtype)
    d = jnp.tile(jnp.asarray([0.0, 0.0, 1.0], dtype), (n, 1))
    r = eslab_jax.electron_slab(jax.random.key(seed), mat, T, d, case["thickness_mm"], tcut, kcut, case["pdg"] == -11,
                                msc=True, annihilation=True)
    g = {k: np.asarray(getattr(r, k), np.float64) for k in ("T", "direction", "position")}
    st, rf, an = (np.asarray(getattr(r, k), bool) for k in ("stopped", "reflected", "annihilated"))
    return {"n": n, "T": g["T"], "trans": ~st & ~rf, "refl": rf, "stop": st & ~an, "annih": an,
            "theta": theta_of(g["direction"]), "r": np.hypot(g["position"][:, 0], g["position"][:, 1]),
            "n_photons": int(np.asarray(r.n_photons).sum()), "n_deltas": int(np.asarray(r.n_deltas).sum())}


def edges_from(x, nbins):
    """Equal-probability edges of the reference, moved off values it takes (exact edges, e.g. theta = 0)."""
    x = x[np.isfinite(x)]
    e = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    u = np.unique(x)
    out = []
    for v in e:
        i = int(np.searchsorted(u, v))
        if i < u.size and u[i] == v:
            v = 0.5 * (u[i - 1] + u[i]) if i > 0 else u[0] - 1e-9 * max(abs(u[0]), 1.0)
        out.append(v)
    out[-1] = max(out[-1], x.max()) * (1 + 1e-12) + 1e-300
    return np.unique(np.array(out))


def chi2_test(a, b, edges):
    """Two-sample chi2 (samples of any size) on shared bins; returns chi2, ndf, p."""
    ha, hb = np.histogram(np.clip(a, edges[0], edges[-1]), edges)[0].astype(float), \
        np.histogram(np.clip(b, edges[0], edges[-1]), edges)[0].astype(float)
    na, nb = ha.sum(), hb.sum()
    m = (ha + hb) > 0
    if na == 0 or nb == 0 or m.sum() < 2:
        return {"chi2": None, "ndf": 0, "p": 1.0, "n": [int(na), int(nb)]}
    c2 = float((((np.sqrt(nb / na) * ha - np.sqrt(na / nb) * hb) ** 2)[m] / (ha + hb)[m]).sum())
    ndf = int(m.sum()) - 1
    return {"chi2": c2, "ndf": ndf, "p": float(stats.chi2.sf(c2, ndf)), "n": [int(na), int(nb)]}


def z_test(ka, na, kb, nb, kind="binomial"):
    """Two-sample test of a fraction (binomial) or a mean count per primary (Poisson)."""
    pa, pb = ka / na, kb / nb
    if kind == "binomial":
        pp = (ka + kb) / (na + nb)
        se = np.sqrt(max(pp * (1 - pp), 1e-300) * (1 / na + 1 / nb))
    else:
        se = np.sqrt(max(ka, 1) / na ** 2 + max(kb, 1) / nb ** 2)
    z = (pb - pa) / se if se > 0 else 0.0
    return {"ref": float(pa), "arm": float(pb), "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}


def compare(ref: dict, arm: dict, case: dict, nbins: int, spectra: bool) -> dict:
    T0 = case["T_MeV"]
    out = {}
    # absorbed = stopped or annihilated: Geant4's end state does not tell annihilation in flight from at rest
    # (the in-flight step also ends with zero kinetic energy), so the two are compared together
    for k, sel in (("trans", ref["trans"]), ("refl", ref["refl"]), ("absorbed", ref["stop"] | ref["annih"])):
        a_sel = arm["trans"] if k == "trans" else arm["refl"] if k == "refl" else arm["stop"] | arm["annih"]
        if sel.sum() + a_sel.sum() > 0:
            out[f"frac_{k}"] = z_test(sel.sum(), ref["n"], a_sel.sum(), arm["n"])
    lr, la = T0 - ref["T"][ref["trans"]], T0 - arm["T"][arm["trans"]]
    if lr.size > 10 * nbins:
        out["loss"] = chi2_test(lr, la, edges_from(lr, nbins))
        for k in ("theta", "r"):
            out[k] = chi2_test(ref[k][ref["trans"]], arm[k][arm["trans"]], edges_from(ref[k][ref["trans"]], nbins))
    rr, ra = ref["T"][ref["refl"]], arm["T"][arm["refl"]]
    if rr.size > 10 * nbins:
        out["refl_T"] = chi2_test(rr, ra, edges_from(rr, nbins))
    for k in ("photons", "deltas"):
        na_ = arm[k].size if spectra else arm[f"n_{k}"]
        out[f"{k}_per_primary"] = z_test(ref[k].size, ref["n"], na_, arm["n"], kind="poisson")
        if spectra and ref[k].size > 10 * nbins:
            out[f"{k}_spectrum"] = chi2_test(ref[k], arm[k], edges_from(ref[k], nbins))
    return out


def verdicts(records: list[dict], crit: dict) -> dict:
    pvals = {}
    for rec in records:
        for name, c in rec["cases"].items():
            reported = name in crit.get("reported_without_pass_line", {})
            skip = {t for a in crit.get("amendments", []) for t, cases in a.get("tests_without_pass_line", {}).items()
                    if name in cases}
            for arm_name, tests in c["arms"].items():
                if not reported or arm_name == "null":
                    pvals.setdefault(arm_name, []).extend(
                        t["p"] for k, t in tests.items() if arm_name == "null" or k not in skip)
    out = {}
    for arm_name, ps in pvals.items():
        bmin = min(1.0, min(ps) * len(ps))
        out[arm_name] = {"n_tests": len(ps), "bonferroni_min_p": bmin, "pass": bmin >= 0.001}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path, nargs="?")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--criteria", type=Path, default=Path(__file__).with_name("criteria.json"))
    ap.add_argument("--cases")
    ap.add_argument("--arms", default="null,numpy", help="null,numpy and/or device (in the device float of "
                    "HEPSIMK_DEVICE_FLOAT; run once per float)")
    ap.add_argument("--device-events", type=int)
    ap.add_argument("--combine", type=Path, nargs="+", help="merge partial records and give the verdict per arm")
    a = ap.parse_args(argv)
    crit_bytes = a.criteria.read_bytes()
    crit = json.loads(crit_bytes)
    if a.combine:
        recs = [json.loads(p.read_text()) for p in a.combine]
        merged = {"criteria_sha256": hashlib.sha256(crit_bytes).hexdigest(), "campaign": recs[0]["campaign"],
                  "cases": {}, "timing_s": {}}
        for r in recs:
            if r["criteria_sha256"] != merged["criteria_sha256"]:
                raise SystemExit("partial records were made with other criteria")
            merged["timing_s"].update(r["timing_s"])
            for name, c in r["cases"].items():
                m = merged["cases"].setdefault(name, {k: v for k, v in c.items() if k != "arms"} | {"arms": {}})
                m["arms"].update(c["arms"])
        merged["verdict"] = verdicts([merged], crit)
        a.output.write_text(json.dumps(merged, indent=1))
        print(json.dumps(merged["verdict"], indent=1))
        return 0
    arms = set(a.arms.split(","))
    if "device" in arms:
        import jax
        from hepsimkernels import precision
        if precision.FLOAT == np.float64:
            jax.config.update("jax_enable_x64", True)
        dt = precision.FLOAT
    camp = json.loads((a.run / "campaign.json").read_text())
    nbins = int(crit["bins"])
    dev_max = a.device_events or int(crit["device_events_max"])
    wanted = set(a.cases.split(",")) if a.cases else None
    rec = {"criteria_sha256": hashlib.sha256(crit_bytes).hexdigest(), "campaign": camp, "cases": {}, "timing_s": {}}
    for ci, case in enumerate(camp["cases"]):
        name = case["case"]
        if wanted and name not in wanted:
            continue
        A, B = read_half(a.run, case, "A"), read_half(a.run, case, "B")
        G = merge(A, B)
        tcut, kcut = G["tcut"], G["kcut"]
        c = {"events_g4": [A["n"], B["n"]], "tcut_MeV": tcut, "kcut_MeV": kcut, "arms": {}}
        if "null" in arms:
            c["arms"]["null"] = compare(A, B, case, nbins, spectra=True)
        if "numpy" in arms:
            t0 = time.time()
            c["arms"]["numpy"] = compare(G, numpy_arm(case, G["n"], tcut, kcut, 7919 + ci), case, nbins, spectra=True)
            rec["timing_s"][f"{name}/numpy"] = time.time() - t0
        if "device" in arms:
            nd = min(G["n"], dev_max)
            t0 = time.time()
            arm = device_arm(case, nd, tcut, kcut, 104729 + ci + (0 if dt == np.float64 else 50), dt)
            rec["timing_s"][f"{name}/device_{dt.name}"] = time.time() - t0
            c["arms"][f"device_{dt.name}"] = compare(G, arm, case, nbins, spectra=False)
        rec["cases"][name] = c
        worst = {k: min(t["p"] for t in v.values()) for k, v in c["arms"].items()}
        print(f"{name}: G4 {A['n']}+{B['n']}, tcut {tcut:.4g} kcut {kcut:.4g} MeV, min p per arm "
              + ", ".join(f"{k} {v:.2g}" for k, v in worst.items()), flush=True)
    rec["verdict"] = verdicts([rec], crit)
    a.output.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
