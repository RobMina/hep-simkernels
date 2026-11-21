"""Fetch the public reference tables of the hep-simkernels validation and write them as JSON with their provenance.

    python validation/tables/fetch_tables.py [--out validation/tables]

- NIST ESTAR (public domain, US Government work): electron collision and radiative stopping powers on ESTAR's default
  grid (0.01-1000 MeV), from the text form (physics.nist.gov/PhysRefData/Star/Text/ESTAR-t.html).
- PDG Atomic and Nuclear Properties, 2024 edition: muon energy-loss tables (Groom, Mokhov and Striganov, Atomic Data
  and Nuclear Data Tables 78 (2001) 183), the MUE/muE_<material>.txt files.

Each material entry records the source URL, the read date and the sha256 of the raw response, so a later fetch can be
compared with the committed one.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

ESTAR_URL = "https://physics.nist.gov/cgi-bin/Star/e_table-t.pl"
PDG_URL = "https://pdg.lbl.gov/2024/AtomicNuclearProperties/MUE/muE_{}.txt"
# hep-simkernels material -> (ESTAR material number, PDG file stem)
MATERIALS = {
    "G4_Al": ("013", "aluminum_Al"), "G4_Si": ("014", "silicon_Si"), "G4_Cu": ("029", "copper_Cu"),
    "G4_W": ("074", "tungsten_W"), "G4_Au": ("079", "gold_Au"), "G4_Pb": ("082", "lead_Pb"),
    "G4_AIR": ("104", "air_dry_1_atm"), "G4_CESIUM_IODIDE": ("141", "cesium_iodide_CsI"),
    "G4_POLYETHYLENE": ("221", "polyethylene"), "G4_MYLAR": ("222", "polyethylene_terephthalate_Mylar"),
    "G4_POLYSTYRENE": ("226", "polystyrene"), "G4_SILICON_DIOXIDE": ("245", "silicon_dioxide_fused_quartz"),
}
NUM = r"[-+]?\d\.\d+E[-+]\d+"


def get(url: str, data: dict | None = None) -> bytes:
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers={"User-Agent": "hep-simkernels-validation"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def parse_estar(raw: bytes) -> dict:
    text = raw.decode("latin-1").replace("<br>", "\n")
    rows = [list(map(float, m)) for m in re.findall(rf"^\s*({NUM})\s+({NUM})\s+({NUM})\s+({NUM})", text, re.M)]
    if len(rows) < 50:
        raise ValueError(f"ESTAR: {len(rows)} rows")
    title = re.search(r"Electrons in\s+(.+?)\s*\n", text)
    dens = re.search(r"Density, g/cm3\s+I, eV\s*\n\s*(" + NUM + r")\s+([\d.]+)", text)
    return {"title": title[1].strip() if title else None, "density_g_cm3": float(dens[1]) if dens else None,
            "I_eV": float(dens[2]) if dens else None, "T_MeV": [r[0] for r in rows],
            "collision_MeV_cm2_g": [r[1] for r in rows], "radiative_MeV_cm2_g": [r[2] for r in rows]}


def parse_pdg(raw: bytes) -> dict:
    text = raw.decode("latin-1")
    cols = ["T_MeV", "p_MeV", "ionization", "brems", "pair", "photonuc", "radloss", "dEdx", "csda_g_cm2", "delta",
            "beta", "dEdx_R"]
    rows = [list(map(float, l.split()[:12])) for l in text.splitlines()
            if re.match(rf"^\s*{NUM}\s+{NUM}", l) and len(l.split()) >= 12]
    if len(rows) < 50:
        raise ValueError(f"PDG: {len(rows)} rows")
    out = {c: [r[i] for r in rows] for i, c in enumerate(cols)}
    dens = re.search(r"density\s*=\s*([\d.E+-]+)", text)
    stern = re.search(r"Sternheimer coef:.*\n\s*((?:[-\d.]+\s+){7})", text)
    out["material"] = (re.search(r"Index =\s*\d+:\s*(.+)", text) or [None, None])[1]
    out["density_g_cm3"] = float(dens[1]) if dens else None
    out["I_eV"] = float(stern[1].split()[4]) if stern else None
    out["units"] = "T, p in MeV (MeV/c); losses in MeV cm2/g; Ionization etc. as in the PDG file"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=Path(__file__).parent)
    a = ap.parse_args(argv)
    today = datetime.date.today().isoformat()
    estar, pdg = {}, {}
    for name, (matno, stem) in MATERIALS.items():
        raw = get(ESTAR_URL, {"matno": matno, "ShowDefault": "on"})
        estar[name] = {"source": f"{ESTAR_URL} (POST matno={matno}, ShowDefault=on)", "read": today,
                       "sha256": hashlib.sha256(raw).hexdigest(), **parse_estar(raw)}
        url = PDG_URL.format(stem)
        raw = get(url)
        pdg[name] = {"source": url, "read": today, "sha256": hashlib.sha256(raw).hexdigest(), **parse_pdg(raw)}
        print(f"{name}: ESTAR {len(estar[name]['T_MeV'])} rows ({estar[name]['title']}), "
              f"PDG {len(pdg[name]['T_MeV'])} rows")
    (a.out / "estar.json").write_text(json.dumps({"reference": "NIST ESTAR, physics.nist.gov/PhysRefData/Star/Text/ESTAR.html",
                                                  "materials": estar}, indent=1))
    (a.out / "pdg_muon.json").write_text(json.dumps({
        "reference": "D.E. Groom, N.V. Mokhov, S.I. Striganov, Muon stopping-power and range tables, 10 MeV-100 TeV, "
                     "Atomic Data and Nuclear Data Tables 78 (2001) 183; PDG Atomic and Nuclear Properties 2024",
        "materials": pdg}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
