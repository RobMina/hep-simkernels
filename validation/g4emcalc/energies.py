"""Print the PDG muon-table kinetic energies from 10 MeV to 1 TeV, one per line (the emcalc energy grid)."""
import json
from pathlib import Path

T = json.loads((Path(__file__).resolve().parent.parent / "tables" / "pdg_muon.json").read_text())["materials"]["G4_Al"]["T_MeV"]
print("\n".join(f"{t:.6g}" for t in T if 10.0 <= t <= 1e6))
