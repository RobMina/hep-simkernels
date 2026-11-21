"""Geant4 slab references for hep-simkernels: GDML files and hepsimlib-recorder macros for every case, half and shard.

    python validation/g4slab/campaign.py RUN_DIR [--campaign validation/g4slab/campaign.json] [--scale 0.02]

Writes RUN_DIR/<case>.gdml and RUN_DIR/<case>_<half>_<shard>.mac (half A or B). Run each macro from RUN_DIR with
`hepsimlib-recorder FTFP_BERT_EMZ <macro> > <macro stem>.log` (one Geant4 thread each); the log keeps the couple table
(`/run/verbose 2`), from which analyze.py reads the production thresholds in energy. --scale shrinks every case for a
pilot. The recorder (github.com/RobMina/hep-simlib, recorder/) places the slab, a G4Box envelope of the case's NIST
material, in a G4_Galactic world; the gun starts 1 um behind the centre of the -z face, along +z.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GDML = """<?xml version="1.0" encoding="UTF-8"?>
<!-- hep-simkernels slab validation: one slab of {material}, {thickness} mm thick, lateral half size {lateral} mm. -->
<gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:noNamespaceSchemaLocation="http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml.xsd">
  <define/>
  <materials/>
  <solids><box name="slab_box" x="{full}" y="{full}" z="{thickness}" lunit="mm"/></solids>
  <structure><volume name="slab"><materialref ref="{material}"/><solidref ref="slab_box"/></volume></structure>
  <setup name="Default" version="1.0"><world ref="slab"/></setup>
</gdml>
"""


def seed(base: int, case_index: int, half: str, shard: int) -> int:
    return base + 100_000 * case_index + (50_000 if half == "B" else 0) + shard


def write(run: Path, campaign: dict, scale: float = 1.0) -> list[Path]:
    run.mkdir(parents=True, exist_ok=True)
    lateral = campaign["lateral_half_mm"]
    macros = []
    for ci, c in enumerate(campaign["cases"]):
        name, t = c["case"], c["thickness_mm"]
        (run / f"{name}.gdml").write_text(GDML.format(material=c["material"], thickness=t, lateral=lateral,
                                                       full=2 * lateral))
        events = max(1, int(round(c["events"] * scale)))
        per = campaign["shard_events"]
        shards = [per] * (events // per) + ([events % per] if events % per else [])
        for half in ("A", "B"):
            for s, n in enumerate(shards):
                stem = f"{name}_{half}_{s:03d}"
                mac = "\n".join([
                    "/control/verbose 0", "/run/verbose 2",
                    f"/recorder/gdml {name}.gdml", "/recorder/envelope slab",
                    f"/recorder/geometryTag {name}", f"/recorder/physicsTag {campaign['physics_list']}-default-cuts",
                    f"/recorder/output {stem}.root", f"/recorder/run {ci + 1}",
                    f"/recorder/subrun {s + (1000 if half == 'B' else 0)}",
                    f"/recorder/seed {seed(campaign['seed_base'], ci, half, s)}",
                    f"/recorder/approach {campaign['approach_um']} um",
                    "/run/initialize",
                    f"/recorder/entrance {c['pdg']} 0 0 {-t / 2} 0 0 1 {c['T_MeV']} {n}",
                    f"/run/beamOn {n}", ""])
                p = run / f"{stem}.mac"
                p.write_text(mac)
                macros.append(p)
    (run / "campaign.json").write_text(json.dumps({**campaign, "scale": scale}, indent=1))
    return macros


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--campaign", type=Path, default=Path(__file__).with_name("campaign.json"))
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args(argv)
    macros = write(a.run, json.loads(a.campaign.read_text()), a.scale)
    print(f"{len(macros)} macros in {a.run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
