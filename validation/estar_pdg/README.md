# Stopping powers against NIST ESTAR and the PDG muon tables

Public tables only (`validation/tables`, written by `fetch_tables.py`; each entry keeps its source URL, read date and
raw sha256):
- NIST ESTAR collision and radiative stopping powers for electrons, 0.01-1000 MeV (public domain);
- the PDG 2024 muon energy-loss tables (Groom, Mokhov and Striganov, Atomic Data and Nuclear Data Tables 78 (2001)
  183): ionization, bremsstrahlung, pair production and photonuclear losses, 10 MeV-1 TeV used here.

They cover the twelve built-in materials.

| Kernel | Geant4 11.3.2 model it re-implements | Compared with |
|---|---|---|
| `eioni.dedx_restricted(T, tcut=inf)` | G4MollerBhabhaModel | ESTAR collision |
| `ebrem.dedx_restricted(T, kcut=inf)` | G4SeltzerBergerModel | ESTAR radiative |
| `eloss.dedx_restricted(p, tcut=inf)`, and `eloss_jax` in float32 and float64 | G4MuBetheBlochModel without shell, higher-order and radiative corrections | PDG ionization |
| `muradiative` brems, pair, nuclear (b E) | G4MuBremsstrahlungModel, G4MuPairProductionModel, G4KokoulinMuonNuclearXS | PDG brems, pair, photonuc |

These tables are not Geant4. Agreement within the criteria shows the kernels compute what the tables compute. A
difference is either a difference between the Geant4 model and the table, or a defect of the kernel. Telling the
two apart needs Geant4's own value of the same quantity (for example from G4EmCalculator), which is not part of
this check.

## Criteria (`criteria.json`, declared before the first comparison)

The largest |kernel / table - 1| over each range must stay within the limit:

| Row | Range | Limit |
|---|---|---|
| ESTAR collision | 0.1-1000 MeV | 1.5 % |
| ESTAR radiative | 1-1000 MeV | 1 % |
| PDG ionization | 0.1-100 GeV | 1 % (NumPy and device) |
| brems, pair, total radiative | 10 GeV-1 TeV | 5 % |
| photonuclear | 10 GeV-1 TeV | 15 % |
| device vs NumPy ionization | all | 1e-12 (float64), 1e-5 (float32) |

Collision and ionization rows have no pass line for a material whose density or I differs from the table's by more
than 1 %: polyethylene (0.94 against 0.89 g/cm3 in the PDG table) and silicon dioxide (2.32 against 2.20, fused
quartz).

## Result (2026-11-21, `e111f42`, CPU, both device floats): tested, not closed

`record_float64.json` and `record_float32.json` hold every check and the deviation at every table energy. Worst
deviation (sign, and the kinetic energy in MeV where it occurs):

| Material | ESTAR collision 0.1-1000 MeV | ESTAR radiative 1-1000 MeV | PDG ionization 0.1-100 GeV | brems 10 GeV-1 TeV | pair | photonuclear | total radiative |
|---|---|---|---|---|---|---|---|
| G4_Al | +0.33 % (0.8) | -0.91 % (25) | -1.94 % (1e+05) | +2.56 % (1e+04) | +2.54 % (2e+04) | -3.82 % (3.5e+05) | +2.26 % (2e+04) |
| G4_Si | +0.36 % (1) | -0.93 % (25) | -1.97 % (1e+05) | +2.40 % (1e+04) | +2.62 % (2e+04) | -3.91 % (3.5e+05) | +2.25 % (2e+04) |
| G4_Cu | +0.12 % (5.5) | **-1.02 % (25)** | -2.03 % (1e+05) | +1.42 % (1e+04) | +4.88 % (2e+04) | -4.58 % (3.5e+05) | +3.08 % (2e+04) |
| G4_W | +0.18 % (2) | -0.56 % (5) | -2.08 % (1e+05) | +0.75 % (3e+05) | **+18.62 % (2e+04)** | +4.61 % (2e+04) | **+9.80 % (2e+04)** |
| G4_Au | +0.16 % (0.8) | -0.57 % (5) | -2.09 % (1e+05) | +0.74 % (3e+05) | **+20.93 % (1e+04)** | +4.77 % (2e+04) | **+10.81 % (2e+04)** |
| G4_Pb | +0.14 % (1.5) | -0.54 % (5) | -2.03 % (1e+05) | +0.73 % (3e+05) | **+22.62 % (1e+04)** | +4.16 % (1.2e+04) | **+11.37 % (2e+04)** |
| G4_AIR | +0.30 % (200) | +0.40 % (400) | -1.60 % (1e+05) | +3.71 % (1e+04) | +2.17 % (2.5e+04) | -2.54 % (8e+05) | +2.58 % (2e+04) |
| G4_CESIUM_IODIDE | +0.45 % (0.8) | -0.70 % (8) | -1.98 % (1e+05) | +0.91 % (1e+04) | **+11.19 % (2e+04)** | -4.23 % (7e+05) | **+6.21 % (2e+04)** |
| G4_POLYETHYLENE | +0.96 % (4) | +0.78 % (300) | -2.03 % (1e+05), no pass line | +3.91 % (1e+04) | +2.18 % (8e+04) | +3.88 % (2e+04) | +3.09 % (2e+04) |
| G4_MYLAR | +1.40 % (4.5) | +0.52 % (300) | -1.90 % (1e+05) | +3.89 % (1e+04) | +2.13 % (2e+04) | +2.87 % (1.7e+04) | +2.75 % (2e+04) |
| G4_POLYSTYRENE | -0.28 % (0.8) | +0.66 % (300) | -1.90 % (1e+05) | +4.09 % (1e+04) | +2.15 % (7e+04) | +3.32 % (1.7e+04) | +2.95 % (2e+04) |
| G4_SILICON_DIOXIDE | -0.80 % (3.5) | -0.72 % (25) | -2.12 % (1e+05), no pass line | +2.86 % (1e+04) | +2.47 % (2e+04) | -3.32 % (3.5e+05) | +2.34 % (2e+04) |

Outcome per row:
- **Passes:** electron collision (all twelve within 1.4 %), muon bremsstrahlung (within 4.1 %), photonuclear (within
  4.8 %), and the JAX ionization twin, which equals NumPy to 1.2e-15 in float64 and 4.5e-7 in float32.
- **Electron radiative:** passes in eleven materials. Copper fails by a hair: -1.02 % at 20-25 MeV against a 1 % line
  (-0.3 % at 1 MeV, back to 0 at 1 GeV).
- **Muon ionization fails the declared range in every material.** The deviation is a smooth shortfall that grows
  with energy and is the same in every material: within 0.3 % at 0.1-1 GeV, -0.8 % at 10 GeV, -1.3 % at 30 GeV, -2 %
  at 100 GeV, -3.7 % at 1 TeV. It matches the corrections the kernel omits on purpose, which the PDG tables include.
  The kernel's docstring says they stay below 1 % "for 0.1-100 GeV muons"; measured against PDG, that holds up to
  about 30 GeV. Below 100 MeV the high-Z materials read high (+1.9 % W, +2.5 % Pb at 10 MeV): shell corrections, also
  omitted.
- **Muon pair production fails for high Z.** The excess grows with Z (Al +0.7 %, Cu +2.4 %, CsI +9 %, W/Au/Pb
  +18-23 % at 10 GeV) and falls with energy (W/Au/Pb +8-9.5 % at 1 TeV). The PDG table states that its pair term
  includes post-Born (Coulomb) corrections, which are largest at high Z. Whether G4MuPairProductionModel itself sits
  this far from the table, or the kernel departs from it, is open: it needs Geant4's own dE/dx for the same material
  and energy.

Status: **tested, not closed** for muon ionization above about 30 GeV and for muon pair production in high-Z
materials. The criteria stay as declared.

## Running

```bash
python validation/tables/fetch_tables.py          # refetch; compare the sha256 with the committed tables
HEPSIMK_DEVICE_FLOAT=float64 python validation/estar_pdg/compare.py --out validation/estar_pdg/record_float64.json
HEPSIMK_DEVICE_FLOAT=float32 python validation/estar_pdg/compare.py --out validation/estar_pdg/record_float32.json
python validation/estar_pdg/compare.py --report validation/estar_pdg/record_float64.json validation/estar_pdg/record_float32.json
```

`run_uva.sbatch` runs all three on one core (measured: 30-38 s, 1.4-1.6 GB).
