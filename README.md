# hep-simkernels

Geant4-matched particle-matter kernels for fast simulation. Each kernel reproduces a named Geant4 11.3.2 model in
NumPy (the reference) and, where it runs on a device, in JAX with float32 and float64 (`hepsimkernels.precision`).

| Module | Physics | Geant4 model it matches |
|---|---|---|
| `eloss`, `eloss_jax` | muon restricted energy loss with fluctuations, delta rays, Birks, radiation length, Highland angle | G4MuBetheBlochModel, G4UniversalFluctuation / G4UrbanFluctuation, G4EmSaturation |
| `eioni` | electron and positron ionisation with delta production | G4MollerBhabhaModel |
| `emsc` | e+- multiple scattering (mixed class-II scheme on the screened-Rutherford cross section, Mott-corrected) | G4WentzelVIModel / G4GoudsmitSaundersonMscModel (EM option 4), G4ScreeningMottCrossSection |
| `ebrem` | bremsstrahlung with Seltzer-Berger tables and 2BS angles | G4SeltzerBergerModel, G4Generator2BS |
| `eannihil` | positron annihilation in flight and at rest | G4eeToTwoGammaModel, G4SimplePositronAtRestModel |
| `muradiative`, `muradiative_jax` | muon bremsstrahlung, pair production, photonuclear | G4MuBremsstrahlungModel, G4MuPairProductionModel, G4KokoulinMuonNuclearXS |
| `eslab`, `eslab_jax`, `slab_jax` | electron propagation through slabs (and a Pallas GPU kernel of the slab stepper) | the processes above, stepped |
| `world`, `transport`, `crossings`, `geometry.annulus_jax` | GDML geometry as convex pieces, straight-line ray tracing, sliced slab loss and correlated multiple scattering, annular foils | - |
| `ehelix` | helix and RK4 stepping in constant or mapped fields; G4BL-format field maps | - |
| `materials` | NIST materials with density-effect and fluctuation parameters; `register()` for an experiment's own | G4NistMaterialBuilder, G4DensityEffectData, G4IonisParamMat |

Bundled data: compact copies of G4EMLOW 8.6.1 `brem_SB` and of the G4MottData coefficients, each checked against
its manifest and content hash on load (through [hep-simlib](https://github.com/RobMina/hep-simlib)'s library store).

## Settings

Environment variables are `<PREFIX>_<KEY>`, with the prefix from `HEPSIMKERNELS_ENV_PREFIX` (default `HEPSIMK`):
`DEVICE_FLOAT` (float32, the default, or float64), `DEVICE_DETERMINISTIC` (1, the default, turns XLA autotuning
off so GPU output repeats bit for bit), `SLAB_KERNEL` (auto, xla or pallas), and `F64_AUDIT`. Set the prefix
before importing any kernel.

## Experiment hooks

Set in `hepsimkernels.settings` (standard library only, so installing them imports no NumPy and fixes no device
float):

- `settings.MATERIAL_MODULES`: modules imported when `hepsimkernels.materials` is first imported. Each registers the
  materials of an experiment's geometry with `materials.register(*materials)` (by GDML name) and per-region delta
  production thresholds with `materials.add_cuts(name, region=cut_MeV)`.
- `settings.MAP_SCALE`: a callable `header_path -> factor` that `ehelix.read_g4bl` applies to a field map.
- `settings.PROVENANCE_HOOKS`: callables whose dicts `precision.provenance()` adds.

## Origin and validation

Extracted from [mu2e-fastsim](https://github.com/RobMina/mu2e-fastsim), which keeps its geometry's materials, its
field-map scale and its GPU launch settings and uses this package for the rest (see NOTICE). The kernels were
validated there against Geant4 (slab and gun comparisons, and chains compared with full simulation stage by stage).
The tests here (`python -m pytest`) check every kernel against its NumPy reference and against the Geant4 numbers
in the tests, on CPU.

## Citing

Please cite the release you used: `CITATION.cff` holds the metadata once a release exists, and each tagged release
is archived on Zenodo with its own DOI.
