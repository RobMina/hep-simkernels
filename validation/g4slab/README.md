# Electron and positron slabs against Geant4

Public inputs only: one slab of a NIST material in vacuum (`campaign.py` writes the GDML), a fixed-energy e- or e+ gun
at normal incidence, and Geant4 11.3.2 through the experiment-neutral recorder of
[hep-simlib](https://github.com/RobMina/hep-simlib) (`recorder/`, built from source in the job). The reference
configuration, chosen by the user: `FTFP_BERT_EMZ` (EM option 4) at its defaults, default production cuts.

| Case | Material, thickness | Particle, energy | Geant4 events |
|---|---|---|---|
| al_0p1_e-_105, al_0p1_e-_95 | Al 0.1 mm | e- 105, 95 MeV | 2 x 500k each |
| al_1_e-_105, al_1_e+_105, al_1_e-_20 | Al 1 mm | e- 105, e+ 105, e- 20 MeV | 2 x 500k each |
| mylar_2_e-_105, mylar_2_e+_50 | Mylar 2 mm | e- 105, e+ 50 MeV | 2 x 500k each |
| w_0p35_e-_105, w_0p35_e+_50 | W 0.35 mm | e- 105, e+ 50 MeV | 2 x 500k each |
| w_3p5_e-_105 | W 3.5 mm (1 X0) | e- 105 MeV | 2 x 100k |

Arms (`analyze.py`): Geant4 half B against half A (the null: it measures the test's own false-failure rate);
`eslab` (NumPy, float64, on as many primaries as Geant4); `eslab_jax` in float64 and float32 (200k primaries). The
kernels use Geant4's production thresholds of each material (read from the couple table of the run). Observables: the
primary's fate (transmitted, reflected, absorbed), the transmitted primaries' energy loss, exit polar angle and
lateral displacement on the exit face, the reflected primaries' exit energy, and the primary's direct photons and
delta rays (per primary; spectra in the NumPy arm).

## Criteria (`criteria.json`)

Declared before any capture: two-sample chi2 for the distributions, z tests for fractions and yields, Bonferroni over
every test of an arm, pass if the corrected minimum p is at least 0.001; `al_0p1_e-_95` is reported without a pass
line (Geant4 option 4 changes its multiple-scattering model at 100 MeV, next to that case). Two amendments, both after
the pilot (2 x 10k events per case) and before any campaign output, are recorded in the file:
1. stopped and annihilated primaries are compared as one absorbed fraction (the recorder's end state cannot tell an
   in-flight annihilation from one at rest);
2. the lateral displacement of the five single-step thin-slab cases above 100 MeV has no pass line: there Geant4's
   WentzelVI caps the displacement at the safety, about 0 on a boundary-limited step, so most exits sit at r = 0
   exactly (83 % in 1 mm Al, 98 % in 0.1 mm Al).

## Result (2026-11-21, `52162d6`): the null passes, the kernel fails (tested, not closed)

| Arm | tests | Bonferroni min p | verdict |
|---|---|---|---|
| null (Geant4 B vs A) | 96 | 0.91 | pass |
| eslab, NumPy float64 | 85 | < 1e-300 | fail |
| eslab_jax, float64 | 67 | < 1e-300 | fail |
| eslab_jax, float32 | 67 | < 1e-300 | fail |

NumPy arm per case (chi2/ndf for distributions, Geant4 / kernel for fractions and yields; **bold**: p < 0.001):

| Case | transmitted | reflected | absorbed | loss | exit angle | displacement | reflected T | photons / primary | deltas / primary | photon spectrum | delta spectrum |
|---|---|---|---|---|---|---|---|---|---|---|---|
| al_0p1_e-_105 | 1/1 | - | - | 1.1 | **126.6** | 0.0 (no line) | - | 0.0139/0.01398 | 0.004375/0.004379 | 1.0 | 0.9 |
| al_0p1_e-_95 (no line) | 1/1 | - | 0/1e-06 | **4.1** | **17.4** | **122.4** | - | 0.01375/0.01363 | 0.004208/0.004394 | 1.0 | 1.0 |
| al_1_e+_105 | 0.9995/0.9995 | 2e-06/3e-06 | 0.000525/0.000487 | 0.9 | **32.9** | 209139 (no line) | - | 0.138/0.1378 | 0.04153/0.04127 | 0.6 | 1.0 |
| al_1_e-_105 | 1/1 | 6e-06/3e-06 | 1.8e-05/1.1e-05 | 0.9 | **32.4** | 206866 (no line) | - | 0.1377/0.1379 | 0.04322/0.04301 | 0.9 | 0.8 |
| al_1_e-_20 | 0.9999/0.9998 | 7.5e-05/8.4e-05 | 6.1e-05/6.8e-05 | 0.9 | **2.6** | **31.1** | - | 0.115/0.1146 | 0.04281/0.04292 | 0.7 | 0.9 |
| mylar_2_e-_105 | 1/1 | 2e-06/1e-06 | 5e-06/6e-06 | 1.2 | **32.1** | 245205 (no line) | - | 0.09534/0.09518 | 0.0677/0.06762 | 1.3 | 0.9 |
| mylar_2_e+_50 | 0.999/0.999 | 4e-06/4e-06 | 0.001038/0.001023 | **3.6** | **11.7** | **55.0** | - | 0.09255/0.09191 | 0.06405/0.0634 | 1.1 | 0.9 |
| w_0p35_e-_105 | 0.9987/0.9988 | 0.000392/0.000404 | 0.000866/0.000843 | 1.1 | **11.5** | 22397 (no line) | - | 0.8402/0.839 | 0.02491/0.02521 | 1.4 | 1.2 |
| w_0p35_e+_50 | 0.9963/0.9965 | **0.000599/0.000419** | 0.003081/0.003096 | 1.2 | **16.2** | **13.7** | **3.0** | 0.7317/0.7335 | 0.02052/0.02048 | 0.9 | 0.7 |
| w_3p5_e-_105 | 0.8154/0.8134 | 0.00321/0.00315 | 0.1814/0.1835 | 1.1 | 0.8 | **2.3** | 1.6 | 7.807/7.81 | 0.2495/0.2505 | 0.8 | 1.0 |

The device arms fail the same observables (exit angle; displacement where it keeps a line; positron loss in Mylar;
the reflected fraction of positrons in W), in float64 and float32 alike.

What agrees: the fates, the energy loss of transmitted primaries (chi2/ndf 0.9-1.2 in eight of ten cases), and the
direct photon and delta yields and spectra, at a million primaries per case.

What does not: the **multiple-scattering angle** in every thin slab, and the displacement where it is tested. The
differences are small in size and resolved only by the statistics (the pilot's 20k primaries passed most of them).
`diag.json` (`diag_quantiles.py`, 200k kernel primaries) shows how small they are:

| Case | exit angle q50 / q90 / q99, Geant4 A vs kernel [mrad] | displacement rms, Geant4 A / B / kernel [um] |
|---|---|---|
| al_0p1_e-_105 | 3.78 / 7.34 / 17.85 vs 3.61 / 7.57 / 17.81 | (98 % at 0 in Geant4) |
| al_0p1_e-_95 | 3.89 / 8.27 / 19.95 vs 3.98 / 8.34 / 19.63 | 0.53 / 0.50 / 0.50 |
| al_1_e-_105 | 13.71 / 26.66 / 62.36 vs 13.41 / 27.04 / 61.64 | (83 % at 0 in Geant4) |
| al_1_e-_20 | 68.8 / 139.7 / 301.2 vs 69.2 / 139.0 / 297.3 | 77.3 / 81.9 / 82.8 |
| w_0p35_e-_105 | 49.7 / 103.7 / 338.1 vs 49.0 / 104.9 / 335.2 | (42 % at 0 in Geant4) |
| w_0p35_e+_50 | 101.1 / 211.2 / 539.3 vs 99.9 / 205.2 / 492.5 | 36.2 / 35.6 / 30.9 |

For electrons the kernel's angular distribution is a little narrower in the core and a little wider above the median
(1-5 % in q50 and q90, within 2 % in q99). Positrons in tungsten scatter less in the kernel than in Geant4: q90 -3 %,
q99 -9 %, displacement -14 %, and 30 % fewer reflected. The kernel is a mixed class-II scheme on the single-scattering
cross section (`emsc`); Geant4 option 4 uses condensed-history models (Goudsmit-Saunderson below 100 MeV, WentzelVI
above) with their own step limitation, so percent-level shape differences at a million primaries are not surprising.
Whether they matter is a question for each user's tolerance. The positron tail in high-Z material is the clearest
candidate for a model fix (the e+ Mott factor used for hard collisions is the McKinley-Feshbach form).

## Running

```bash
sbatch ... run_g4.sbatch WORK SIMKERNELS SIMLIB G4_IMAGE ANA_IMAGE 1        # captures (0.02 = the pilot)
sbatch --array=0-29 ... run_analysis.sbatch WORK SIMKERNELS SIMLIB ANA_IMAGE
python validation/g4slab/analyze.py --combine WORK/rec_*.json --output record.json
```

Cost of this campaign: 0.9 core-h of Geant4 (184 shards on 8 cores, 36 GB of captures, not kept in the repository),
about 11 core-h of comparison (two device tasks were rerun after hitting a time limit set too low), 0.5 core-h of
diagnostics; with the pilot about 13 core-h. The records hold summary statistics only.
