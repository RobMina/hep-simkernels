"""Material constants of the kernels, matched to Geant4 11.3.2.

Sources:
  * composition, density and mean excitation energy I: Geant4's NIST material database (G4NistMaterialBuilder);
  * Sternheimer density-effect parameters: G4DensityEffectData.cc (NIST materials by name);
  * Birks constant: G4EmSaturation::InitialiseG4materials (only G4_POLYSTYRENE has one: 0.07943 mm/MeV);
  * Urban fluctuation parameters: G4IonisParamMat::ComputeFluctModel (from Zeff and I).
Electron densities are computed here from the composition (electrons per volume = rho N_A sum_i w_i Z_i / A_i).

The NIST materials below are registered by name in BY_NAME, which the kernels look up with get(). An experiment adds
the materials of its own geometry with register() (composition, density and I as in its GDML) and per-region
production cuts with add_cuts(); mu2e-fastsim registers the Mu2e GDML materials this way.
"""
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

N_A = 6.02214076e23
# Z, A [g/mol] of the elements of the materials here and of the experiment materials registered so far
ELEMENTS = {"H": (1, 1.00794), "C": (6, 12.0107), "N": (7, 14.0067), "O": (8, 15.9994), "Na": (11, 22.98977),
            "Al": (13, 26.981538), "Si": (14, 28.0855), "Ar": (18, 39.948), "K": (19, 39.0983), "Ca": (20, 40.078),
            "Fe": (26, 55.845), "Mn": (25, 54.938049), "Cu": (29, 63.546), "W": (74, 183.84), "Au": (79, 196.967), "Mg": (12, 24.305),
            "Pb": (82, 207.2), "Sn": (50, 118.71), "Ni": (28, 58.6934), "Zn": (30, 65.38), "Sb": (51, 121.76),
            "Cl": (17, 35.453),
            "B": (5, 10.811), "Ti": (22, 47.867), "F": (9, 18.9984032),
            "Ag": (47, 107.8682), "Cr": (24, 51.9961), "Cs": (55, 132.905), "I": (53, 126.904), "Lu": (71, 174.967),
            "Y": (39, 88.9058), "Ce": (58, 140.116)}


@dataclass(frozen=True)
class Material:
    name: str
    rho: float                  # g/cm3
    I_eV: float                 # mean excitation energy
    fractions: dict             # element -> mass fraction
    sternheimer: tuple = ()     # (C, X0, X1, a, m, delta0) of G4DensityEffectData, empty = no table
    birks_mm_per_MeV: float = 0.0
    cuts_e_MeV: dict = field(default_factory=dict)   # production threshold for delta electrons, by region tag
    state: str = "solid"        # "gas": Geant4's gas branch of the Sternheimer parametrisation (at STP)

    @cached_property
    def zeff(self):
        return sum(w * ELEMENTS[e][0] for e, w in self.fractions.items())

    @cached_property
    def electron_density_per_mm3(self):
        z_over_a = sum(w * ELEMENTS[e][0] / ELEMENTS[e][1] for e, w in self.fractions.items())
        return self.rho * N_A * z_over_a * 1e-3  # cm-3 -> mm-3

    def density_params(self):
        return self._density_params

    @cached_property
    def _density_params(self):
        """(C, X0, X1, a, m, delta0): the NIST table when given, else Sternheimer 1971
        (G4IonisParamMat::ComputeDensityEffectParameters): solid branch, or the gas branch for state "gas" (no
        pressure/temperature correction: the gas is taken at STP)."""
        if self.sternheimer:
            return self.sternheimer
        hbarc_mm = 197.3269804e-12            # MeV mm
        re_mm = 2.8179403262e-12
        e_plasma = np.sqrt(4 * np.pi * hbarc_mm ** 2 * re_mm * self.electron_density_per_mm3)   # MeV
        C = 1 + 2 * np.log(self.I_eV * 1e-6 / e_plasma)
        if self.state == "gas":
            x1 = 4.0
            for c_hi, x0 in ((10.0, 1.6), (10.5, 1.7), (11.0, 1.8), (11.5, 1.9), (12.25, 2.0)):
                if C <= c_hi:
                    break
            else:
                x0, x1 = (2.0, 5.0) if C <= 13.804 else (0.326 * C - 2.5, 5.0)
            a = 2 * np.log(10) * (C / (2 * np.log(10)) - x0) / (x1 - x0) ** 3.0
            return (C, x0, x1, a, 3.0, 0.0)
        icase = 0 if self.I_eV < 100.0 else 1
        x0 = 0.2 if C < (3.681, 5.215)[icase] else 0.326 * C - (1.0, 1.5)[icase]
        x1 = (2.0, 3.0)[icase]
        m = 3.0
        a = 2 * np.log(10) * (C / (2 * np.log(10)) - x0) / (x1 - x0) ** m
        return (C, x0, x1, a, m, 0.0)

    def fluct_params(self):
        return self._fluct_params

    @cached_property
    def _fluct_params(self):
        """G4IonisParamMat::ComputeFluctModel: (f1, f2, e1, e2) in MeV, plus e0 = 10 eV and log I."""
        I = self.I_eV * 1e-6
        z = self.zeff
        if z > 2.1:
            f2 = 2.0 / z
            f1 = 1.0 - f2
            e2 = 10.0 * z * z * 1e-6
            loge1 = (np.log(I) - f2 * np.log(e2)) / f1
        elif z > 1.1:
            f2, f1, e2, loge1 = 0.0, 1.0, 40e-6, np.log(I)
        else:
            f2, f1, e2, loge1 = 0.0, 1.0, 10e-6, np.log(I)
        return {"f1": f1, "f2": f2, "e1": float(np.exp(loge1)), "e2": e2, "e0": 10e-6, "I": I}


POLYSTYRENE = Material("G4_POLYSTYRENE", 1.06, 68.7, {"C": 0.922577329294378, "H": 0.0774226707056217},
                       sternheimer=(3.2999, 0.1647, 2.5031, 0.16454, 3.2224, 0.0), birks_mm_per_MeV=0.07943)
ALUMINIUM = Material("G4_Al", 2.699, 166.0, {"Al": 1.0}, sternheimer=(4.2395, 0.1708, 3.0127, 0.08024, 3.6345, 0.12))
AIR = Material("G4_AIR", 0.00120479, 85.7,
               {"C": 0.000124000124000124, "N": 0.755267755267755, "O": 0.231781231781232, "Ar": 0.0128270128270128},
               sternheimer=(10.5961, 1.7418, 4.2759, 0.10914, 3.3994, 0.0))
COPPER = Material("G4_Cu", 8.96, 322.0, {"Cu": 1.0}, sternheimer=(4.4190, -0.0254, 3.2792, 0.14339, 2.9044, 0.08))
SILICON = Material("G4_Si", 2.33, 173.0, {"Si": 1.0}, sternheimer=(4.4351, 0.2014, 2.8715, 0.14921, 3.2546, 0.14))
TUNGSTEN = Material("G4_W", 19.30, 727.0, {"W": 1.0}, sternheimer=(5.4059, 0.2167, 3.4960, 0.15509, 2.8447, 0.14))
LEAD = Material("G4_Pb", 11.35, 823.0, {"Pb": 1.0}, sternheimer=(6.2018, 0.3776, 3.8073, 0.09359, 3.1608, 0.14))   # slab test (NIST)
GOLD = Material("G4_Au", 19.32, 790.0, {"Au": 1.0}, sternheimer=(5.5747, 0.2021, 3.6979, 0.09756, 3.1101, 0.14))
MYLAR = Material("G4_MYLAR", 1.40, 78.7, {"C": 0.625010832340889, "H": 0.0419607170679432, "O": 0.333028450591168})
POLYETHYLENE = Material("G4_POLYETHYLENE", 0.94, 57.4, {"C": 0.856, "H": 0.144})
SILICA = Material("G4_SILICON_DIOXIDE", 2.32, 139.2, {"Si": 0.4674, "O": 0.5326})
CESIUM_IODIDE = Material("G4_CESIUM_IODIDE", 4.51, 553.1, {"Cs": 0.511548868591927, "I": 0.488451131408073},
                         sternheimer=(6.2807, 0.0395, 3.3353, 0.25381, 2.6657, 0.0))

BY_NAME = {m.name: m for m in (POLYSTYRENE, ALUMINIUM, AIR, COPPER, SILICON, TUNGSTEN, LEAD, GOLD, MYLAR, POLYETHYLENE, SILICA, CESIUM_IODIDE)}


def register(*materials):
    """Add materials (by their GDML name) to BY_NAME; a name already present must be the same material."""
    for m in materials:
        if m.name in BY_NAME and BY_NAME[m.name] is not m and BY_NAME[m.name] != m:
            raise ValueError(f"material {m.name!r} is registered already with other constants")
        unknown = set(m.fractions) - set(ELEMENTS)
        if unknown:
            raise KeyError(f"material {m.name!r}: elements {sorted(unknown)} have no Z, A (add them to ELEMENTS)")
        BY_NAME[m.name] = m


def add_cuts(name, **cuts_e_MeV):
    """Delta-electron production thresholds [MeV] of a material by region tag, e.g. add_cuts("G4_POLYSTYRENE",
    crv=0.0872): the cuts_e_MeV a kernel reads for that region."""
    get(name).cuts_e_MeV.update(cuts_e_MeV)


def get(name):
    """Material by GDML name; a name without constants is an error (no silent default)."""
    try:
        return BY_NAME[name]
    except KeyError:
        raise KeyError(f"material {name!r} has no constants: register it (hepsimkernels.materials.register)") from None
