"""hep-simkernels: Geant4-matched particle-matter kernels for fast simulation.

NumPy references with JAX twins (float32 or float64, hepsimkernels.precision): energy loss with Urban fluctuations
(eloss, eloss_jax), electron ionisation (eioni), multiple scattering (emsc), bremsstrahlung (ebrem), annihilation
(eannihil), muon radiative losses (muradiative, muradiative_jax), electron slab propagation (eslab, eslab_jax,
slab_jax), straight-line transport through GDML geometry (world, transport, crossings, geometry.annulus_jax),
helix stepping in field maps (ehelix), and the materials they use (materials).

Importing the package applies the GPU determinism setting (determinism.apply(), before JAX creates a backend).
Environment variable names follow hepsimkernels.settings (set HEPSIMKERNELS_ENV_PREFIX before importing).
"""
from . import determinism as _determinism

_determinism.apply()

__version__ = "0.1.0"
