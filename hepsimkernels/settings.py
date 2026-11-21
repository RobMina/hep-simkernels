"""Environment settings of the kernels.

Each setting is the environment variable ``<PREFIX>_<KEY>``: DEVICE_FLOAT (precision), DEVICE_DETERMINISTIC
(determinism), SLAB_KERNEL (slab_jax), F64_AUDIT and F64_AUDIT_DIR (precision). The prefix comes from
``HEPSIMKERNELS_ENV_PREFIX`` (default ``HEPSIMK``) and is read once, when this module is first imported. An experiment
that keeps its own variable names sets the prefix before it imports any kernel (mu2e-fastsim sets ``MU2EFS``, so its
jobs keep MU2EFS_DEVICE_FLOAT and the rest).

Experiment hooks live here too, so an experiment installs them without importing NumPy or fixing the device float:
MATERIAL_MODULES (modules imported when hepsimkernels.materials is first imported; they register the experiment's
materials), MAP_SCALE (a callable header_path -> factor that ehelix.read_g4bl applies to a field map) and
PROVENANCE_HOOKS (callables whose dicts precision.provenance() adds) and FALLBACK_MATERIAL (the registered material
name that transport.world_loss uses for a world piece whose material is not registered; None, the default, makes an
unregistered material an error). Standard library only.
"""
import os

PREFIX = (os.environ.get("HEPSIMKERNELS_ENV_PREFIX") or "HEPSIMK").strip()


MATERIAL_MODULES = []
MAP_SCALE = None
PROVENANCE_HOOKS = []
FALLBACK_MATERIAL = None


def name(key):
    """The environment variable of a setting, e.g. name("DEVICE_FLOAT") -> "HEPSIMK_DEVICE_FLOAT"."""
    return f"{PREFIX}_{key}"


def get(key, default=None):
    return os.environ.get(name(key), default)
