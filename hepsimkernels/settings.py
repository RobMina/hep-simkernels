"""Environment settings of the kernels.

Each setting is the environment variable ``<PREFIX>_<KEY>``: DEVICE_FLOAT (precision), DEVICE_DETERMINISTIC
(determinism), SLAB_KERNEL (slab_jax), F64_AUDIT and F64_AUDIT_DIR (precision). The prefix comes from
``HEPSIMKERNELS_ENV_PREFIX`` (default ``HEPSIMK``) and is read once, when this module is first imported. An experiment
that keeps its own variable names sets the prefix before it imports any kernel (mu2e-fastsim sets ``MU2EFS``, so its
jobs keep MU2EFS_DEVICE_FLOAT and the rest). Standard library only.
"""
import os

PREFIX = (os.environ.get("HEPSIMKERNELS_ENV_PREFIX") or "HEPSIMK").strip()


def name(key):
    """The environment variable of a setting, e.g. name("DEVICE_FLOAT") -> "HEPSIMK_DEVICE_FLOAT"."""
    return f"{PREFIX}_{key}"


def get(key, default=None):
    return os.environ.get(name(key), default)
