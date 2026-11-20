"""Test hooks: float64 device kernels (the suites check exact device-versus-NumPy parity, which holds in float64 only;
float32 has its own subprocess tests), no GPU memory preallocation, JAX caches released per module,
@pytest.mark.slow / weekly opt-in (--runslow, --runweekly), @pytest.mark.pr_params; and the example experiment's
materials registered (example_materials)."""
import gc
import os
import sys

import pytest

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("HEPSIMK_DEVICE_FLOAT", "float64")
if "hepsimkernels.precision" in sys.modules and sys.modules["hepsimkernels.precision"].NAME != os.environ["HEPSIMK_DEVICE_FLOAT"]:
    raise RuntimeError("hepsimkernels.precision was imported before the test hooks")

sys.path.insert(0, os.path.dirname(__file__))
import example_materials  # noqa: E402,F401  (registers the example experiment's materials)


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="also run @pytest.mark.slow statistical validations (large samples, GB-scale memory)")
    parser.addoption("--runweekly", action="store_true", default=False,
                     help="also run @pytest.mark.weekly validations (tens of minutes; not selected by --runslow)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: large-sample statistical validation; skipped unless --runslow")
    config.addinivalue_line("markers", "pr_params(*ids): parameter ids kept per PR; the other ids are slow")
    config.addinivalue_line("markers", "weekly: validation too long for the nightly run; skipped unless --runweekly")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    for item in items:
        keep = item.get_closest_marker("pr_params")
        callspec = getattr(item, "callspec", None)
        if keep is not None and callspec is not None and callspec.id not in keep.args:
            item.add_marker(pytest.mark.slow)
    selected = config.getoption("-m") or ""
    runslow = config.getoption("--runslow") or "slow" in selected
    runweekly = config.getoption("--runweekly") or "weekly" in selected
    skip_slow = pytest.mark.skip(reason="slow validation: run with --runslow")
    skip_weekly = pytest.mark.skip(reason="weekly validation: run with --runweekly")
    for item in items:
        if "weekly" in item.keywords:
            if not runweekly:
                item.add_marker(skip_weekly)
        elif "slow" in item.keywords and not runslow:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True, scope="module")
def _release_jax_caches():
    """Drop compiled programs and their buffers when a test module finishes."""
    yield
    jax = sys.modules.get("jax")
    if jax is not None:
        jax.clear_caches()
    gc.collect()
