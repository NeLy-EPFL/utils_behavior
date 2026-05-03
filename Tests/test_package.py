"""Smoke tests for the top-level package."""
from __future__ import annotations


def test_version_is_set():
    import utils_behavior

    assert isinstance(utils_behavior.__version__, str)
    assert utils_behavior.__version__


def test_core_submodules_importable():
    # These have only lightweight deps (numpy, scipy, pandas, h5py, cv2)
    from utils_behavior import processing, utils  # noqa: F401
    from utils_behavior.sleap import tracks  # noqa: F401


def test_no_heavy_import_at_package_load():
    """Importing utils_behavior must not pull in moviepy, holoviews, or fafbseg."""
    import sys

    # Clear any prior imports to measure a fresh load
    for name in list(sys.modules):
        if name.startswith(("moviepy", "holoviews", "bokeh", "fafbseg", "navis")):
            del sys.modules[name]

    import utils_behavior  # noqa: F401

    assert "moviepy" not in sys.modules
    assert "holoviews" not in sys.modules
    assert "fafbseg" not in sys.modules
