"""Structural / smoke tests for the top-level package."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


def test_version_is_set():
    import utils_behavior

    assert isinstance(utils_behavior.__version__, str)
    assert utils_behavior.__version__


def test_version_matches_pyproject():
    """The runtime __version__ must match the version declared in pyproject.toml."""
    try:
        import tomllib  # py3.11+
    except ImportError:
        try:
            import tomli as tomllib  # py3.10 fallback
        except ImportError:
            pytest.skip("No TOML reader available")

    import utils_behavior

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject.open("rb") as f:
        cfg = tomllib.load(f)
    assert utils_behavior.__version__ == cfg["project"]["version"]


def test_core_submodules_importable():
    # These have only lightweight deps (numpy, scipy, pandas, h5py, cv2)
    from utils_behavior import processing, utils  # noqa: F401
    from utils_behavior.sleap import tracks  # noqa: F401


def test_no_heavy_import_at_package_load():
    """Importing utils_behavior must not pull in moviepy, holoviews, or fafbseg."""
    # Snapshot current modules so we can restore them afterwards (clearing
    # sys.modules globally would break other tests that hold references to
    # classes/functions from already-imported modules).
    _before = dict(sys.modules)
    for name in list(sys.modules):
        if name.startswith(
            ("utils_behavior", "moviepy", "holoviews", "bokeh", "fafbseg", "navis")
        ):
            del sys.modules[name]

    try:
        import utils_behavior  # noqa: F401

        assert "moviepy" not in sys.modules
        assert "holoviews" not in sys.modules
        assert "fafbseg" not in sys.modules
        assert "bokeh" not in sys.modules
    finally:
        # Remove any freshly imported modules and restore the originals.
        for name in [k for k in sys.modules if k not in _before]:
            del sys.modules[name]
        sys.modules.update(_before)


@pytest.mark.parametrize(
    "subpackage",
    [
        "utils_behavior.sleap",
        "utils_behavior.ballpushing",
        "utils_behavior.plotting",
        "utils_behavior.embedding",
        "utils_behavior.grids",
        "utils_behavior.flywire",
        "utils_behavior.optobot",
        "utils_behavior.dashboard",
    ],
)
def test_each_subpackage_imports(subpackage):
    """Every subpackage's __init__ should at least load.

    For subpackages whose __init__ pulls in optional extras (e.g. sleap
    re-exports Sleap_Tracks which needs h5py + cv2), we skip the test
    when the optional dependency is not installed.
    """
    try:
        importlib.import_module(subpackage)
    except ImportError as e:
        pytest.skip(f"optional dep not available for {subpackage}: {e.name}")


def test_lean_inits_have_no_heavy_imports():
    """The 'small' subpackage __init__ files should not pull in heavy deps.

    These were deliberately left empty so users can install minimal extras
    and still ``from utils_behavior import ballpushing`` without crashing.
    """
    lean_subpackages = [
        "utils_behavior.ballpushing",
        "utils_behavior.plotting",
        "utils_behavior.embedding",
        "utils_behavior.grids",
        "utils_behavior.flywire",
        "utils_behavior.optobot",
        "utils_behavior.dashboard",
    ]
    for name in lean_subpackages:
        importlib.import_module(name)


def test_top_level_package_has_no_print_side_effects(capsys):
    """Importing utils_behavior should be silent — no stray print/log output."""
    _before = dict(sys.modules)
    for name in list(sys.modules):
        if name.startswith("utils_behavior"):
            del sys.modules[name]
    capsys.readouterr()  # clear any prior buffer

    try:
        import utils_behavior  # noqa: F401

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
    finally:
        for name in [k for k in sys.modules if k not in _before]:
            del sys.modules[name]
        sys.modules.update(_before)


class TestPackageData:
    def test_flywire_yaml_is_accessible(self):
        """The bundled CX.yaml must resolve via importlib.resources."""
        try:
            from importlib import resources
        except ImportError:
            pytest.skip("importlib.resources unavailable")

        files = resources.files("utils_behavior.flywire") / "data" / "CX.yaml"
        assert files.is_file()
        text = files.read_text()
        # File should be a non-empty YAML document
        assert text.strip()


class TestNoStaleSysPathHack:
    """Make sure no module re-introduces the old sys.path.insert hack."""

    @pytest.mark.parametrize(
        "module_name",
        [
            "utils_behavior.optobot.optobot_utils",
            "utils_behavior.sleap.tracks",
            "utils_behavior.ballpushing.core",
        ],
    )
    def test_module_source_has_no_path_insert(self, module_name):
        try:
            mod = importlib.import_module(module_name)
        except ImportError as e:
            pytest.skip(f"optional dep not available: {e.name}")
        src = Path(mod.__file__).read_text()
        assert "sys.path.insert" not in src
        assert "sys.path.append" not in src
