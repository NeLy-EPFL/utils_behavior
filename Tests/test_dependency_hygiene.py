"""Static checks that catch stale or missing dependency declarations."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_ROOT = REPO_ROOT / "utils_behavior"


def _load_pyproject():
    try:
        import tomllib  # py3.11+
    except ImportError:
        try:
            import tomli as tomllib  # py3.10 fallback
        except ImportError:
            pytest.skip("No TOML reader available")
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


# Map distribution name (as listed in pyproject) -> import name (in source)
DIST_TO_IMPORT = {
    "opencv-python": "cv2",
    "opencv_contrib_python": "cv2",
    "scikit-learn": "sklearn",
    "umap-learn": "umap",
    "pyyaml": "yaml",
    "stdlib_list": "stdlib_list",
    "icecream": "icecream",
    "fafbseg": "fafbseg",
    "navis": "navis",
    "flybrains": "flybrains",
    "moviepy": "moviepy",
    "pygame": "pygame",
    "holoviews": "holoviews",
    "bokeh": "bokeh",
    "seaborn": "seaborn",
    "shiny": "shiny",
}


def _scan_imports() -> set[str]:
    """Return the set of top-level imported names across the package source.

    Uses the AST so docstrings, comments, and quoted code blocks aren't
    miscounted as real imports. Captures both module-level and lazy imports
    inside function bodies.
    """
    import ast

    found: set[str] = set()
    for py in PKG_ROOT.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(errors="ignore"), filename=str(py))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                # Relative imports (level > 0) are intra-package — skip
                if node.level == 0 and node.module:
                    found.add(node.module.split(".")[0])
    found.discard("utils_behavior")
    found.discard("")
    return found


@pytest.fixture(scope="module")
def imports() -> set[str]:
    return _scan_imports()


@pytest.fixture(scope="module")
def all_declared_deps() -> list[str]:
    cfg = _load_pyproject()
    declared: list[str] = []
    for spec in cfg["project"]["dependencies"]:
        declared.append(re.split(r"[<>=!~ ]", spec, maxsplit=1)[0])
    for _grp, specs in cfg["project"]["optional-dependencies"].items():
        for spec in specs:
            # Skip self-references like "utils_behavior[plots,...]"
            if spec.startswith("utils_behavior"):
                continue
            declared.append(re.split(r"[<>=!~ ]", spec, maxsplit=1)[0])
    return declared


class TestDeclaredDepsAreUsed:
    # Tooling/dev deps are never imported by the package source itself
    TOOLING_ONLY = {"pytest", "pytest-cov", "ruff"}

    def test_every_declared_dep_is_imported_somewhere(
        self, all_declared_deps, imports
    ):
        """Every dep listed in pyproject.toml should be imported by at least one module.

        Catches stale deps left over after refactors.
        """
        unused: list[str] = []
        for dep in set(all_declared_deps):
            if dep in self.TOOLING_ONLY:
                continue
            import_name = DIST_TO_IMPORT.get(dep, dep)
            if import_name not in imports:
                unused.append(dep)
        assert not unused, f"Declared but never imported: {sorted(unused)}"


STDLIB = {
    "argparse",
    "ast",
    "base64",
    "collections",
    "concurrent",
    "contextlib",
    "copy",
    "csv",
    "ctypes",
    "dataclasses",
    "datetime",
    "enum",
    "functools",
    "gc",
    "glob",
    "hashlib",
    "importlib",
    "io",
    "itertools",
    "json",
    "logging",
    "math",
    "multiprocessing",
    "operator",
    "os",
    "pathlib",
    "pickle",
    "platform",
    "queue",
    "random",
    "re",
    "shutil",
    "signal",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
    "time",
    "traceback",
    "typing",
    "unittest",
    "uuid",
    "warnings",
    "xml",
    "zipfile",
}


class TestNoUndeclaredImports:
    """Every third-party import in the package source must be declared in
    ``pyproject.toml`` (in core deps or any optional extra)."""

    def test_no_third_party_imports_outside_pyproject(
        self, imports, all_declared_deps
    ):
        declared_import_names = {
            DIST_TO_IMPORT.get(d, d) for d in all_declared_deps
        }
        permitted = STDLIB | declared_import_names

        unknown = sorted(name for name in imports if name not in permitted)
        assert not unknown, (
            f"Imports not declared anywhere in pyproject.toml: {unknown}. "
            f"Add them to [project.dependencies] or an extra."
        )


class TestPackageDataDeclared:
    def test_flywire_yaml_listed_in_package_data(self):
        cfg = _load_pyproject()
        pkg_data = cfg["tool"]["setuptools"]["package-data"]
        assert "utils_behavior.flywire" in pkg_data
        patterns = pkg_data["utils_behavior.flywire"]
        assert any("yaml" in p or "yml" in p for p in patterns)

    def test_yaml_file_actually_exists_in_tree(self):
        yaml = PKG_ROOT / "flywire" / "data" / "CX.yaml"
        assert yaml.is_file()
