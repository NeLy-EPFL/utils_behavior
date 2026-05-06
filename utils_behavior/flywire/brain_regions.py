"""Render each FlyWire neuropil mesh as a transparent PNG.

Caches mesh objects to ``brain_meshes.pkl`` in the cwd so re-runs skip
the FlyWire fetch. Outputs go to ``./brain_region_pngs/`` by default.

Run as a script — importing this module no longer triggers any work.
"""

import os
import pickle
from pathlib import Path

import fafbseg
import matplotlib.pyplot as plt
import navis


def load_or_fetch_meshes(cache_path: Path = Path("brain_meshes.pkl")) -> list:
    """Load brain meshes from a pickle cache, or fetch + write the cache."""
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    neuropil_names = fafbseg.flywire.get_neuropil_volumes(None)
    brain_meshes = [
        fafbseg.flywire.get_neuropil_volumes(name) for name in neuropil_names
    ]
    with open(cache_path, "wb") as f:
        pickle.dump(brain_meshes, f)
    return brain_meshes


def render_brain_meshes(brain_meshes: list, output_dir: Path) -> None:
    """Render each mesh in `brain_meshes` as a transparent PNG into `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for mesh in brain_meshes:
        region_name = mesh.name if hasattr(mesh, "name") else "unknown"
        print(f"Rendering {region_name}...")

        fig, ax = navis.plot2d(
            [mesh],
            color="gray",
            alpha=1,
            method="3d",
            linewidth=10,
            view=("x", "-y"),
        )
        ax.axis("off")
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        plt.savefig(output_dir / f"{region_name}.png", bbox_inches="tight")
        plt.close(fig)

    print(f"Done! PNGs saved in {output_dir}/")


def main() -> None:
    brain_meshes = load_or_fetch_meshes()

    print("Brain meshes loaded:")
    for mesh in brain_meshes:
        if hasattr(mesh, "name"):
            print(mesh.name)
        elif isinstance(mesh, dict):
            print(list(mesh.keys()))

    render_brain_meshes(brain_meshes, Path("brain_region_pngs"))


if __name__ == "__main__":
    main()
