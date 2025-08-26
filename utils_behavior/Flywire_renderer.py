import os
import sys
import argparse
import yaml
import matplotlib.pyplot as plt
import navis
import flybrains
from fafbseg import flywire


def load_neuron_ids(neuron_arg):
    """Load neuron IDs from YAML file or comma-separated list."""
    if neuron_arg.endswith(".yaml") or neuron_arg.endswith(".yml"):
        with open(neuron_arg, "r") as f:
            neuron_ids = yaml.safe_load(f)
    else:
        neuron_ids = [int(nid) for nid in neuron_arg.split(",")]
    return neuron_ids


def render_neurons(
    neuron_ids, color="red", formats=["png"], out_prefix="flywire_neurons"
):
    # Set FlyWire dataset
    flywire.set_default_dataset("public")
    # Fetch neuron meshes
    meshes = flywire.get_mesh_neuron(neuron_ids, dataset="flat_783", lod=3)
    meshes = navis.downsample_neuron(meshes, 20)

    kwargs = {
        "method": "2d",
        "view": ("x", "-y"),
        "rasterize": True,
    }

    fig, ax = navis.plot2d(flybrains.FLYWIRE, color=(0.7, 0.7, 0.7, 0.05), **kwargs)
    navis.plot2d(meshes, color=color, ax=ax, **kwargs)
    ax.axis("off")
    ax.grid(False)

    for fmt in formats:
        plt.savefig(
            f"{out_prefix}.{fmt}",
            format=fmt,
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
        )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render FlyWire neurons on brain template."
    )
    parser.add_argument(
        "--neurons",
        required=True,
        help="YAML file with neuron IDs or comma-separated list of IDs",
    )
    parser.add_argument(
        "--color", default="red", help="Color for neuron traces (default: red)"
    )
    parser.add_argument(
        "--formats",
        default="png",
        help="Comma-separated list of formats (e.g., png,svg,pdf)",
    )
    parser.add_argument(
        "--out_prefix", default="flywire_neurons", help="Prefix for output files"
    )
    args = parser.parse_args()

    neuron_ids = load_neuron_ids(args.neurons)
    formats = [fmt.strip() for fmt in args.formats.split(",")]

    render_neurons(
        neuron_ids=neuron_ids,
        color=args.color,
        formats=formats,
        out_prefix=args.out_prefix,
    )
