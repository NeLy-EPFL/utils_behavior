import os
import pickle
import fafbseg
import navis
import matplotlib.pyplot as plt

brain_meshes_file = "brain_meshes.pkl"

if os.path.exists(brain_meshes_file):
    with open(brain_meshes_file, "rb") as f:
        brain_meshes = pickle.load(f)
else:
    neuropil_names = fafbseg.flywire.get_neuropil_volumes(None)
    brain_meshes = [
        fafbseg.flywire.get_neuropil_volumes(name) for name in neuropil_names
    ]
    with open(brain_meshes_file, "wb") as f:
        pickle.dump(brain_meshes, f)

# Print the list of brain meshes
print("Brain meshes loaded:")
for mesh in brain_meshes:
    if hasattr(mesh, "name"):
        print(mesh.name)
    elif isinstance(mesh, dict):
        print(list(mesh.keys()))

# --- Output directory ---
output_dir = "brain_region_pngs"
os.makedirs(output_dir, exist_ok=True)

for mesh in brain_meshes:
    region_name = mesh.name if hasattr(mesh, "name") else "unknown"
    print(f"Rendering {region_name}...")

    # Plot only this region
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

    plt.savefig(os.path.join(output_dir, f"{region_name}.png"), bbox_inches="tight")
    plt.close(fig)

print(f"Done! PNGs saved in {output_dir}/")
