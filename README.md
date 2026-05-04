# utils_behavior

Utility functions for manipulating and analyzing behavioral recordings — video, tracking exports (SLEAP), derived datasets, and the plots / dashboards / embeddings we use to explore them.

The package is organised as a small core (`utils`, `processing`, `filter_registry`) plus a handful of themed submodules. Heavy dependencies (video I/O, interactive plotting, FlyWire, dimensionality reduction) live behind optional install extras so the base install stays light.

## Installation

### From source (editable)

```bash
git clone https://github.com/NeLy-EPFL/utils_behavior.git
cd utils_behavior
pip install -e .
```

### With optional extras

Pick only what you need:

```bash
pip install -e ".[video]"       # moviepy, pygame (Ballpushing / grids)
pip install -e ".[plots]"       # holoviews, bokeh, seaborn
pip install -e ".[embedding]"   # umap-learn, scikit-learn
pip install -e ".[flywire]"     # fafbseg, navis, flybrains
pip install -e ".[dashboard]"   # shiny
pip install -e ".[dev]"         # pytest, ruff

pip install -e ".[all]"         # everything except dev tools
```

Requires Python 3.9+.

## Package layout

```
utils_behavior/
├── utils.py              # environment / path helpers
├── processing.py         # filters, bootstrap CI, coordinate helpers
├── filter_registry.py    # named filter registry
├── sleap/                # SLEAP .h5 tracking wrappers
│   ├── tracks.py         #   Sleap_Tracks, generate_annotated_frame
│   └── tracker.py        #   SleapTracker CLI wrapper
├── ballpushing/          # ball-pushing experiment pipeline
├── plotting/             # Holoviews + Seaborn templates
├── embedding/            # UMAP / PCA / t-SNE behavior maps
├── grids/                # video grid builders
├── flywire/              # FlyWire connectomics helpers
├── optobot/              # optobot experiment utilities
└── dashboard/            # Shiny data explorer
```

## Quick start — SLEAP tracks

```python
from utils_behavior.sleap import Sleap_Tracks

tracks = Sleap_Tracks("predictions.h5", object_type="fly")
print(tracks.node_names)         # ['head', 'thorax', 'abdomen', ...]
print(tracks.fps)                # 30.0
print(tracks.objects[0].dataset) # pandas DataFrame: frame, time, x_*, y_*

# Access xy coordinates of a node directly
head_xy = tracks.objects[0].head  # list of (x, y) tuples per frame

# Restrict to a time window (seconds)
tracks.filter_data(time_range=(10.0, 30.0))
```

## Quick start — signal processing

```python
from utils_behavior import processing

# Smooth noisy tracking
smoothed = processing.savgol_lowpass_filter(raw_signal, window_length=51, polyorder=2)

# Bootstrap 95% CI for the mean
ci = processing.draw_bs_ci(values, n_reps=1000)
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The test suite covers `processing` (pure functions) and `sleap.tracks`
(SLEAP H5 wrapper, with `cv2.VideoCapture` mocked so no real videos are
needed). Fixtures build a minimal synthetic SLEAP `.h5` on the fly.

## Development

Code style is enforced by `ruff` (config in `pyproject.toml`):

```bash
ruff check utils_behavior
ruff format utils_behavior
```

Intra-package imports use the relative form, e.g.:

```python
from .. import processing as Processing
from ..sleap import tracks as Sleap_utils
```

## Repository layout

```
.
├── pyproject.toml        # build + metadata + extras
├── setup.py              # shim only; real config lives in pyproject.toml
├── utils_behavior/       # the installable package
├── tests/                # pytest suite
├── notebooks/            # exploratory notebooks (ballpushing, sleap, misc)
├── scripts/              # standalone runnable scripts (sleap, video, misc)
├── LICENSE
└── README.md
```

## License

MIT — see [LICENSE](LICENSE).
