"""SLEAP tracking utilities.

Work with ``.h5`` exports from SLEAP tracking data and run inference with
the SLEAP CLI wrapper.

Key exports
-----------
- :class:`Sleap_Tracks` — wrapper around a SLEAP ``.h5`` tracking file.
- :func:`generate_annotated_frame` — render a single frame with tracks overlaid.
- :class:`SleapTracker` — inference wrapper around the SLEAP CLI.
"""

from .tracks import Sleap_Tracks, generate_annotated_frame

__all__ = ["Sleap_Tracks", "generate_annotated_frame"]

# ``SleapTracker`` is intentionally not imported at package-load time because
# it expects SLEAP to be available on the PATH. Import it explicitly:
#     from utils_behavior.sleap.tracker import SleapTracker
