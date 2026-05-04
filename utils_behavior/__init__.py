"""utils_behavior — utilities for manipulating and analyzing behavior recordings.

Submodules are imported lazily (i.e. you import what you need) to avoid
pulling in heavy optional dependencies (holoviews, moviepy, fafbseg, ...)
at package import time.

Examples
--------
>>> from utils_behavior.sleap import Sleap_Tracks
>>> from utils_behavior import processing
>>> processing.butter_lowpass_filter(data, cutoff=0.1, order=4)
"""

from ._version import __version__

__all__ = ["__version__"]
