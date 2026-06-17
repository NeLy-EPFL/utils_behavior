#!/usr/bin/env python3
"""CLI entry point for SLEAP track denoising.

Thin wrapper around :mod:`utils_behavior.sleap.clean_tracks` so it sits next to
the other ``scripts/sleap`` tools. Equivalent to::

    uv run python -m utils_behavior.sleap.clean_tracks ...
"""

from utils_behavior.sleap.clean_tracks import main

if __name__ == "__main__":
    main()
