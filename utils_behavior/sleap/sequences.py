#!/usr/bin/env python3
"""Optogenetic stimulation sequences for the Optobot Vglut experiment.

Two protocols are used, both alternating off/on periods (durations in ms):

- ``short`` — videos under the ``PG`` directory. Five 60 s ON pulses separated
  by 15 s rests, bracketed by 0.5 s off. Total 361 s.
- ``long`` — videos under ``PG_YYYYMMDD_Vglut_B6VF_1min``. Identical for the
  first five ON pulses, then (instead of ending) a 300 s rest followed by two
  more blocks of five 60 s pulses, the blocks separated by a 300 s rest.

The two protocols are bit-identical for the first 360.5 s (the five shared
pulses), so flies from both can be pooled over that **shared window**; the
extended pulses (6-15) exist only in the ``long`` protocol.
"""

from __future__ import annotations

import re
from pathlib import Path

# (state, duration_ms) timelines.
_SHARED = [
    ("off", 500),
    ("on", 60000), ("off", 15000),
    ("on", 60000), ("off", 15000),
    ("on", 60000), ("off", 15000),
    ("on", 60000), ("off", 15000),
    ("on", 60000),
]
_LONG_BLOCK = [
    ("off", 300000),
    ("on", 60000), ("off", 15000),
    ("on", 60000), ("off", 15000),
    ("on", 60000), ("off", 15000),
    ("on", 60000), ("off", 15000),
    ("on", 60000),
]

SEQUENCES_MS = {
    "short": _SHARED + [("off", 500)],
    "long": _SHARED + _LONG_BLOCK + _LONG_BLOCK,
}


def intervals(name: str):
    """Return the protocol as a list of dicts with seconds and ON index.

    Each entry: ``{state, start_s, end_s, on_index}`` where ``on_index`` is the
    1-based pulse number for ON periods and ``None`` for OFF.
    """
    out = []
    t = 0.0
    on_count = 0
    for state, ms in SEQUENCES_MS[name]:
        dur = ms / 1000.0
        on_idx = None
        if state == "on":
            on_count += 1
            on_idx = on_count
        out.append({"state": state, "start_s": t, "end_s": t + dur, "on_index": on_idx})
        t += dur
    return out


def on_intervals(name: str):
    """List of ``(start_s, end_s)`` for every ON pulse in the protocol."""
    return [(iv["start_s"], iv["end_s"]) for iv in intervals(name) if iv["state"] == "on"]


def total_duration(name: str) -> float:
    """Total protocol duration in seconds."""
    return sum(ms for _, ms in SEQUENCES_MS[name]) / 1000.0


def shared_window_end(a: str = "short", b: str = "long") -> float:
    """Seconds up to which protocols ``a`` and ``b`` are identical."""
    t = 0.0
    for (sa, ma), (sb, mb) in zip(SEQUENCES_MS[a], SEQUENCES_MS[b]):
        if sa != sb or ma != mb:
            break
        t += ma / 1000.0
    return t


def detect_sequence(path) -> str | None:
    """Infer the protocol from a video/h5 path.

    The session directory is the 4th parent
    (``.../<session>/<genotype>/<data_dir>/<experiment_dir>/<file>``):
    ``PG`` -> ``short``; ``PG_YYYYMMDD_...`` -> ``long``.
    """
    path = Path(path)
    if len(path.parents) < 4:
        return None
    session = path.parents[3].name
    if session.upper() == "PG":
        return "short"
    if re.match(r"PG_\d{8}", session):
        return "long"
    return None


def label_times(times_s, name: str):
    """Map an array of times (s) to (stim_on: bool, on_index: float).

    ``on_index`` is the 1-based pulse number during ON frames, NaN otherwise.
    """
    import numpy as np

    times = np.asarray(times_s, dtype=float)
    stim_on = np.zeros(times.shape, dtype=bool)
    on_index = np.full(times.shape, np.nan)
    for iv in intervals(name):
        mask = (times >= iv["start_s"]) & (times < iv["end_s"])
        if iv["state"] == "on":
            stim_on[mask] = True
            on_index[mask] = iv["on_index"]
    return stim_on, on_index
