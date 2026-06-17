#!/usr/bin/env python3
"""Directional velocity kinematics from SLEAP head/thorax/abdomen tracks.

For each tracked fly we compute, per frame:

- **centroid** = mean of the head, thorax and abdomen nodes (NaN-robust);
- **body axis** = unit vector from abdomen -> head; heading angle ``theta``;
- **speed** = magnitude of the centroid velocity;
- **forward velocity** = centroid velocity projected on the body axis
  (signed: >0 forward, <0 backward);
- **lateral velocity** = projection on the perpendicular axis (sideways slip);
- **rotational velocity** = time derivative of the (unwrapped) heading, in deg/s
  (signed, counter-clockwise positive in image coordinates).

Translational quantities are in px/s, or mm/s when a ``px_per_mm`` calibration is
given. The genotype/group label is parsed from the directory layout
``.../<genotype>/<data_dir>/<experiment_dir>/<video>.mp4`` (the genotype dir is
the 3rd parent of the video / h5 file).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .tracks import Sleap_Tracks

# Control genotypes (everything else parsed as PG<NN> is an experimental line).
CONTROL_GROUPS = ("B6VF", "B6XOGL7")


@dataclass
class GroupInfo:
    group: str
    is_control: bool
    raw_dir: str


def parse_group(path) -> GroupInfo:
    """Parse the genotype group from a video/h5 path.

    The genotype directory is the 3rd parent of the file
    (``.../<genotype>/<data_dir>/<experiment_dir>/<file>``). Rules:

    - ``PG_?(\\d+)...`` -> ``PG<NN>`` (``PG_14`` and ``PG14`` merge; suffixes like
      ``_OK`` / ``_2`` are dropped).
    - anything containing ``XOGL7`` -> ``B6XOGL7`` (pools ``B6VFXOGL7`` with
      ``B6XOGL7``).
    - else starting ``B6VF`` -> ``B6VF``.
    - otherwise the directory name is used verbatim.
    """
    path = Path(path)
    raw = path.parents[2].name if len(path.parents) >= 3 else path.parent.name
    up = raw.upper()

    m = re.match(r"^PG_?(\d+)", up)
    if m:
        return GroupInfo(f"PG{int(m.group(1))}", False, raw)
    if "XOGL7" in up:
        return GroupInfo("B6XOGL7", True, raw)
    if up.startswith("B6VF"):
        return GroupInfo("B6VF", True, raw)
    return GroupInfo(raw, False, raw)


def _smooth(values: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Savitzky-Golay smooth a 1-D series, interpolating NaN gaps first.

    Frames that were NaN in the input are restored to NaN afterwards so absent
    flies do not contribute spurious velocities.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    present = ~np.isnan(values)
    if present.sum() < 3:
        return values  # not enough to smooth/differentiate

    filled = pd.Series(values).interpolate(limit_direction="both").to_numpy()

    # window must be odd, > poly, and <= length.
    w = min(window, n)
    if w % 2 == 0:
        w -= 1
    if w <= poly:
        return filled
    return savgol_filter(filled, w, poly)


def compute_track_kinematics(
    dataset: pd.DataFrame,
    fps: float,
    centroid_nodes=("head", "thorax", "abdomen"),
    head_node: str = "head",
    abdomen_node: str = "abdomen",
    px_per_mm: float | None = None,
    smooth_window: int = 25,
    smooth_poly: int = 2,
) -> pd.DataFrame:
    """Compute per-frame kinematics for a single fly's track.

    Args:
        dataset: DataFrame with ``frame``, ``time`` and ``x_<node>``/``y_<node>``
            columns (a :class:`Sleap_Tracks.Object.dataset`).
        fps: frames per second.
        centroid_nodes: nodes averaged into the centroid.
        head_node, abdomen_node: nodes defining the body axis (abdomen -> head).
        px_per_mm: if given, translational outputs are in mm/s (else px/s).
        smooth_window, smooth_poly: Savitzky-Golay parameters (frames, order).

    Returns:
        DataFrame with frame, time, centroid_x/y, heading_deg, speed,
        forward_velocity, lateral_velocity, rotational_velocity, plus a
        ``units`` column ("mm/s" or "px/s").
    """
    if fps is None or fps <= 0:
        raise ValueError("fps must be a positive number to compute velocities.")

    # Resolve requested node names against the dataset case-insensitively
    # (SLEAP models may name nodes "Head"/"Thorax"/"Abdomen", etc.).
    available = {c[2:].lower(): c[2:] for c in dataset.columns if c.startswith("x_")}

    def resolve(name):
        key = name.lower()
        if key not in available:
            raise KeyError(
                f"node '{name}' not found; available nodes: {sorted(available.values())}"
            )
        return available[key]

    centroid_nodes = [resolve(n) for n in centroid_nodes]
    head_node = resolve(head_node)
    abdomen_node = resolve(abdomen_node)

    # Presence from the RAW data (before smoothing/interpolation): a frame counts
    # only if at least one centroid node was actually tracked there. Smoothing
    # interpolates across gaps, so this mask must be taken before that.
    raw_centroid = np.vstack([dataset[f"x_{n}"].to_numpy(dtype=float) for n in centroid_nodes])
    present = ~np.all(np.isnan(raw_centroid), axis=0)

    def xy(node):
        return (
            _smooth(dataset[f"x_{node}"].to_numpy(), smooth_window, smooth_poly),
            _smooth(dataset[f"y_{node}"].to_numpy(), smooth_window, smooth_poly),
        )

    # Centroid = mean of the requested nodes (NaN-robust).
    import warnings

    xs = np.vstack([xy(n)[0] for n in centroid_nodes])
    ys = np.vstack([xy(n)[1] for n in centroid_nodes])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN frames
        cx = np.nanmean(xs, axis=0)
        cy = np.nanmean(ys, axis=0)

    hx, hy = xy(head_node)
    ax_, ay = xy(abdomen_node)

    # Body axis: abdomen -> head (forward), as a unit vector.
    axis_x = hx - ax_
    axis_y = hy - ay
    axis_len = np.hypot(axis_x, axis_y)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux = axis_x / axis_len
        uy = axis_y / axis_len
    heading = np.arctan2(uy, ux)  # radians, image coords

    # Centroid velocity (px/frame -> px/s).
    vx = np.gradient(cx) * fps
    vy = np.gradient(cy) * fps

    # Projections: forward along axis, lateral along the perpendicular (-uy, ux).
    forward = vx * ux + vy * uy
    lateral = vx * (-uy) + vy * ux
    speed = np.hypot(vx, vy)

    # Rotational velocity: derivative of unwrapped heading (rad/s -> deg/s).
    heading_unwrapped = heading.copy()
    valid = ~np.isnan(heading_unwrapped)
    if valid.sum() > 1:
        heading_unwrapped[valid] = np.unwrap(heading_unwrapped[valid])
    rotational = np.degrees(np.gradient(heading_unwrapped) * fps)

    # Unit conversion for translational quantities.
    if px_per_mm:
        cx_o, cy_o = cx / px_per_mm, cy / px_per_mm
        forward, lateral, speed = (a / px_per_mm for a in (forward, lateral, speed))
        units = "mm/s"
    else:
        cx_o, cy_o = cx, cy
        units = "px/s"

    # Restrict outputs to frames where the fly was actually tracked.
    absent = ~present
    for arr in (cx_o, cy_o, forward, lateral, speed, rotational):
        arr[absent] = np.nan
    heading_out = np.degrees(heading)
    heading_out[absent] = np.nan

    return pd.DataFrame(
        {
            "frame": dataset["frame"].to_numpy(),
            "time": dataset["time"].to_numpy(),
            "dt": 1.0 / fps,
            "centroid_x": cx_o,
            "centroid_y": cy_o,
            "heading_deg": heading_out,
            "speed": speed,
            "forward_velocity": forward,
            "lateral_velocity": lateral,
            "rotational_velocity": rotational,
            "units": units,
        }
    )


def kinematics_for_h5(
    h5_path,
    px_per_mm: float | None = None,
    smooth_window: int = 25,
    smooth_poly: int = 2,
    object_type: str = "fly",
) -> pd.DataFrame:
    """Load a cleaned analysis ``.h5`` and return tidy per-frame kinematics.

    The returned DataFrame has one row per (track, frame) with ``video``,
    ``group``, ``is_control``, ``genotype_dir`` and ``track`` columns added.
    Returns an empty DataFrame if the file has no fps (video unreadable) or no
    tracks.
    """
    h5_path = Path(h5_path)
    st = Sleap_Tracks(h5_path, object_type=object_type, smoothed_tracks=False)
    if getattr(st, "fps", None) in (None, 0):
        print(f"  [skip] no fps for {h5_path.name} (video unreadable)")
        return pd.DataFrame()

    info = parse_group(h5_path)
    frames = []
    for obj in st.objects:
        k = compute_track_kinematics(
            obj.dataset,
            fps=st.fps,
            px_per_mm=px_per_mm,
            smooth_window=smooth_window,
            smooth_poly=smooth_poly,
        )
        track_id = obj.dataset["track_id"].iloc[0] if "track_id" in obj.dataset else None
        k.insert(0, "track", track_id)
        frames.append(k)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "video", h5_path.stem.replace("_tracked", ""))
    out.insert(1, "genotype_dir", info.raw_dir)
    out.insert(2, "group", info.group)
    out.insert(3, "is_control", info.is_control)
    out["source_h5"] = str(h5_path)
    return out


def summarize_per_fly(tidy: pd.DataFrame) -> pd.DataFrame:
    """Collapse the tidy per-frame table to one row per (video, group, track).

    Computes interpretable per-fly metrics: mean speed, mean forward speed while
    moving forward, mean backward speed while moving backward, fraction of time
    moving forward/backward, and mean absolute rotational velocity.
    """
    rows = []
    grp_cols = ["video", "genotype_dir", "group", "is_control", "track"]
    for keys, g in tidy.groupby(grp_cols, dropna=False):
        fwd = g["forward_velocity"]
        present = fwd.notna()
        n = max(int(present.sum()), 1)
        rows.append(
            {
                **dict(zip(grp_cols, keys)),
                "units": g["units"].iloc[0] if "units" in g else "",
                "n_frames": int(present.sum()),
                "mean_speed": g["speed"].mean(),
                "mean_forward_velocity": fwd.mean(),
                "mean_forward_speed": fwd[fwd > 0].mean(),
                "mean_backward_speed": fwd[fwd < 0].mean(),
                "frac_forward": float((fwd > 0).sum()) / n,
                "frac_backward": float((fwd < 0).sum()) / n,
                "mean_abs_rotational": g["rotational_velocity"].abs().mean(),
            }
        )
    return pd.DataFrame(rows)
