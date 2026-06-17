#!/usr/bin/env python3
"""Directional velocity analysis for the Optobot Vglut experiment.

Loads cleaned SLEAP analysis ``.h5`` files, computes per-fly directional velocity
(forward/backward, lateral, rotational, speed) from the head/thorax/abdomen
keypoints, groups flies by genotype parsed from the directory layout, and writes:

- ``directional_velocity_frames.feather`` — tidy per-frame table.
- ``directional_velocity_per_fly.feather`` — one row per fly with summary metrics.
- ``summary_<metric>.png`` — per-group box + strip plots (lines vs controls).
- ``timeseries_<metric>.png`` — per-group mean +/- sem over time, with the pooled
  control mean as a gray reference and optional opto-stimulus shading.

Run with::

    uv run python scripts/sleap/directional_velocity.py <yaml|dir|h5...> \\
        --px-per-mm 27.5 --output-dir /path/to/figs

Inputs may be the tracker's video-list YAML (each video -> its ``*_tracked.h5``),
a directory (searched recursively for ``*_tracked.h5``), or explicit ``.h5`` paths.
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / screen session
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_behavior.sleap import sequences
from utils_behavior.sleap.kinematics import (
    CONTROL_GROUPS,
    kinematics_for_h5,
    summarize_per_fly,
)

METRICS = {
    "speed": "Speed",
    "forward_velocity": "Forward velocity (+fwd / -back)",
    "rotational_velocity": "Rotational velocity",
}


# --------------------------------------------------------------------------- IO


def collect_h5_files(paths, pattern="*_tracked.h5", recursive=True):
    """Resolve input paths (YAML / directory / file) to a list of .h5 files."""
    files = []
    for p in paths:
        path = Path(p)
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            with open(path) as fh:
                data = yaml.safe_load(fh) or {}
            for v in data.get("videos", []):
                vp = Path(v)
                h5 = vp.with_name(f"{vp.stem}_tracked.h5")
                if h5.exists():
                    files.append(h5)
        elif path.is_dir():
            globber = path.rglob if recursive else path.glob
            files.extend(sorted(globber(pattern)))
        else:
            files.append(path)
    return files


def order_groups(groups):
    """Sort groups: PG lines numerically first, then controls, others last."""
    def key(g):
        if g.startswith("PG") and g[2:].isdigit():
            return (0, int(g[2:]), g)
        if g in CONTROL_GROUPS:
            return (2, 0, g)
        return (1, 0, g)

    return sorted(groups, key=key)


# ---------------------------------------------------------------------- plotting


def plot_summary(per_fly, metric, label, units, out_path):
    """Per-group box + jittered strip of a per-fly metric."""
    col = {
        "speed": "mean_speed",
        "forward_velocity": "mean_forward_velocity",
        "rotational_velocity": "mean_abs_rotational",
    }[metric]
    groups = order_groups(per_fly["group"].unique())
    data = [per_fly.loc[per_fly["group"] == g, col].dropna().to_numpy() for g in groups]

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 0.45), 5))
    ax.boxplot(data, showfliers=False, medianprops=dict(color="black"))
    rng = np.random.default_rng(0)
    for i, (g, vals) in enumerate(zip(groups, data), start=1):
        if not len(vals):
            continue
        x = i + rng.uniform(-0.15, 0.15, size=len(vals))
        color = "tab:red" if g in CONTROL_GROUPS else "tab:blue"
        ax.scatter(x, vals, s=14, alpha=0.6, color=color, edgecolors="none")
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(groups, rotation=90)
    yunit = "deg/s" if metric == "rotational_velocity" else units
    ax.set_ylabel(f"{label} [{yunit}]" + (" (|.|)" if metric == "rotational_velocity" else ""))
    ax.set_title(f"Per-fly {label} by group (red = control)")
    ax.axhline(0, color="grey", lw=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def group_timeseries(tidy, metric, bin_s):
    """Per-group mean/sem of a metric over binned time, averaged across flies."""
    df = tidy.dropna(subset=[metric])[["group", "video", "track", "time", metric]].copy()
    df["tbin"] = (df["time"] // bin_s) * bin_s
    per_fly = df.groupby(["group", "video", "track", "tbin"], dropna=False)[metric].mean().reset_index()
    agg = per_fly.groupby(["group", "tbin"])[metric].agg(["mean", "sem", "count"]).reset_index()
    return agg


def plot_timeseries(tidy, metric, label, units, out_path, bin_s=0.25, stim_on=None):
    """Per-group small-multiples of mean+/-sem over time vs pooled-control mean."""
    agg = group_timeseries(tidy, metric, bin_s)
    groups = order_groups(agg["group"].unique())

    # Pooled control reference.
    ctrl = agg[agg["group"].isin(CONTROL_GROUPS)]
    ctrl_ref = ctrl.groupby("tbin")["mean"].mean() if not ctrl.empty else None

    ncols = min(4, len(groups)) or 1
    nrows = math.ceil(len(groups) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    yunit = "deg/s" if metric == "rotational_velocity" else units

    for idx, g in enumerate(groups):
        ax = axes[idx // ncols][idx % ncols]
        gd = agg[agg["group"] == g].sort_values("tbin")
        if ctrl_ref is not None:
            ax.plot(ctrl_ref.index, ctrl_ref.values, color="grey", lw=1,
                    ls="--", label="pooled control")
        color = "tab:red" if g in CONTROL_GROUPS else "tab:blue"
        ax.plot(gd["tbin"], gd["mean"], color=color, lw=1.2)
        ax.fill_between(gd["tbin"], gd["mean"] - gd["sem"], gd["mean"] + gd["sem"],
                        color=color, alpha=0.25)
        ax.axhline(0, color="grey", lw=0.5)
        n_flies = tidy.loc[tidy["group"] == g, ["video", "track"]].drop_duplicates().shape[0]
        ax.set_title(f"{g} (n={n_flies})", fontsize=9)
        if stim_on:
            for s, e in stim_on:
                ax.axvspan(s, e, color="gold", alpha=0.2, lw=0)

    for j in range(len(groups), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.supxlabel("Time (s)")
    fig.supylabel(f"{label} [{yunit}]")
    fig.suptitle(f"{label} over time by group (grey dashed = pooled control)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------- stim-locked analyses


def _facet_axes(groups, ncols=4, panel=(4, 2.6), **kw):
    """Make a facet grid sized to the number of groups; returns (fig, axes_flat)."""
    ncols = min(ncols, len(groups)) or 1
    nrows = math.ceil(len(groups) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows),
                             sharex=True, sharey=True, squeeze=False, **kw)
    flat = [axes[i // ncols][i % ncols] for i in range(nrows * ncols)]
    for ax in flat[len(groups):]:
        ax.axis("off")
    return fig, flat


def summarize_on_off(sub):
    """Per-fly mean speed / forward velocity split by stimulus ON vs OFF."""
    rows = []
    for keys, g in sub.groupby(["group", "is_control", "video", "track"], dropna=False):
        on, off = g[g["stim_on"]], g[~g["stim_on"]]
        rows.append({
            "group": keys[0], "is_control": keys[1], "video": keys[2], "track": keys[3],
            "speed_on": on["speed"].mean(), "speed_off": off["speed"].mean(),
            "forward_on": on["forward_velocity"].mean(), "forward_off": off["forward_velocity"].mean(),
        })
    return pd.DataFrame(rows)


def plot_on_off(onoff, base_metric, label, units, out_path):
    """Per-group paired box of ON vs OFF for a metric."""
    groups = order_groups(onoff["group"].unique())
    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 0.6), 5))
    positions_on = [i * 2.0 for i in range(len(groups))]
    positions_off = [p + 0.8 for p in positions_on]
    data_on = [onoff.loc[onoff["group"] == g, f"{base_metric}_on"].dropna() for g in groups]
    data_off = [onoff.loc[onoff["group"] == g, f"{base_metric}_off"].dropna() for g in groups]
    b1 = ax.boxplot(data_on, positions=positions_on, widths=0.7, showfliers=False, patch_artist=True)
    b2 = ax.boxplot(data_off, positions=positions_off, widths=0.7, showfliers=False, patch_artist=True)
    for box in b1["boxes"]:
        box.set_facecolor("gold")
    for box in b2["boxes"]:
        box.set_facecolor("lightgrey")
    ax.set_xticks([p + 0.4 for p in positions_on])
    ax.set_xticklabels(groups, rotation=90)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel(f"{label} [{units}]")
    ax.legend([b1["boxes"][0], b2["boxes"][0]], ["ON", "OFF"], loc="best")
    ax.set_title(f"{label}: stimulus ON vs OFF by group")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def onset_locked(sub, metric, on_iv, pre, post, bin_s):
    """Average a metric in a window around each pulse onset, per group.

    Pools pulses and flies; returns group, rel-time bin, mean, sem, count.
    """
    recs = []
    for s, _e in on_iv:
        w = sub[(sub["time"] >= s - pre) & (sub["time"] < s + post)]
        w = w[["group", "video", "track", "time", metric]].dropna(subset=[metric]).copy()
        if not w.empty:
            w["rel"] = w["time"] - s
            recs.append(w)
    if not recs:
        return None
    allw = pd.concat(recs, ignore_index=True)
    allw["rb"] = (allw["rel"] // bin_s) * bin_s
    per_fly = allw.groupby(["group", "video", "track", "rb"])[metric].mean().reset_index()
    return per_fly.groupby(["group", "rb"])[metric].agg(["mean", "sem", "count"]).reset_index()


def plot_psth(agg, label, units, out_path, on_dur):
    """Per-group small-multiples of the onset-locked mean+/-sem (t=0 = pulse on)."""
    if agg is None or agg.empty:
        return
    groups = order_groups(agg["group"].unique())
    fig, axes = _facet_axes(groups)
    for ax, g in zip(axes, groups):
        gd = agg[agg["group"] == g].sort_values("rb")
        color = "tab:red" if g in CONTROL_GROUPS else "tab:blue"
        ax.axvspan(0, on_dur, color="gold", alpha=0.18, lw=0)
        ax.axvline(0, color="k", lw=0.6)
        ax.axhline(0, color="grey", lw=0.5)
        ax.plot(gd["rb"], gd["mean"], color=color, lw=1.2)
        ax.fill_between(gd["rb"], gd["mean"] - gd["sem"], gd["mean"] + gd["sem"], color=color, alpha=0.25)
        ax.set_title(g, fontsize=9)
    fig.supxlabel("Time from stimulus onset (s)")
    fig.supylabel(f"{label} [{units}]")
    fig.suptitle(f"{label} aligned to stimulus onset (gold = ON; pooled over pulses)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def per_stim_metrics(sub):
    """Per (group, fly, pulse) path integrals over each stimulation pulse.

    - ``back_dist``  : cumulated backward distance (integral of backward speed);
    - ``fwd_dist``   : cumulated forward distance (integral of forward speed);
    - ``net_forward``: net forward displacement (signed integral, +fwd / -back).
    """
    d = sub.dropna(subset=["forward_velocity", "on_index"]).copy()
    fv = d["forward_velocity"].to_numpy()
    dt = d["dt"].to_numpy()
    d["back_dist"] = np.clip(-fv, 0, None) * dt
    d["fwd_dist"] = np.clip(fv, 0, None) * dt
    d["net_forward"] = fv * dt
    return (
        d.groupby(["group", "is_control", "video", "track", "on_index"], dropna=False)[
            ["back_dist", "fwd_dist", "net_forward"]
        ]
        .sum()
        .reset_index()
    )


def per_stim_stats(per_stim, value_col, min_n=3):
    """Mann-Whitney U (two-sided) per (PG line, pulse) vs pooled control.

    p-values are Benjamini-Hochberg corrected across all valid tests.
    """
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    ctrl = per_stim[per_stim["is_control"]]
    pulses = sorted(per_stim["on_index"].dropna().unique())
    rows = []
    for g in per_stim.loc[~per_stim["is_control"], "group"].unique():
        gd = per_stim[per_stim["group"] == g]
        for pulse in pulses:
            a = gd.loc[gd["on_index"] == pulse, value_col].dropna()
            b = ctrl.loc[ctrl["on_index"] == pulse, value_col].dropna()
            U, p = np.nan, np.nan
            if len(a) >= min_n and len(b) >= min_n:
                try:
                    U, p = mannwhitneyu(a, b, alternative="two-sided")
                except ValueError:
                    pass
            rows.append({"group": g, "on_index": pulse, "metric": value_col,
                         "n_pg": len(a), "n_ctrl": len(b), "U": U, "p": p})
    df = pd.DataFrame(rows)
    df["p_adj"] = np.nan
    mask = df["p"].notna()
    if mask.any():
        df.loc[mask, "p_adj"] = multipletests(df.loc[mask, "p"], method="fdr_bh")[1]
    df["significant"] = df["p_adj"] < 0.05
    return df


def plot_per_stim(per_stim, stats_df, value_col, label, units, out_path):
    """Per PG line: a path-integral metric per pulse vs the pooled control,
    with significant pulses (BH-corrected Mann-Whitney) starred."""
    dist_unit = "mm" if units == "mm/s" else "px"
    ctrl = per_stim[per_stim["is_control"]]
    ctrl_agg = (
        ctrl.groupby("on_index")[value_col].agg(["mean", "sem"]).reset_index()
        if not ctrl.empty else None
    )
    pg_groups = order_groups(per_stim.loc[~per_stim["is_control"], "group"].unique())
    if not pg_groups:
        return
    n_ctrl_flies = ctrl[["video", "track"]].drop_duplicates().shape[0]
    fig, axes = _facet_axes(pg_groups, panel=(3.6, 2.6))
    for ax, g in zip(axes, pg_groups):
        gd = per_stim[per_stim["group"] == g].groupby("on_index")[value_col].agg(["mean", "sem", "count"]).reset_index()
        ax.axhline(0, color="grey", lw=0.5)
        if ctrl_agg is not None:
            ax.errorbar(ctrl_agg["on_index"], ctrl_agg["mean"], yerr=ctrl_agg["sem"],
                        color="grey", marker="o", ms=3, lw=1, ls="--", capsize=2)
        ax.errorbar(gd["on_index"], gd["mean"], yerr=gd["sem"], color="tab:blue",
                    marker="o", ms=3, lw=1.2, capsize=2)
        # significance stars
        sig = stats_df[(stats_df["group"] == g) & stats_df["significant"]]
        for _, r in sig.iterrows():
            row = gd[gd["on_index"] == r["on_index"]]
            if row.empty:
                continue
            y = row["mean"].iloc[0] + row["sem"].iloc[0]
            yc = ctrl_agg.loc[ctrl_agg["on_index"] == r["on_index"], "mean"]
            top = max(y, (yc.iloc[0] if not yc.empty else y))
            ax.annotate("*", (r["on_index"], top), ha="center", va="bottom",
                        fontsize=12, color="black")
        n = int(gd["count"].max()) if not gd.empty else 0
        ax.set_title(f"{g} (n<={n})", fontsize=9)
    fig.supxlabel("Stimulation (pulse #)")
    fig.supylabel(f"{label} per pulse [{dist_unit}]")
    fig.suptitle(f"{label} per stimulation: PG line (blue) vs pooled control (grey, "
                 f"n={n_ctrl_flies}); * = BH-corrected p<0.05")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- CLI


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="YAML video-list, directory, or .h5 files")
    # Default calibration: 832 px frame spans a 32 mm field of view (pixel_size
    # = 32/832 mm/px) -> 26 px/mm. Verified: head<->abdomen ~= 2.4 mm. Pass 0 for px/s.
    parser.add_argument("--px-per-mm", type=float, default=832 / 32, help="Pixels per mm (default 26.0 = 832px/32mm; pass 0 for px/s)")
    parser.add_argument("--output-dir", type=Path, default=Path("directional_velocity"), help="Where to write tables/figures")
    parser.add_argument("--smooth-window", type=int, default=25, help="Savitzky-Golay window (frames)")
    parser.add_argument("--smooth-poly", type=int, default=2, help="Savitzky-Golay polynomial order")
    parser.add_argument("--bin-s", type=float, default=0.25, help="Time bin (s) for the time-series plots")
    parser.add_argument("--psth-pre", type=float, default=10.0, help="Seconds before pulse onset for onset-locked plots")
    parser.add_argument("--psth-post", type=float, default=70.0, help="Seconds after pulse onset for onset-locked plots")
    parser.add_argument("--scope", choices=["shared", "long", "both"], default="both",
                        help="'shared': pool short+long flies over the common 360.5s window; "
                             "'long': long-protocol flies over the full timeline; 'both' writes both.")
    parser.add_argument("--exclude-name-contains", nargs="*", default=[], help="Skip files whose genotype dir contains any of these (e.g. BAD)")
    parser.add_argument("--no-plots", action="store_true", help="Only write the tables, skip figures")
    args = parser.parse_args()

    files = collect_h5_files(args.paths)
    if args.exclude_name_contains:
        keep = []
        for f in files:
            gdir = f.parents[2].name if len(f.parents) >= 3 else f.parent.name
            if any(tok.lower() in gdir.lower() for tok in args.exclude_name_contains):
                continue
            keep.append(f)
        print(f"Excluded {len(files) - len(keep)} file(s) by name filter.")
        files = keep

    if not files:
        raise SystemExit("No *_tracked.h5 files found.")
    if not args.px_per_mm:
        print("WARNING: --px-per-mm=0; translational units will be px/s.")

    print(f"Computing kinematics for {len(files)} file(s)...")
    frames = []
    for i, f in enumerate(files, 1):
        df = kinematics_for_h5(f, px_per_mm=args.px_per_mm,
                               smooth_window=args.smooth_window, smooth_poly=args.smooth_poly)
        if not df.empty:
            frames.append(df)
        if i % 25 == 0:
            print(f"  {i}/{len(files)}")

    if not frames:
        raise SystemExit("No usable tracks found.")
    tidy = pd.concat(frames, ignore_index=True)
    units = tidy["units"].iloc[0]

    # Per-video stimulation protocol + per-frame stim labels.
    tidy["sequence"] = tidy["source_h5"].map(sequences.detect_sequence)
    tidy["stim_on"] = False
    tidy["on_index"] = np.nan
    for seq, sub in tidy.groupby("sequence"):
        if seq is None:
            print(f"  WARNING: {sub['video'].nunique()} video(s) with unknown protocol (no shading).")
            continue
        on, oi = sequences.label_times(sub["time"].to_numpy(), seq)
        tidy.loc[sub.index, "stim_on"] = on
        tidy.loc[sub.index, "on_index"] = oi

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tidy.to_feather(args.output_dir / "directional_velocity_frames.feather")
    summarize_per_fly(tidy).to_feather(args.output_dir / "directional_velocity_per_fly.feather")
    print(f"\nWrote full tables to {args.output_dir} "
          f"({len(tidy)} frame-rows, units={units}).")

    if args.no_plots:
        return

    shared_end = sequences.shared_window_end()  # 360.5 s
    scope_defs = {
        # pooled short+long flies over the common window
        "shared": (tidy[tidy["time"] <= shared_end], sequences.on_intervals("short")),
        # long-protocol flies over their full timeline (all 15 pulses)
        "long": (tidy[tidy["sequence"] == "long"], sequences.on_intervals("long")),
    }
    scopes = ["shared", "long"] if args.scope == "both" else [args.scope]

    for scope in scopes:
        sub, on_iv = scope_defs[scope]
        if sub.empty:
            print(f"  [scope:{scope}] no data, skipping.")
            continue
        outdir = args.output_dir / scope
        outdir.mkdir(parents=True, exist_ok=True)
        per_fly = summarize_per_fly(sub)
        per_fly.to_feather(outdir / "per_fly.feather")
        n_groups = per_fly["group"].nunique()
        n_flies = len(per_fly)
        print(f"  [scope:{scope}] {n_flies} flies, {n_groups} groups -> {outdir}")
        for metric, label in METRICS.items():
            plot_summary(per_fly, metric, label, units, outdir / f"summary_{metric}.png")
            plot_timeseries(sub, metric, label, units, outdir / f"timeseries_{metric}.png",
                            bin_s=args.bin_s, stim_on=on_iv)

        # Stimulus ON vs OFF (per-fly means).
        onoff = summarize_on_off(sub)
        onoff.to_feather(outdir / "per_fly_on_off.feather")
        plot_on_off(onoff, "speed", "Speed", units, outdir / "onoff_speed.png")
        plot_on_off(onoff, "forward", "Forward velocity", units, outdir / "onoff_forward.png")

        # Onset-locked averages (PSTH-style), per metric.
        for metric, label in METRICS.items():
            agg = onset_locked(sub, metric, on_iv, args.psth_pre, args.psth_post, args.bin_s)
            plot_psth(agg, label, units, outdir / f"psth_{metric}.png", on_dur=60.0)

        # Path integrals per stimulation: PG lines vs pooled control, with
        # per-pulse Mann-Whitney tests (BH-corrected).
        per_stim = per_stim_metrics(sub)
        per_stim.to_feather(outdir / "per_stim_metrics.feather")
        stats_all = []
        for value_col, mlabel in [("back_dist", "Cumulated backward distance"),
                                  ("net_forward", "Net forward displacement")]:
            stats_df = per_stim_stats(per_stim, value_col)
            stats_all.append(stats_df)
            plot_per_stim(per_stim, stats_df, value_col, mlabel, units,
                          outdir / f"{value_col}_per_stim.png")
        if stats_all:
            pd.concat(stats_all, ignore_index=True).to_feather(outdir / "per_stim_stats.feather")


if __name__ == "__main__":
    main()
