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

from utils_behavior.sleap import sequences, stats
from utils_behavior.sleap.kinematics import (
    CONTROL_GROUPS,
    kinematics_for_h5,
    summarize_per_fly,
)

METRICS = {
    "speed": "Speed",
    "forward_velocity": "Forward velocity (+fwd / -back)",
    "backward_speed": "Backward speed (rectified)",
    "rotational_velocity": "Rotational velocity",
}

# Statistics config (distribution-free): permutation tests for group comparisons,
# percentile bootstrap CIs for variability bands. Set from CLI args in main().
SCFG = {"n_perm": 10000, "n_boot": 1000, "ci": 95.0, "alpha": 0.05, "seed": 0}


def _ci_point(values):
    """(point, lo, hi) percentile-bootstrap CI using the global stats config."""
    return stats.bootstrap_ci(values, n_boot=SCFG["n_boot"], ci=SCFG["ci"], seed=SCFG["seed"])


def _yerr_from_ci(point, lo, hi):
    """Asymmetric yerr column for ax.errorbar from a (point, lo, hi) CI."""
    return [[max(point - lo, 0.0)], [max(hi - point, 0.0)]]


def _stars(p):
    """Significance stars for a (corrected) p-value; '' if NS or undefined."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


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
        "backward_speed": "mean_backward_speed_rect",
        "rotational_velocity": "mean_abs_rotational",
    }[metric]
    groups = order_groups(per_fly["group"].unique())
    data = [per_fly.loc[per_fly["group"] == g, col].dropna().to_numpy() for g in groups]

    # Permutation test of each PG line vs the pooled control (BH-corrected).
    ctrl_vals = per_fly.loc[per_fly["is_control"], col].dropna().to_numpy()
    pvals = []
    for g in groups:
        gv = per_fly.loc[per_fly["group"] == g, col].dropna().to_numpy()
        if g in CONTROL_GROUPS or ctrl_vals.size == 0 or gv.size == 0:
            pvals.append(np.nan)
        else:
            _, p = stats.permutation_test(gv, ctrl_vals, n_perm=SCFG["n_perm"], seed=SCFG["seed"])
            pvals.append(p)
    p_adj = stats.bh_adjust(pvals)

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 0.45), 5))
    ax.boxplot(data, showfliers=False, medianprops=dict(color="black"))
    rng = np.random.default_rng(0)
    for i, (g, vals) in enumerate(zip(groups, data), start=1):
        if not len(vals):
            continue
        x = i + rng.uniform(-0.15, 0.15, size=len(vals))
        color = "tab:red" if g in CONTROL_GROUPS else "tab:blue"
        ax.scatter(x, vals, s=14, alpha=0.6, color=color, edgecolors="none")
        star = _stars(p_adj[i - 1])
        if star:
            ax.annotate(star, (i, np.nanmax(vals)), ha="center", va="bottom",
                        fontsize=11, color="black")
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(groups, rotation=90)
    yunit = "deg/s" if metric == "rotational_velocity" else units
    ax.set_ylabel(f"{label} [{yunit}]" + (" (|.|)" if metric == "rotational_velocity" else ""))
    ax.set_title(f"Per-fly {label} by group (red = control; * = BH perm test vs pooled control)")
    ax.axhline(0, color="grey", lw=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _fly_bin_matrix(df_group, metric, bin_s, bins):
    """Per-fly x per-time-bin matrix (rows = flies) aligned to a common ``bins`` grid."""
    d = df_group.dropna(subset=[metric])[["video", "track", "time", metric]].copy()
    if d.empty:
        return np.empty((0, len(bins)))
    d["tbin"] = (d["time"] // bin_s) * bin_s
    pf = d.groupby(["video", "track", "tbin"])[metric].mean().reset_index()
    mat = pf.pivot_table(index=["video", "track"], columns="tbin", values=metric)
    return mat.reindex(columns=bins).to_numpy()


def plot_timeseries(tidy, metric, label, units, out_path, bin_s=0.25, stim_on=None):
    """Per-group small-multiples of mean + bootstrap-CI band vs pooled-control mean."""
    df_all = tidy.dropna(subset=[metric])
    if df_all.empty:
        return
    bins = np.sort(((df_all["time"] // bin_s) * bin_s).unique())
    groups = order_groups(df_all["group"].unique())

    # Pooled control reference (mean over control flies per bin).
    ctrl_mat = _fly_bin_matrix(df_all[df_all["is_control"]], metric, bin_s, bins)
    with np.errstate(invalid="ignore"):
        ctrl_ref = np.nanmean(ctrl_mat, axis=0) if ctrl_mat.shape[0] else None

    ncols = min(4, len(groups)) or 1
    nrows = math.ceil(len(groups) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    yunit = "deg/s" if metric == "rotational_velocity" else units

    for idx, g in enumerate(groups):
        ax = axes[idx // ncols][idx % ncols]
        mat = _fly_bin_matrix(df_all[df_all["group"] == g], metric, bin_s, bins)
        if ctrl_ref is not None:
            ax.plot(bins, ctrl_ref, color="grey", lw=1, ls="--", label="pooled control")
        color = "tab:red" if g in CONTROL_GROUPS else "tab:blue"
        point, lo, hi = stats.bootstrap_ci_matrix(mat, n_boot=SCFG["n_boot"],
                                                  ci=SCFG["ci"], seed=SCFG["seed"])
        ax.plot(bins, point, color=color, lw=1.2)
        ax.fill_between(bins, lo, hi, color=color, alpha=0.25)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"{g} (n={mat.shape[0]})", fontsize=9)
        if stim_on:
            for s, e in stim_on:
                ax.axvspan(s, e, color="gold", alpha=0.2, lw=0)

    for j in range(len(groups), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.supxlabel("Time (s)")
    fig.supylabel(f"{label} [{yunit}]")
    fig.suptitle(f"{label} over time by group (mean +/- {int(SCFG['ci'])}% bootstrap CI; "
                 f"grey dashed = pooled control)")
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
    """Per-group paired box of ON vs OFF with raw per-fly points and a paired
    permutation test (ON vs OFF, BH-corrected across groups)."""
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

    # Raw per-fly points (jittered) + paired ON-vs-OFF permutation test per group.
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        sub = onoff[onoff["group"] == g]
        on_v = sub[f"{base_metric}_on"].to_numpy()
        off_v = sub[f"{base_metric}_off"].to_numpy()
        ax.scatter(positions_on[i] + rng.uniform(-0.18, 0.18, on_v.size), on_v,
                   s=12, alpha=0.6, color="darkgoldenrod", edgecolors="none")
        ax.scatter(positions_off[i] + rng.uniform(-0.18, 0.18, off_v.size), off_v,
                   s=12, alpha=0.6, color="dimgrey", edgecolors="none")
    pvals = []
    for g in groups:
        sub = onoff[onoff["group"] == g]
        _, p = stats.paired_permutation_test(sub[f"{base_metric}_on"].to_numpy(),
                                             sub[f"{base_metric}_off"].to_numpy(),
                                             n_perm=SCFG["n_perm"], seed=SCFG["seed"])
        pvals.append(p)
    p_adj = stats.bh_adjust(pvals)
    for i, g in enumerate(groups):
        star = _stars(p_adj[i])
        if star:
            top = np.nanmax(np.concatenate([data_on[i].to_numpy(), data_off[i].to_numpy(), [0]]))
            ax.annotate(star, (positions_on[i] + 0.4, top), ha="center", va="bottom", fontsize=11)

    ax.set_xticks([p + 0.4 for p in positions_on])
    ax.set_xticklabels(groups, rotation=90)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel(f"{label} [{units}]")
    ax.legend([b1["boxes"][0], b2["boxes"][0]], ["ON", "OFF"], loc="best")
    ax.set_title(f"{label}: stimulus ON vs OFF by group (* = BH paired perm test)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def onset_locked(sub, metric, on_iv, pre, post, bin_s):
    """Per-fly onset-locked means, pooling pulses (t=0 = pulse onset).

    Returns a long table (group, video, track, rb, value) — one row per fly per
    rel-time bin — so downstream plotting can bootstrap CIs over flies.
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
    return (allw.groupby(["group", "video", "track", "rb"])[metric].mean()
            .reset_index().rename(columns={metric: "value"}))


def plot_psth(agg, label, units, out_path, on_dur):
    """Per-group onset-locked mean + bootstrap-CI band (pooled over pulses)."""
    if agg is None or agg.empty:
        return
    groups = order_groups(agg["group"].unique())
    bins = np.sort(agg["rb"].unique())
    fig, axes = _facet_axes(groups)
    for ax, g in zip(axes, groups):
        gd = agg[agg["group"] == g]
        mat = (gd.pivot_table(index=["video", "track"], columns="rb", values="value")
               .reindex(columns=bins).to_numpy())
        color = "tab:red" if g in CONTROL_GROUPS else "tab:blue"
        ax.axvspan(0, on_dur, color="gold", alpha=0.18, lw=0)
        ax.axvline(0, color="k", lw=0.6)
        ax.axhline(0, color="grey", lw=0.5)
        point, lo, hi = stats.bootstrap_ci_matrix(mat, n_boot=SCFG["n_boot"],
                                                  ci=SCFG["ci"], seed=SCFG["seed"])
        ax.plot(bins, point, color=color, lw=1.2)
        ax.fill_between(bins, lo, hi, color=color, alpha=0.25)
        ax.set_title(f"{g} (n={mat.shape[0]})", fontsize=9)
    fig.supxlabel("Time from stimulus onset (s)")
    fig.supylabel(f"{label} [{units}]")
    fig.suptitle(f"{label} aligned to stimulus onset (mean +/- {int(SCFG['ci'])}% bootstrap CI; "
                 f"gold = ON; pooled over pulses)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def onset_locked_per_pulse(sub, metric, on_iv, pre, post, bin_s):
    """Onset-locked average of a metric for EACH pulse separately (not pooled).

    Like :func:`onset_locked` but keeps the 1-based pulse number so the response
    can be compared across the stimulation train. Returns group, pulse, rel-time
    bin, and the flies-averaged ``value``.
    """
    recs = []
    for pulse, (s, _e) in enumerate(on_iv, start=1):
        w = sub[(sub["time"] >= s - pre) & (sub["time"] < s + post)]
        w = w[["group", "video", "track", "time", metric]].dropna(subset=[metric]).copy()
        if w.empty:
            continue
        w["rel"] = w["time"] - s
        w["pulse"] = pulse
        recs.append(w)
    if not recs:
        return None
    allw = pd.concat(recs, ignore_index=True)
    allw["rb"] = (allw["rel"] // bin_s) * bin_s
    per_fly = allw.groupby(["group", "pulse", "video", "track", "rb"])[metric].mean().reset_index()
    return (per_fly.groupby(["group", "pulse", "rb"])[metric].mean()
            .reset_index().rename(columns={metric: "value"}))


def plot_psth_per_pulse(agg, label, units, out_path, on_dur, rot=False):
    """Per-group onset-locked traces, one line per pulse colored by pulse number."""
    if agg is None or agg.empty:
        return
    groups = order_groups(agg["group"].unique())
    pulses = sorted(int(p) for p in agg["pulse"].unique())
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(min(pulses), max(pulses) if len(pulses) > 1 else min(pulses) + 1)
    fig, axes = _facet_axes(groups)
    for ax, g in zip(axes, groups):
        gd = agg[agg["group"] == g]
        ax.axvspan(0, on_dur, color="gold", alpha=0.15, lw=0)
        ax.axvline(0, color="k", lw=0.5)
        ax.axhline(0, color="grey", lw=0.5)
        for p in pulses:
            d = gd[gd["pulse"] == p].sort_values("rb")
            if d.empty:
                continue
            ax.plot(d["rb"], d["value"], color=cmap(norm(p)), lw=1.0)
        ax.set_title(g, fontsize=9)
    yunit = "deg/s" if rot else units
    fig.supxlabel("Time from stimulus onset (s)")
    fig.supylabel(f"{label} [{yunit}]")
    fig.suptitle(f"{label} onset-locked, per stimulation (color = pulse #; gold = ON)")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:len(groups)], fraction=0.025, pad=0.01)
    cbar.set_label("pulse #")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def per_pulse_means(sub, metric_cols):
    """Per (group, fly, pulse) mean of each metric over that pulse's ON frames."""
    on = sub[sub["stim_on"] & sub["on_index"].notna()]
    if on.empty:
        return pd.DataFrame()
    return (on.groupby(["group", "is_control", "video", "track", "on_index"], dropna=False)
            [metric_cols].mean().reset_index())


def _ci_by_pulse(df, metric, pulses):
    """Per-pulse (point, lo, hi) bootstrap CI arrays for one group's per-fly table."""
    pts, los, his = [], [], []
    for pulse in pulses:
        vals = df.loc[df["on_index"] == pulse, metric].to_numpy()
        point, lo, hi = _ci_point(vals)
        pts.append(point); los.append(lo); his.append(hi)
    return np.array(pts), np.array(los), np.array(his)


def plot_per_pulse_trend(per_pulse, metric, label, units, out_path, rot=False):
    """Per-group mean + bootstrap-CI of a metric during each ON pulse vs pulse #,
    with the pooled control reference and a per-pulse permutation test vs control."""
    if per_pulse.empty:
        return
    pulses = sorted(per_pulse["on_index"].dropna().unique())
    ctrl = per_pulse[per_pulse["is_control"]]
    ctrl_pt, ctrl_lo, ctrl_hi = (_ci_by_pulse(ctrl, metric, pulses)
                                 if not ctrl.empty else (None, None, None))
    groups = order_groups(per_pulse["group"].unique())

    # Permutation test PG vs pooled control per (group, pulse), BH across all.
    pg_groups = [g for g in groups if g not in CONTROL_GROUPS]
    raw_p, keys = [], []
    for g in pg_groups:
        gd = per_pulse[per_pulse["group"] == g]
        for pulse in pulses:
            a = gd.loc[gd["on_index"] == pulse, metric].to_numpy()
            b = ctrl.loc[ctrl["on_index"] == pulse, metric].to_numpy() if not ctrl.empty else np.array([])
            _, p = stats.permutation_test(a, b, n_perm=SCFG["n_perm"], seed=SCFG["seed"])
            raw_p.append(p); keys.append((g, pulse))
    p_adj = stats.bh_adjust(raw_p)
    sig = {k: pa for k, pa in zip(keys, p_adj)}

    fig, axes = _facet_axes(groups, panel=(3.6, 2.6))
    for ax, g in zip(axes, groups):
        gd = per_pulse[per_pulse["group"] == g]
        pt, lo, hi = _ci_by_pulse(gd, metric, pulses)
        ax.axhline(0, color="grey", lw=0.5)
        if ctrl_pt is not None:
            ax.errorbar(pulses, ctrl_pt, yerr=[ctrl_pt - ctrl_lo, ctrl_hi - ctrl_pt],
                        color="grey", marker="o", ms=3, lw=1, ls="--", capsize=2)
        color = "tab:red" if g in CONTROL_GROUPS else "tab:blue"
        ax.errorbar(pulses, pt, yerr=[pt - lo, hi - pt], color=color,
                    marker="o", ms=3, lw=1.2, capsize=2)
        for j, pulse in enumerate(pulses):
            star = _stars(sig.get((g, pulse), np.nan))
            if star:
                ax.annotate(star, (pulse, max(hi[j], ctrl_hi[j] if ctrl_hi is not None else hi[j])),
                            ha="center", va="bottom", fontsize=10)
        ax.set_title(g, fontsize=9)
    yunit = "deg/s" if rot else units
    fig.supxlabel("Stimulation (pulse #)")
    fig.supylabel(f"mean {label} during ON [{yunit}]")
    fig.suptitle(f"{label}: mean during each ON pulse +/- {int(SCFG['ci'])}% bootstrap CI "
                 f"(grey dashed = pooled control; * = BH perm test vs control)")
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
    """Two-sided permutation test per (PG line, pulse) vs pooled control.

    Distribution-free (difference of means); p-values are Benjamini-Hochberg
    corrected across all valid tests.
    """
    ctrl = per_stim[per_stim["is_control"]]
    pulses = sorted(per_stim["on_index"].dropna().unique())
    rows = []
    for g in per_stim.loc[~per_stim["is_control"], "group"].unique():
        gd = per_stim[per_stim["group"] == g]
        for pulse in pulses:
            a = gd.loc[gd["on_index"] == pulse, value_col].dropna().to_numpy()
            b = ctrl.loc[ctrl["on_index"] == pulse, value_col].dropna().to_numpy()
            diff, p = np.nan, np.nan
            if len(a) >= min_n and len(b) >= min_n:
                diff, p = stats.permutation_test(a, b, n_perm=SCFG["n_perm"], seed=SCFG["seed"])
            rows.append({"group": g, "on_index": pulse, "metric": value_col,
                         "n_pg": len(a), "n_ctrl": len(b), "diff": diff, "p": p})
    df = pd.DataFrame(rows)
    df["p_adj"] = stats.bh_adjust(df["p"].to_numpy()) if not df.empty else []
    df["significant"] = df["p_adj"] < SCFG["alpha"]
    return df


def plot_per_stim(per_stim, stats_df, value_col, label, units, out_path):
    """Per PG line: a path-integral metric per pulse vs the pooled control,
    with significant pulses (BH-corrected Mann-Whitney) starred."""
    dist_unit = "mm" if units == "mm/s" else "px"
    ctrl = per_stim[per_stim["is_control"]]
    pulses = sorted(per_stim["on_index"].dropna().unique())
    ctrl_pt, ctrl_lo, ctrl_hi = (_ci_by_pulse(ctrl, value_col, pulses)
                                 if not ctrl.empty else (None, None, None))
    pg_groups = order_groups(per_stim.loc[~per_stim["is_control"], "group"].unique())
    if not pg_groups:
        return
    n_ctrl_flies = ctrl[["video", "track"]].drop_duplicates().shape[0]
    fig, axes = _facet_axes(pg_groups, panel=(3.6, 2.6))
    for ax, g in zip(axes, pg_groups):
        gd = per_stim[per_stim["group"] == g]
        pt, lo, hi = _ci_by_pulse(gd, value_col, pulses)
        ax.axhline(0, color="grey", lw=0.5)
        if ctrl_pt is not None:
            ax.errorbar(pulses, ctrl_pt, yerr=[ctrl_pt - ctrl_lo, ctrl_hi - ctrl_pt],
                        color="grey", marker="o", ms=3, lw=1, ls="--", capsize=2)
        ax.errorbar(pulses, pt, yerr=[pt - lo, hi - pt], color="tab:blue",
                    marker="o", ms=3, lw=1.2, capsize=2)
        # significance stars (BH-corrected permutation test)
        sig = stats_df[(stats_df["group"] == g) & stats_df["significant"]]
        for _, r in sig.iterrows():
            j = pulses.index(r["on_index"]) if r["on_index"] in pulses else None
            if j is None:
                continue
            top = max(hi[j], ctrl_hi[j] if ctrl_hi is not None else hi[j])
            ax.annotate(_stars(r["p_adj"]) or "*", (r["on_index"], top),
                        ha="center", va="bottom", fontsize=11, color="black")
        n = int(gd.groupby("on_index").size().max()) if not gd.empty else 0
        ax.set_title(f"{g} (n<={n})", fontsize=9)
    fig.supxlabel("Stimulation (pulse #)")
    fig.supylabel(f"{label} per pulse [{dist_unit}]")
    fig.suptitle(f"{label} per stimulation: PG line (blue) +/- {int(SCFG['ci'])}% bootstrap CI "
                 f"vs pooled control (grey, n={n_ctrl_flies}); * = BH perm test")
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
    parser.add_argument("--max-clean-tracks", type=int, default=6,
                        help="Skip h5 files with more tracks than this — they were not "
                             "successfully cleaned (identity fragmentation); 0 disables the check")
    parser.add_argument("--no-plots", action="store_true", help="Only write the tables, skip figures")
    parser.add_argument("--n-perm", type=int, default=10000, help="Permutation-test resamples")
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples for CI bands/points")
    parser.add_argument("--ci", type=float, default=95.0, help="Bootstrap CI percentage")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for stars")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for permutation/bootstrap reproducibility")
    args = parser.parse_args()

    SCFG.update(n_perm=args.n_perm, n_boot=args.n_boot, ci=args.ci, alpha=args.alpha, seed=args.seed)

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

    # Skip files that were never successfully cleaned (still have many tracks from
    # identity fragmentation) — they would inject hundreds of spurious "flies".
    if args.max_clean_tracks:
        import h5py

        kept, skipped = [], []
        for f in files:
            try:
                with h5py.File(f, "r") as fh:
                    n = fh["tracks"].shape[0]
            except (OSError, KeyError):
                n = -1
            (skipped if n > args.max_clean_tracks else kept).append((f, n))
        if skipped:
            print(f"Skipping {len(skipped)} un-cleaned file(s) (> {args.max_clean_tracks} "
                  f"tracks — fragmented, re-track these):")
            for f, n in skipped:
                print(f"   [{n} tracks] {f}")
        files = [f for f, _ in kept]
        if not files:
            raise SystemExit("All files exceeded --max-clean-tracks; nothing to plot.")

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

        # Onset-locked averages (PSTH-style), per metric: pooled over pulses, plus
        # the non-pooled per-pulse overlay (one trace per stimulation).
        for metric, label in METRICS.items():
            agg = onset_locked(sub, metric, on_iv, args.psth_pre, args.psth_post, args.bin_s)
            plot_psth(agg, label, units, outdir / f"psth_{metric}.png", on_dur=60.0)
            agg_pp = onset_locked_per_pulse(sub, metric, on_iv, args.psth_pre, args.psth_post, args.bin_s)
            plot_psth_per_pulse(agg_pp, label, units, outdir / f"psth_per_pulse_{metric}.png",
                                on_dur=60.0, rot=metric == "rotational_velocity")

        # Per-pulse scalar trend: mean of each metric during each ON pulse vs pulse #.
        pp = per_pulse_means(sub, list(METRICS.keys()))
        if not pp.empty:
            pp.to_feather(outdir / "per_pulse_means.feather")
            for metric, label in METRICS.items():
                plot_per_pulse_trend(pp, metric, label, units, outdir / f"per_pulse_{metric}.png",
                                     rot=metric == "rotational_velocity")

        # Path integrals per stimulation: PG lines vs pooled control, with
        # per-pulse permutation tests (BH-corrected).
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
