# Optobot Vglut SLEAP tracking pipeline (uv-native)

Batch-track the Optobot `PG_YYYYMMDD_Vglut_B6VF_1min` arena videos (1–3 flies,
alone) plus the `PG` directory with the new **sleap-nn** (SLEAP 1.6) top-down
model, then denoise the resulting tracks.

Everything runs through **uv** — no conda/mamba. The SLEAP CLIs
(`sleap track`, `sleap-convert`) are the uv-installed `sleap` tool (v1.6.x) on
PATH; the `utils_behavior` tools run via `uv run` against this repo's
`pyproject.toml`. (`sleap-nn-track` still works but prints a deprecation note
pointing to the unified `sleap track` subcommand.)

## Models (sleap-nn, top-down)

- centroid: `/mnt/upramdya_data/_Tracking_models/Sleap/optobot/FlyTracking/SB_Vglut/models/260611_144441.centroid.n=186`
- centered_instance: `/mnt/upramdya_data/_Tracking_models/Sleap/optobot/FlyTracking/SB_Vglut/models/260611_150308.centered_instance.n=186`

Both are sleap-nn dirs (`best.ckpt` + `training_config.yaml`).

## Tracking recipe

- `-n 3` — at most 3 instances detected per frame.
- `-t --use_flow` — flow (FlowShiftTracker) identity tracking.
- `--max_tracks` is **not** used (only valid for the `local_queues` method,
  incompatible with flow). Identity count is consolidated to ≤3 flies by the
  denoising step below.
- analysis `.h5` is produced from the `.slp` with
  `sleap_io.save_analysis_h5`, run inside an isolated `uv run --with sleap-io`
  env (avoids the `sleap-convert` Qt crash seen when launched from conda).

## Steps to run (in a screen session)

```bash
export PATH="$HOME/.local/bin:$PATH"   # uv + the uv-installed sleap tools
cd /home/durrieu/utils_behavior

CENTROID="/mnt/upramdya_data/_Tracking_models/Sleap/optobot/FlyTracking/SB_Vglut/models/260611_144441.centroid.n=186"
CINST="/mnt/upramdya_data/_Tracking_models/Sleap/optobot/FlyTracking/SB_Vglut/models/260611_150308.centered_instance.n=186"
YAML=/mnt/upramdya_data/SB/Optogenetics/Optobot/optobot_vglut_videos.yaml

# 1. (Re)generate the video list (dated dirs 20260521–20260605 + the PG dir).
#    New recordings are picked up automatically.
uv run python scripts/sleap/make_video_list.py --extra-dirs PG --output "$YAML"

# 2. Batch tracking. Backend auto-detected as sleap-nn. Outputs *_tracked.slp +
#    *_tracked.h5 next to each video. Already-tracked videos are skipped, so the
#    run is resumable and only touches new/untracked recordings.
uv run python -m utils_behavior.sleap.tracker "$CENTROID" \
    --model_centered_instance_path "$CINST" \
    --yaml_file "$YAML" \
    --max_instances 3 \
    --batch_size 16

# 3. Denoise: drop parasite tracks, keep <=3 flies present for most of the video.
#    --dry-run first to preview; --inplace to overwrite the *_tracked.h5.
uv run python -m utils_behavior.sleap.clean_tracks "$YAML" --dry-run
uv run python -m utils_behavior.sleap.clean_tracks "$YAML" --inplace
```

> `clean_tracks` accepts the video-list YAML (maps each video to its
> `*_tracked.h5`), a directory (with `--recursive`), or explicit `.h5` paths.

## Raw single-video commands (what the tracker runs per video)

```bash
sleap track -i <video_80fps.mp4> \
    -m "$CENTROID" -m "$CINST" \
    -o <video_80fps_tracked.slp> \
    -b 16 -t -n 3 --use_flow -d auto

uv run --no-project --with sleap-io python -c \
  'import sys, sleap_io as sio; sio.save_analysis_h5(sio.load_file(sys.argv[1]), sys.argv[2])' \
  <video_80fps_tracked.slp> <video_80fps_tracked.h5>
```

## Directional velocity analysis (task 3)

After tracking + cleaning, compute per-fly directional velocity grouped by line:

```bash
uv run python scripts/sleap/directional_velocity.py "$YAML" \
    --output-dir /mnt/upramdya_data/SB/Optogenetics/Optobot/dv_figs
```

- Calibration default `--px-per-mm 26.0` (832 px = 32 mm) → mm/s; rotational in deg/s.
- Centroid = mean(Head, Thorax, Abdomen); forward = velocity along abdomen→head
  (signed, +fwd/−back); rotational = d(heading)/dt; speed = |v|.
- Groups parsed from the genotype dir: `PG_?NN` → `PGNN`; `B6VF` and
  `B6XOGL7` (pools `B6VFXOGL7`) are controls.

**Opto protocols** (`utils_behavior/sleap/sequences.py`), auto-detected per video:
- `short` (PG dir): five 60 s ON pulses, 15 s rests, total 361 s.
- `long` (dated dirs): identical for the first 360.5 s, then two more blocks of
  five pulses separated by 300 s rests (15 pulses total, ~1680 s).

`--scope both` (default) writes two figure sets:
- `dv_figs/shared/` — short + long flies **pooled** over the common 360.5 s
  window (5 ON bands shaded).
- `dv_figs/long/` — long-protocol flies over the full timeline (15 ON bands).

Metrics: `speed`, `forward_velocity` (signed, +fwd/−back), `backward_speed`
(rectified backward component, 0 while moving forward), `rotational_velocity`.

Each scope directory contains:
- `summary_<metric>.png` — per-group box+strip (controls in red).
- `timeseries_<metric>.png` — per-group mean±sem over time, pooled-control
  reference, stim shading.
- `onoff_speed.png`, `onoff_forward.png` — per-group **stimulus ON vs OFF** box plots.
- `psth_<metric>.png` — **onset-locked** averages **pooled over pulses** (metric vs
  time-from-pulse-onset; ON window shaded), one panel per group.
- `psth_per_pulse_<metric>.png` — **non-pooled** onset-locked overlay: one trace
  **per stimulation**, colored by pulse # (viridis), one panel per group. Shows how
  the response evolves across the train (habituation/sensitization).
- `per_pulse_<metric>.png` — **per-pulse scalar trend**: mean of the metric during
  each ON pulse vs pulse #, per group, with the pooled control as a grey dashed
  reference.
- `back_dist_per_stim.png`, `net_forward_per_stim.png` — per-pulse path integrals
  (cumulated backward distance / net forward displacement) vs pooled control, with
  BH-corrected Mann-Whitney stars.
- Tables: `per_fly.feather`, `per_fly_on_off.feather`, `per_pulse_means.feather`,
  `per_stim_metrics.feather`, `per_stim_stats.feather`.

The full per-frame tidy table (`directional_velocity_frames.feather`) carries
`sequence`, `stim_on`, `on_index`, `dt` for any further custom analysis.

## Cleanup of failed-attempt artifacts (done 2026-06-16)

The one failed DLC attempt under
`PG_20260521_Vglut_B6VF_1min/PG25_OK/.../134418_s0a0_p6-0/` was cleaned with
`scripts/sleap/cleanup_failed_attempts.py … --apply` (removed `*_tracked.mp4`,
`*_tracked.pkl`, `resultsDLC/`; ~1.2 GB). The raw `*_80fps.mp4` is protected.

## Older experiments (legacy backend)

`SleapTracker(..., backend="legacy")` runs the uv-installed `sleap-track`
(`--tracking.tracker flow`) + `sleap-convert` — still no conda. Reading old
`.h5`/`.slp` with `Sleap_Tracks` is unchanged.
