# `utils_behavior.dataverse` — batch upload to a Dataverse dataset

A small wrapper around the
[DVUploader](https://github.com/GlobalDataverseCommunityConsortium/dataverse-uploader)
Java CLI for uploading a directory of files (typically videos) to a Dataverse
dataset.

It adds the things DVUploader doesn't do out of the box:

- a glob filter so you can target a subset (e.g. only `*_compressed.mp4`)
- a per-file size validation against Dataverse's 2.5 GiB hard cap (oversized
  files are listed and skipped, not aborted)
- a **skip-already-uploaded** check: queries the dataset before uploading
  and skips files whose names are already on the server, so you can re-run
  the same command after generating new videos and only the new ones go up
- an interactive manifest that lists every file to be uploaded and the
  resolved dataset **title** (fetched from the Dataverse API) so you can
  visually confirm you're pointing at the right dataset before saying yes
- per-file streaming progress (file count, byte percentage, transfer rate)
- a final summary listing failures, skipped-oversized files and
  skipped-already-on-server files so a long `screen` session never leaves
  you guessing what got through

## Setup

1. Grab the DVUploader jar (one-time):

   ```bash
   mkdir -p ~/dataverse-uploader
   curl -L -o ~/dataverse-uploader/DVUploader-v1.3.0-beta.jar \
       https://github.com/GlobalDataverseCommunityConsortium/dataverse-uploader/releases/download/v1.3.0-beta/DVUploader-v1.3.0-beta.jar
   ```

   You can also clone the repo and build it with `mvn clean compile assembly:single`
   if you'd rather run from source. Pass `--jar /path/to/your.jar` to the
   wrapper if it lives somewhere else.

2. Make sure a JRE is on your `PATH`. Java 11+ works (Java 21 is fine).

3. Create a `.env` file with your credentials. **Do not commit it** — the
   repo's `.gitignore` already covers `.env`.

   `DATAVERSE_SERVER` is the **installation base URL** (e.g.
   `https://dataverse.harvard.edu`), *not* the URL of your dataset's web
   page. The dataset is identified by `DATAVERSE_DOI`. If you accidentally
   paste a `…/dataset.xhtml?persistentId=…` URL, the script detects it,
   prints a warning, and auto-strips it down to the base — but cleaner to
   put the right value in directly.

   Example:

   ```
   DATAVERSE_SERVER=https://dataverse.your-institution.example
   DATAVERSE_API_KEY=00000000-0000-0000-0000-000000000000
   DATAVERSE_DOI=doi:10.5072/FK2/ABC123
   ```

   The script reads `./.env` from the current working directory by default;
   pass `--env-file /path/to/something.env` to point elsewhere. You can also
   pass `--server`, `--key`, `--doi` directly, or export them as env vars.

   Get an API key from your Dataverse profile page; the DOI is the dataset
   you're uploading **into** (not creating — create the dataset in the web
   UI first).

## Usage

Run as a module from the repo root:

```bash
python -m utils_behavior.dataverse.upload <directory> [options]
```

### What the prompt looks like

```
Scanned : /data/experiment_videos  (filter: '*_compressed.mp4', recurse: False)
Matched : 12 files, 4.20 GiB total
  To upload         : 4  (1.40 GiB)
  Already on server : 8  (will be skipped)

Server  : https://dataverse.your-institution.example
DOI     : doi:10.5072/FK2/ABC123
Dataset : "Larval locomotion under closed-loop optogenetic stimulation"

Files to upload (4):
  - fly09_trial02_compressed.mp4  [342.10 MiB]
  - fly09_trial03_compressed.mp4  [351.44 MiB]
  ...

Already on server — skipped (8):
  - fly01_trial01_compressed.mp4
  - fly01_trial02_compressed.mp4
  ...

Upload 4 files (skipping 8 already on server) to "Larval locomotion under closed-loop optogenetic stimulation" (doi:10.5072/FK2/ABC123)? [y/N]
```

If the dataset-info lookup fails (wrong DOI, wrong key, server unreachable),
the line becomes `Dataset : (could not fetch info — check server/key/doi)`,
the skip-existing check is silently disabled (everything is treated as new),
and the prompt falls back to just the DOI — handy as an early sanity check.

### Examples

Dry-run (just print the manifest, don't contact Dataverse):

```bash
python -m utils_behavior.dataverse.upload /data/experiment_videos --dry-run
```

Upload only the compressed copies in a directory that holds both originals
and compressed versions:

```bash
python -m utils_behavior.dataverse.upload /data/experiment_videos \
    --filter '*_compressed.mp4'
```

Re-run the same command after producing more videos — only the new files
are uploaded, the rest are skipped (filename match against the server's
file list):

```bash
python -m utils_behavior.dataverse.upload /data/experiment_videos \
    --filter '*_compressed.mp4'
```

Re-upload a single file (e.g. one whose content you regenerated). Because
skip-existing matches by filename, you need `--no-skip-existing` to force
the upload through:

```bash
python -m utils_behavior.dataverse.upload /data/experiment_videos \
    --filter 'fly42_trial07_compressed.mp4' --no-skip-existing
```

If you replace a file this way, delete the old version in the Dataverse
web UI first — uploads with a duplicate filename are rejected by the
server, and DVUploader will report the failure.

Recurse into subdirectories, auto-confirm (useful inside `screen`):

```bash
python -m utils_behavior.dataverse.upload /data/experiment_videos \
    --filter '*_compressed.mp4' --recurse -y
```

Use checksum verification against existing dataset entries (slower but
catches partial uploads from a previous run):

```bash
python -m utils_behavior.dataverse.upload /data/experiment_videos --verify
```

### Running inside `screen`

```bash
screen -S dataverse-upload
cd ~/utils_behavior
python -m utils_behavior.dataverse.upload /data/videos --filter '*_compressed.mp4' -y
# Ctrl-A D to detach. screen -r dataverse-upload to come back.
```

## Exit codes

- `0` — everything matched was uploaded successfully (no oversized, no failures)
- `1` — user aborted at the confirmation prompt
- `2` — at least one upload failed, or a configuration error (missing creds, etc.)
- `3` — all attempted uploads succeeded but some files were skipped because
  they exceeded the 2.5 GiB cap

## Filter cookbook

The `--filter` value is a [glob](https://docs.python.org/3/library/fnmatch.html),
matched against the filename (not the full path):

| What you want                         | Pattern                |
| ------------------------------------- | ---------------------- |
| All MP4s                              | `*.mp4`                |
| Only compressed videos                | `*_compressed.mp4`     |
| One specific file                     | `fly42_trial07.mp4`    |
| Files starting with a prefix          | `fly42_*`              |
| All files (default)                   | `*`                    |

For more powerful filtering (e.g. regex), the simplest workaround is to
move/symlink the wanted files into a separate directory and point the
script at that.

## Notes & gotchas

- **Per-file 2.5 GiB cap is hard.** Dataverse refuses larger files. Compress
  or split before uploading, or arrange external storage with your Dataverse
  admin. The wrapper lists oversized files in the manifest and the final
  summary.
- **Skip-existing is name-based.** The wrapper queries the dataset's file
  list before uploading and skips files whose names already exist on the
  server. It does *not* compare contents — if you regenerate a file with the
  same name, it will be skipped silently unless you pass `--no-skip-existing`
  (and even then you'll need to delete the old copy in the web UI first).
  Use `--verify` to additionally have DVUploader checksum-compare on its
  side — slower but catches partial uploads from a previous run.
- **Direct upload vs. via-server.** This wrapper uses DVUploader's default
  (direct upload) plus `-singlefile`. If your dataset doesn't have direct
  upload enabled, you'll see errors — talk to your Dataverse admin or pass
  the underlying flag through (the wrapper doesn't expose `-uploadviaserver`
  yet; add it to `extra` in `stream_upload` if you need it).
- **Success detection is conservative.** DVUploader doesn't always exit
  non-zero on partial failures, so we additionally scan its output for error
  tokens. If a file is flagged as failed but actually went through, a re-run
  with `--verify` will confirm it's already there and skip it.
