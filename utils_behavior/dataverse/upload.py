"""Batch-upload videos to a Dataverse dataset via DVUploader.

Wraps the DVUploader Java CLI with: glob filtering, a 2.5 GiB-per-file
hard cap (Dataverse limit), a manifest + confirmation step, per-file
streaming progress, and a final summary of failures and skipped files.

Run with ``python -m utils_behavior.dataverse.upload --help``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests

from utils_behavior.dataverse.rename import fetch_remote_files, rename_file

DEFAULT_JAR = Path.home() / "dataverse-uploader" / "DVUploader-v1.3.0-beta.jar"
SIZE_LIMIT_BYTES = int(2.5 * 1024**3)  # 2.5 GiB — Dataverse per-file limit
SIZE_LIMIT_HUMAN = "2.5 GiB"

ENV_KEYS = {
    "server": "DATAVERSE_SERVER",
    "key": "DATAVERSE_API_KEY",
    "doi": "DATAVERSE_DOI",
}

# Tokens in DVUploader's stdout that indicate a per-file failure even when
# the JVM exits 0. Conservative — better to flag a maybe-failed file for
# the user to re-check than to silently miss it.
FAIL_MARKERS = (
    "could not",
    "failed to upload",
    "error uploading",
    "exception",
    "unauthorized",
    "forbidden",
    " 401",
    " 403",
    " 500",
)


@dataclass
class FileEntry:
    path: Path
    size: int


def human_size(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"


def normalize_server_url(url: str) -> tuple[str, str | None]:
    """Catch the common mistake of pasting a dataset detail URL.

    Strips path+query and returns ``scheme://host`` if the URL clearly came
    from a browser address bar on a dataset/collection page. Any path or
    query we don't specifically recognise is left alone so sub-path-hosted
    Dataverse installations still work.

    Returns ``(normalized_url, warning_message_or_None)``.
    """
    parsed = urlparse(url)
    if not (parsed.scheme and parsed.netloc):
        return url, None
    looks_like_dataset_url = (
        "dataset.xhtml" in parsed.path.lower() or "persistentid" in parsed.query.lower()
    )
    if looks_like_dataset_url:
        fixed = f"{parsed.scheme}://{parsed.netloc}"
        warning = (
            "DATAVERSE_SERVER looks like a dataset URL, not the installation base.\n"
            f"  Got     : {url}\n"
            f"  Using   : {fixed}\n"
            "  Action  : update DATAVERSE_SERVER in your .env so it's just "
            f"'{fixed}' (no path, no query)."
        )
        return fixed, warning
    return url.rstrip("/"), None


def load_env_file(path: Path) -> None:
    """Populate os.environ from KEY=VALUE lines, without overriding existing vars."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def collect_files(root: Path, pattern: str, recurse: bool) -> list[Path]:
    if root.is_file():
        return [root] if fnmatch(root.name, pattern) else []
    walker = root.rglob(pattern) if recurse else root.glob(pattern)
    return sorted(p for p in walker if p.is_file())


@dataclass
class DatasetInfo:
    """Result of a Dataverse dataset lookup."""

    title: str | None = None
    filenames: frozenset[str] = frozenset()
    fetched: bool = False  # True iff the API call succeeded


def fetch_dataset_info(
    server: str, key: str, doi: str, timeout: float = 30.0
) -> DatasetInfo:
    """Best-effort lookup of dataset title and the filenames in its latest version.

    Returns DatasetInfo() with fetched=False on any failure (network, auth,
    schema surprise) so callers can fall back to a no-skip / no-title path.
    """
    url = f"{server.rstrip('/')}/api/datasets/:persistentId/"
    try:
        r = requests.get(
            url,
            params={"persistentId": doi},
            headers={"X-Dataverse-key": key},
            timeout=timeout,
        )
        r.raise_for_status()
        latest = r.json()["data"]["latestVersion"]
    except (requests.RequestException, KeyError, ValueError):
        return DatasetInfo()

    title = None
    for f in latest.get("metadataBlocks", {}).get("citation", {}).get("fields", []):
        if f.get("typeName") == "title":
            v = f.get("value")
            if isinstance(v, str):
                title = v
                break

    names: set[str] = set()
    for entry in latest.get("files", []):
        name = entry.get("dataFile", {}).get("filename") or entry.get("label")
        if name:
            names.add(name)

    return DatasetInfo(title=title, filenames=frozenset(names), fetched=True)


def classify(files: Iterable[Path]) -> tuple[list[FileEntry], list[FileEntry]]:
    ok: list[FileEntry] = []
    oversized: list[FileEntry] = []
    for f in files:
        entry = FileEntry(f, f.stat().st_size)
        (oversized if entry.size > SIZE_LIMIT_BYTES else ok).append(entry)
    return ok, oversized


def partition_against_remote(
    ok: list[FileEntry], remote_names: frozenset[str]
) -> tuple[list[FileEntry], list[FileEntry]]:
    """Split locally-uploadable files into (to_upload, already_on_server)."""
    to_upload: list[FileEntry] = []
    already: list[FileEntry] = []
    for e in ok:
        (already if e.path.name in remote_names else to_upload).append(e)
    return to_upload, already


def _mask_secret(line: str, secret: str) -> str:
    if secret and secret in line:
        return line.replace(secret, "***")
    return line


def stream_upload(
    jar: Path,
    server: str,
    key: str,
    doi: str,
    file: Path,
    extra_args: list[str],
) -> tuple[int, str]:
    """Run DVUploader for a single file, streaming output line-by-line."""
    cmd = [
        "java",
        "-jar",
        str(jar),
        f"-server={server}",
        f"-key={key}",
        f"-did={doi}",
        "-singlefile",
        *extra_args,
        str(file),
    ]
    captured: list[str] = []
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        captured.append(line)
        # Filter the noisy ASCII banner DVUploader prints every invocation,
        # and mask the API key (DVUploader echoes it in plaintext).
        if not line:
            continue
        if any(tok in line for tok in ("TTTTT", "Texas", "Digital", "Library")):
            continue
        # DVUploader's argv echo line still leaks part of the key — drop it.
        if "-key=" in line:
            continue
        print(f"    | {_mask_secret(line, key)}", flush=True)
    proc.wait()
    return proc.returncode, "\n".join(captured)


def upload_succeeded(returncode: int, output: str) -> bool:
    if returncode != 0:
        return False
    lower = output.lower()
    return not any(m in lower for m in FAIL_MARKERS)


def confirm(prompt: str) -> bool:
    try:
        ans = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return ans in ("y", "yes")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="utils_behavior.dataverse.upload",
        description=(
            "Batch upload to a Dataverse dataset via DVUploader, with glob "
            "filtering, size validation, per-file progress, and a final "
            "summary of failures/skipped files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("directory", type=Path, help="Directory containing files to upload")
    p.add_argument(
        "--filter",
        default="*",
        help="Glob applied to filenames, e.g. '*_compressed.mp4' or 'fly42_*.mp4'",
    )
    p.add_argument("--recurse", action="store_true", help="Recurse into subdirectories")
    p.add_argument(
        "--jar", type=Path, default=DEFAULT_JAR, help="Path to DVUploader jar"
    )
    p.add_argument(
        "--env-file",
        type=Path,
        help="Path to a .env file (defaults to ./.env in the cwd, if present)",
    )
    p.add_argument("--server", help=f"Dataverse server URL (env: {ENV_KEYS['server']})")
    p.add_argument("--key", help=f"Dataverse API key (env: {ENV_KEYS['key']})")
    p.add_argument("--doi", help=f"Target dataset DOI (env: {ENV_KEYS['doi']})")
    p.add_argument(
        "--verify",
        action="store_true",
        help="Pass -verify to DVUploader (checksum compare with existing dataset entries)",
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help=(
            "Re-upload files even if a same-named file is already on the server "
            "(default: query the dataset and skip same-named files)"
        ),
    )
    p.add_argument(
        "-y", "--yes", action="store_true", help="Skip the confirmation prompt"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the manifest and exit without contacting Dataverse",
    )
    p.add_argument(
        "--add-prefix",
        default="",
        metavar="PREFIX",
        help="Prepend PREFIX to each uploaded filename on the server (e.g. 'Ballscents-')",
    )
    p.add_argument(
        "--add-suffix",
        default="",
        metavar="SUFFIX",
        help="Insert SUFFIX before the file extension on the server (e.g. '_v2')",
    )
    return p.parse_args(argv)


def resolve_credentials(
    args: argparse.Namespace,
) -> tuple[str | None, str | None, str | None]:
    env_file = args.env_file if args.env_file else Path.cwd() / ".env"
    load_env_file(env_file)
    return (
        args.server or os.environ.get(ENV_KEYS["server"]),
        args.key or os.environ.get(ENV_KEYS["key"]),
        args.doi or os.environ.get(ENV_KEYS["doi"]),
    )


def print_manifest(
    directory: Path,
    pattern: str,
    recurse: bool,
    to_upload: list[FileEntry],
    already_present: list[FileEntry],
    oversized: list[FileEntry],
    server: str | None,
    doi: str | None,
    info: DatasetInfo,
    skip_existing_active: bool,
) -> None:
    total_n = len(to_upload) + len(already_present) + len(oversized)
    total_bytes = (
        sum(e.size for e in to_upload)
        + sum(e.size for e in already_present)
        + sum(e.size for e in oversized)
    )
    print()
    print(f"Scanned : {directory}  (filter: '{pattern}', recurse: {recurse})")
    print(f"Matched : {total_n} files, {human_size(total_bytes)} total")
    print(
        f"  To upload         : {len(to_upload)}  "
        f"({human_size(sum(e.size for e in to_upload))})"
    )
    if already_present:
        print(f"  Already on server : {len(already_present)}  (will be skipped)")
    if oversized:
        print(
            f"  Oversized         : {len(oversized)}  "
            f"(>{SIZE_LIMIT_HUMAN}, will be skipped)"
        )
    print()
    print(f"Server  : {server or '(missing)'}")
    print(f"DOI     : {doi or '(missing)'}")
    if info.title:
        print(f'Dataset : "{info.title}"')
    elif server and doi and not info.fetched:
        print("Dataset : (could not fetch info — check server/key/doi)")
    if server and doi and info.fetched and not skip_existing_active:
        print("        (skip-existing disabled by --no-skip-existing)")

    if to_upload:
        print()
        print(f"Files to upload ({len(to_upload)}):")
        for e in to_upload:
            print(f"  - {e.path.name}  [{human_size(e.size)}]")

    if already_present:
        print()
        print(f"Already on server — skipped ({len(already_present)}):")
        for e in already_present:
            print(f"  - {e.path.name}")

    if oversized:
        print()
        print(f"Oversized — skipped (>{SIZE_LIMIT_HUMAN}, {len(oversized)}):")
        for e in oversized:
            print(f"  - {e.path.name}  [{human_size(e.size)}]")
    print()


def print_summary(
    successes: list[tuple[FileEntry, float]],
    failures: list[tuple[FileEntry, str]],
    skipped_oversized: list[FileEntry],
    skipped_existing: list[FileEntry],
    elapsed: float,
) -> None:
    print()
    print("=" * 70)
    print(
        f"Done in {elapsed:.1f}s.  "
        f"OK: {len(successes)}   Failed: {len(failures)}   "
        f"Skipped (oversized): {len(skipped_oversized)}   "
        f"Skipped (already on server): {len(skipped_existing)}"
    )
    if failures:
        print()
        print("FAILED:")
        for entry, _ in failures:
            print(f"  - {entry.path}  ({human_size(entry.size)})")
    if skipped_oversized:
        print()
        print(f"SKIPPED (>{SIZE_LIMIT_HUMAN}):")
        for entry in skipped_oversized:
            print(f"  - {entry.path}  ({human_size(entry.size)})")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    server, key, doi = resolve_credentials(args)

    if server:
        server, warning = normalize_server_url(server)
        if warning:
            print()
            print("WARNING:")
            for line in warning.splitlines():
                print(f"  {line}")
            print()

    if not args.directory.exists():
        print(f"Directory not found: {args.directory}", file=sys.stderr)
        return 2

    print(
        f"Scanning {args.directory} (filter: {args.filter}, recurse: {args.recurse})..."
    )
    files = collect_files(args.directory, args.filter, args.recurse)
    if not files:
        print("No files matched the filter.")
        return 0
    ok, oversized = classify(files)

    info = (
        fetch_dataset_info(server, key, doi)
        if (server and key and doi)
        else DatasetInfo()
    )
    skip_existing_active = info.fetched and not args.no_skip_existing
    if skip_existing_active:
        to_upload, already_present = partition_against_remote(ok, info.filenames)
    else:
        to_upload, already_present = ok, []

    print_manifest(
        args.directory,
        args.filter,
        args.recurse,
        to_upload,
        already_present,
        oversized,
        server,
        doi,
        info,
        skip_existing_active,
    )

    if args.dry_run:
        print("Dry run — exiting without uploading.")
        return 0

    if not to_upload:
        if already_present:
            print("Nothing to upload — all matched files are already on the server.")
        else:
            print("Nothing to upload.")
        return 0

    missing = [
        name
        for name, val in (("server", server), ("key", key), ("doi", doi))
        if not val
    ]
    if missing:
        print(
            f"Missing credentials: {', '.join(missing)}. "
            f"Provide via --{missing[0]} flag, env vars ({', '.join(ENV_KEYS.values())}), "
            "or a .env file.",
            file=sys.stderr,
        )
        return 2

    if not args.jar.exists():
        print(
            f"DVUploader jar not found at {args.jar}. "
            "Download it from the dataverse-uploader releases page or pass --jar.",
            file=sys.stderr,
        )
        return 2

    target = f'"{info.title}" ({doi})' if info.title else doi
    skipped_msg = (
        f" (skipping {len(already_present)} already on server)"
        if already_present
        else ""
    )
    prompt_msg = f"Upload {len(to_upload)} files{skipped_msg} to {target}? [y/N] "
    if not args.yes and not confirm(prompt_msg):
        print("Aborted.")
        return 1

    extra = ["-verify"] if args.verify else []
    successes: list[tuple[FileEntry, float]] = []
    failures: list[tuple[FileEntry, str]] = []
    started = time.time()
    total_bytes = sum(e.size for e in to_upload)
    bytes_done = 0

    print()
    print(f"Uploading {len(to_upload)} files ({human_size(total_bytes)}).")
    print("-" * 70)
    for idx, entry in enumerate(to_upload, start=1):
        prefix = f"[{idx}/{len(to_upload)}]"
        print(f"{prefix} {entry.path.name}  ({human_size(entry.size)})", flush=True)
        t0 = time.time()
        rc, output = stream_upload(args.jar, server, key, doi, entry.path, extra)
        dt = time.time() - t0
        bytes_done += entry.size
        pct = bytes_done / total_bytes * 100
        if upload_succeeded(rc, output):
            speed = entry.size / dt if dt > 0 else 0
            print(
                f"  -> OK in {dt:.1f}s ({human_size(speed)}/s)  "
                f"[{pct:.1f}% done, {human_size(bytes_done)}/{human_size(total_bytes)}]"
            )
            successes.append((entry, dt))
        else:
            print(f"  -> FAILED in {dt:.1f}s  (exit={rc})")
            failures.append((entry, output))

    print_summary(
        successes,
        failures,
        list(oversized),
        list(already_present),
        time.time() - started,
    )

    if (args.add_prefix or args.add_suffix) and successes:
        uploaded_names = {e.path.name for e, _ in successes}
        print()
        print(f"Renaming {len(uploaded_names)} uploaded file(s) on server...")
        print("-" * 70)
        remote_files = fetch_remote_files(server, key, doi)
        rename_ok = 0
        rename_fail = 0
        for rf in remote_files:
            if rf.label not in uploaded_names:
                continue
            dot = rf.label.rfind(".")
            if dot > 0:
                stem, ext = rf.label[:dot], rf.label[dot:]
            else:
                stem, ext = rf.label, ""
            new_label = args.add_prefix + stem + args.add_suffix + ext
            try:
                rename_file(server, key, rf, new_label)
                print(f"  {rf.label!r}  →  {new_label!r}")
                rename_ok += 1
            except Exception as exc:
                print(f"  FAILED rename {rf.label!r}: {exc}")
                rename_fail += 1
        print(f"Renamed: {rename_ok}   Failed: {rename_fail}")

    if failures:
        return 2
    if oversized:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
