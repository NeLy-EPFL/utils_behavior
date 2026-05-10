"""Batch-rename files in a Dataverse dataset via the native API.

Fetches the file listing for a dataset, applies a renaming transformation,
shows a manifest of old→new names, asks for confirmation, then PATCHes each
file's ``label`` (display name) while preserving all other metadata.

Supported rename modes (combinable, applied left to right):
  --replace OLD NEW      simple substring replacement
  --regex PATTERN REPL   regex substitution (re.sub, first match per call)
  --add-prefix PREFIX    prepend PREFIX to every matched filename
  --add-suffix SUFFIX    insert SUFFIX just before the file extension

Use ``--filter GLOB`` to restrict which remote files are processed.
Use ``--dry-run`` to preview changes without touching the server.

Run with ``python -m utils_behavior.dataverse.rename --help``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import PurePosixPath
from typing import Callable
from urllib.parse import urlparse

import requests

ENV_KEYS = {
    "server": "DATAVERSE_SERVER",
    "key": "DATAVERSE_API_KEY",
    "doi": "DATAVERSE_DOI",
}


# ---------------------------------------------------------------------------
# Shared helpers (mirrors upload.py style)
# ---------------------------------------------------------------------------


def normalize_server_url(url: str) -> tuple[str, str | None]:
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


def load_env_file(path: str | os.PathLike) -> None:
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def confirm(prompt: str) -> bool:
    try:
        ans = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return ans in ("y", "yes")


# ---------------------------------------------------------------------------
# Dataverse API helpers
# ---------------------------------------------------------------------------


@dataclass
class RemoteFile:
    file_id: int
    label: str  # current display name / filename
    description: str
    categories: list[str]
    restrict: bool
    directory_label: str  # subdirectory path within the dataset (may be "")


def fetch_remote_files(
    server: str, key: str, doi: str, timeout: float = 60.0
) -> list[RemoteFile]:
    """Return the file listing for the latest version of the dataset."""
    url = f"{server.rstrip('/')}/api/datasets/:persistentId/versions/:latest/files"
    files: list[RemoteFile] = []
    offset = 0
    limit = 1000
    while True:
        r = requests.get(
            url,
            params={"persistentId": doi, "limit": limit, "offset": offset},
            headers={"X-Dataverse-key": key},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        for entry in data:
            df = entry.get("dataFile", {})
            fid = df.get("id")
            if fid is None:
                continue
            files.append(
                RemoteFile(
                    file_id=fid,
                    label=entry.get("label") or df.get("filename", ""),
                    description=entry.get("description", ""),
                    categories=entry.get("categories") or [],
                    restrict=bool(entry.get("restrict", False)),
                    directory_label=entry.get("directoryLabel", ""),
                )
            )
        if len(data) < limit:
            break
        offset += limit
    return files


def rename_file(
    server: str,
    key: str,
    rf: RemoteFile,
    new_label: str,
    timeout: float = 60.0,
) -> None:
    """Update only the label of a remote file, preserving all other metadata."""
    url = f"{server.rstrip('/')}/api/files/{rf.file_id}/metadata"
    payload: dict = {"label": new_label}
    if rf.description:
        payload["description"] = rf.description
    if rf.categories:
        payload["categories"] = rf.categories
    payload["restrict"] = rf.restrict
    if rf.directory_label:
        payload["directoryLabel"] = rf.directory_label

    r = requests.post(
        url,
        headers={"X-Dataverse-key": key},
        files={"jsonData": (None, json.dumps(payload))},
        timeout=timeout,
    )
    r.raise_for_status()
    # Some Dataverse versions return an empty or non-JSON body on success.
    if r.text.strip():
        try:
            resp = r.json()
            if resp.get("status") not in ("OK", None):
                raise RuntimeError(
                    f"Unexpected response status: {resp.get('status')!r}"
                )
        except json.JSONDecodeError:
            pass  # Non-JSON body after a successful HTTP status — treat as OK.


# ---------------------------------------------------------------------------
# Renaming logic
# ---------------------------------------------------------------------------


@dataclass
class RenamePlan:
    remote: RemoteFile
    new_label: str

    @property
    def changed(self) -> bool:
        return self.remote.label != self.new_label


def build_transform(args: argparse.Namespace) -> Callable[[str], str]:
    """Return a function that applies all requested rename operations in order."""
    ops: list[Callable[[str], str]] = []

    for old, new in args.replace or []:
        ops.append(lambda s, o=old, n=new: s.replace(o, n))

    for pattern, repl in args.regex or []:
        compiled = re.compile(pattern)
        ops.append(lambda s, p=compiled, r=repl: p.sub(r, s))

    if args.add_prefix:
        prefix = args.add_prefix
        ops.append(lambda s, p=prefix: p + s)

    if args.add_suffix:
        suffix = args.add_suffix

        def _add_suffix(s: str, suf: str = suffix) -> str:
            pp = PurePosixPath(s)
            # Insert suffix before the extension; if no extension, append.
            return pp.stem + suf + pp.suffix

        ops.append(_add_suffix)

    def transform(name: str) -> str:
        for op in ops:
            name = op(name)
        return name

    return transform


def build_plans(
    remote_files: list[RemoteFile],
    pattern: str,
    transform: Callable[[str], str],
) -> list[RenamePlan]:
    plans: list[RenamePlan] = []
    for rf in remote_files:
        if not fnmatch(rf.label, pattern):
            continue
        plans.append(RenamePlan(remote=rf, new_label=transform(rf.label)))
    return plans


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def print_manifest(
    plans: list[RenamePlan],
    server: str | None,
    doi: str | None,
    dataset_title: str | None,
    pattern: str,
) -> None:
    matched = len(plans)
    changed = sum(1 for p in plans if p.changed)
    unchanged = matched - changed

    print()
    print(
        f"Dataset : {dataset_title!r}"
        if dataset_title
        else f"DOI     : {doi or '(missing)'}"
    )
    print(f"Server  : {server or '(missing)'}")
    print(f"Filter  : '{pattern}'")
    print(
        f"Matched : {matched} files  →  {changed} will be renamed, {unchanged} unchanged"
    )

    if changed:
        print()
        print(f"Renames ({changed}):")
        for p in plans:
            if p.changed:
                loc = (
                    f"  [{p.remote.directory_label}/]"
                    if p.remote.directory_label
                    else ""
                )
                print(f"  {p.remote.label!r:50s}  →  {p.new_label!r}{loc}")

    if unchanged:
        print()
        print(f"Unchanged ({unchanged}) — will be skipped:")
        for p in plans:
            if not p.changed:
                print(f"  {p.remote.label!r}")

    print()


def print_summary(
    successes: list[RenamePlan],
    failures: list[tuple[RenamePlan, str]],
    elapsed: float,
) -> None:
    print()
    print("=" * 70)
    print(
        f"Done in {elapsed:.1f}s.  "
        f"Renamed: {len(successes)}   Failed: {len(failures)}"
    )
    if failures:
        print()
        print("FAILED:")
        for plan, msg in failures:
            print(f"  - {plan.remote.label!r}  →  {plan.new_label!r}  ({msg})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="utils_behavior.dataverse.rename",
        description=(
            "Batch-rename files in a Dataverse dataset. "
            "Fetches the remote file list, applies one or more rename "
            "transformations, shows a manifest, then renames via the API."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--filter",
        default="*",
        metavar="GLOB",
        help="Glob matched against remote filenames to select files, e.g. 'fly*.mp4'",
    )
    p.add_argument(
        "--replace",
        nargs=2,
        metavar=("OLD", "NEW"),
        action="append",
        help="Replace OLD with NEW in the filename (repeatable)",
    )
    p.add_argument(
        "--regex",
        nargs=2,
        metavar=("PATTERN", "REPLACEMENT"),
        action="append",
        help="Apply re.sub(PATTERN, REPLACEMENT, filename) (repeatable)",
    )
    p.add_argument(
        "--add-prefix",
        metavar="PREFIX",
        help="Prepend PREFIX to every matched filename",
    )
    p.add_argument(
        "--add-suffix",
        metavar="SUFFIX",
        help="Insert SUFFIX before the file extension (e.g. '_compressed')",
    )
    p.add_argument(
        "--env-file",
        metavar="PATH",
        help="Path to a .env file (defaults to ./.env in cwd, if present)",
    )
    p.add_argument("--server", help=f"Dataverse server URL (env: {ENV_KEYS['server']})")
    p.add_argument("--key", help=f"Dataverse API key (env: {ENV_KEYS['key']})")
    p.add_argument("--doi", help=f"Target dataset DOI (env: {ENV_KEYS['doi']})")
    p.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the manifest and exit without modifying anything on the server",
    )
    return p.parse_args(argv)


def resolve_credentials(
    args: argparse.Namespace,
) -> tuple[str | None, str | None, str | None]:
    from pathlib import Path

    env_file = args.env_file if args.env_file else Path.cwd() / ".env"
    load_env_file(env_file)
    return (
        args.server or os.environ.get(ENV_KEYS["server"]),
        args.key or os.environ.get(ENV_KEYS["key"]),
        args.doi or os.environ.get(ENV_KEYS["doi"]),
    )


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

    if not (args.replace or args.regex or args.add_prefix or args.add_suffix):
        print(
            "No rename operation specified. "
            "Use --replace, --regex, --add-prefix, or --add-suffix.",
            file=sys.stderr,
        )
        return 2

    missing = [
        name
        for name, val in (("server", server), ("key", key), ("doi", doi))
        if not val
    ]
    if missing:
        print(
            f"Missing credentials: {', '.join(missing)}. "
            f"Provide via --{missing[0].replace('_', '-')} flag, "
            f"env vars ({', '.join(ENV_KEYS.values())}), or a .env file.",
            file=sys.stderr,
        )
        return 2

    print(f"Fetching file list for {doi} ...")
    try:
        remote_files = fetch_remote_files(server, key, doi)
    except requests.HTTPError as e:
        print(f"Failed to fetch file list: {e}", file=sys.stderr)
        return 2

    if not remote_files:
        print("Dataset has no files.")
        return 0

    transform = build_transform(args)
    plans = build_plans(remote_files, args.filter, transform)

    if not plans:
        print(f"No files matched the filter '{args.filter}'.")
        return 0

    # Best-effort dataset title lookup (reuse the listing data; title isn't in it,
    # so fetch from the dataset metadata endpoint).
    dataset_title: str | None = None
    try:
        r = requests.get(
            f"{server.rstrip('/')}/api/datasets/:persistentId/",
            params={"persistentId": doi},
            headers={"X-Dataverse-key": key},
            timeout=30,
        )
        r.raise_for_status()
        for f in (
            r.json()
            .get("data", {})
            .get("latestVersion", {})
            .get("metadataBlocks", {})
            .get("citation", {})
            .get("fields", [])
        ):
            if f.get("typeName") == "title" and isinstance(f.get("value"), str):
                dataset_title = f["value"]
                break
    except Exception:
        pass

    print_manifest(plans, server, doi, dataset_title, args.filter)

    if args.dry_run:
        print("Dry run — no changes made.")
        return 0

    to_rename = [p for p in plans if p.changed]
    if not to_rename:
        print("Nothing to rename — all matched files already have the target name.")
        return 0

    target = f'"{dataset_title}" ({doi})' if dataset_title else doi
    prompt = f"Rename {len(to_rename)} file(s) in {target}? [y/N] "
    if not args.yes and not confirm(prompt):
        print("Aborted.")
        return 1

    successes: list[RenamePlan] = []
    failures: list[tuple[RenamePlan, str]] = []
    started = time.time()

    print()
    print(f"Renaming {len(to_rename)} file(s).")
    print("-" * 70)
    for idx, plan in enumerate(to_rename, start=1):
        prefix = f"[{idx}/{len(to_rename)}]"
        print(f"{prefix} {plan.remote.label!r}  →  {plan.new_label!r}", flush=True)
        try:
            rename_file(server, key, plan.remote, plan.new_label)
            print(f"  -> OK")
            successes.append(plan)
        except Exception as e:
            msg = str(e)
            print(f"  -> FAILED: {msg}")
            failures.append((plan, msg))

    print_summary(successes, failures, time.time() - started)
    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
