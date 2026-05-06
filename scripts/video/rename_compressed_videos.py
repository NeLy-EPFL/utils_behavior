#!/usr/bin/env python3
"""
Rename TNT_Screen_RawGrids compressed videos to match Simplified Nicknames
from the new Region_map_260506.csv, sanitising names for Dataverse upload.

Sanitisation rules applied to every target name:
  - Greek letters  → ASCII name  (α→alpha, β→beta, γ→gamma, ′→prime, etc.)
  - Spaces         → hyphen  (-)
  - Parentheses    → removed
  - Apostrophe / curly-quote / prime → removed after Greek replacement

Usage:
    python rename_compressed_videos.py           # dry run (default)
    python rename_compressed_videos.py --execute # actually rename files
"""

import argparse
import csv
import re
import sys
from pathlib import Path

VIDEO_DIR = Path("/mnt/upramdya_data/MD/TNT_Screen_RawGrids")
NEW_MAP = Path("/mnt/upramdya_data/MD/Region_map_260506.csv")
OLD_MAP = Path("/mnt/upramdya_data/MD/Region_map_250908.csv")
SUFFIX = "_grid_compressed.mp4"

NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")

# Greek letter substitution table (order matters: longer sequences first if needed)
GREEK_MAP = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "ο": "omicron",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
    # prime / apostrophe variants that appear in biological names
    "\u2032": "p",  # ′  (prime) → p   e.g. α′ becomes alphap
    "\u2019": "",  # '  (right single quotation mark) → removed
    "'": "p",  # ASCII apostrophe used as prime → p
}

# Manual overrides: stem (as it appears in the filename) → target Simplified Nickname.
# Use the raw CSV nickname here; sanitize_for_dataverse() is applied automatically.
# Set value to None to keep the file as-is (no rename).
MANUAL_OVERRIDES: dict[str, str | None] = {
    # E-PG-2 exists as a distinct file; keep it (the CSV maps it to E-PG which conflicts)
    "E-PG-2": None,
    # These 5 files were named from the old Nickname column instead of Simplified Nickname.
    # Their correct target is the Simplified Nickname — but a file with that name already
    # exists, so they will be flagged as CONFLICT (duplicates to resolve manually).
    "MBON-10-GaL4(MBON-\u03b2\u20321)": "MBON-\u03b2\u20321",  # → MBON-betap1
    "MBON-12-GaL4(MBON-\u03b32\u03b1\u20321)": (
        "MBON-\u03b32\u03b1\u20321"
    ),  # → MBON-gamma2alphap1
    "MBON-13-GaL4(MBON-\u03b1\u20322)": "MBON-\u03b1\u20322",  # → MBON-alphap2
    "MBON-14-GaL4(MBON-\u03b13)": "MBON-\u03b13",  # → MBON-alpha3
    "MBON-22-GaL4(MBON-calyx)": "MBON-calyx",  # → MBON-calyx
}


def sanitize_for_dataverse(name: str) -> str:
    """Return a fully ASCII, Dataverse-safe version of a name.

    Steps (in order):
    1. Replace Greek letters with their ASCII names
    2. Replace prime/apostrophe characters
    3. Replace spaces with hyphens
    4. Remove parentheses
    5. Strip any remaining non-ASCII characters (safety net)
    """
    s = name
    for char, replacement in GREEK_MAP.items():
        s = s.replace(char, replacement)
    s = s.replace(" ", "-")
    s = s.replace("(", "").replace(")", "")
    # Safety net: drop anything else that slipped through
    s = NON_ASCII_RE.sub("", s)
    return s


def remaining_issues(name: str) -> list[str]:
    """Return a list of problem descriptions still in *name* after sanitisation."""
    issues = []
    if " " in name:
        issues.append("space")
    if "(" in name or ")" in name:
        issues.append("parentheses")
    non_ascii = NON_ASCII_RE.findall(name)
    if non_ascii:
        issues.append(f"non-ASCII: {''.join(sorted(set(non_ascii)))}")
    return issues


def load_map(csv_path: Path) -> tuple:
    """Load a Region_map CSV. Returns (rows, by_genotype) where by_genotype maps
    Genotype → row dict."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: v.strip() for k, v in row.items()})

    by_genotype = {r["Genotype"]: r for r in rows}
    return rows, by_genotype


def build_lookup(rows: list, new_rows_by_genotype: dict) -> tuple:
    """Build lookup dicts from a set of CSV rows (can be old or new map).

    Returns:
        by_simplified  : Simplified Nickname → new Simplified Nickname
        by_old_sn      : Old Simplified Nickname → new Simplified Nickname
        by_nickname    : Nickname → new Simplified Nickname (for partial matching)
    """
    by_simplified: dict[str, str] = {}
    by_old_sn: dict[str, str] = {}
    by_nickname: dict[str, str] = {}

    for row in rows:
        geno = row["Genotype"]
        sn = row["Simplified Nickname"]
        osn = row["Old Simplified Nickname"]
        nick = row["Nickname"]

        # Resolve what the NEW Simplified Nickname is for this genotype
        new_row = new_rows_by_genotype.get(geno)
        if new_row:
            target_sn = new_row["Simplified Nickname"]
        else:
            target_sn = sn  # fallback: use the same row's value

        def _add(d, key, val):
            if key and key not in d:
                d[key] = val
            if key and key.lower() not in d:
                d[key.lower()] = val

        _add(by_simplified, sn, target_sn)
        _add(by_old_sn, osn, target_sn)
        _add(by_nickname, nick, target_sn)

    return by_simplified, by_old_sn, by_nickname


def find_match(
    stem: str,
    by_simplified,
    by_old_sn,
    by_nickname,
    by_simplified_new,
    by_old_sn_new,
    by_nickname_new,
):
    """Try to find a mapping for a video stem → new Simplified Nickname.
    Checks new map first, then falls back to old map lookups.
    """
    lookups = [
        # (dict, description)
        (by_simplified_new, "new map simplified nickname"),
        (by_old_sn_new, "new map old-simplified-nickname"),
        (by_nickname_new, "new map nickname"),
        (by_simplified, "old map simplified nickname"),
        (by_old_sn, "old map old-simplified-nickname"),
        (by_nickname, "old map nickname"),
    ]

    for d, desc in lookups:
        if stem in d:
            return d[stem], f"matched via {desc}"
        if stem.lower() in d:
            return d[stem.lower()], f"matched via {desc} (case-insensitive)"

    # Partial match: stem appears at the start of a nickname key
    # e.g. stem="86830" matches nickname key "86830 (pC1-SS1)"
    for d, desc in lookups:
        for key, val in d.items():
            if key.startswith(stem + " ") or key.startswith(stem + "("):
                return val, f"partial match via {desc} (key='{key}')"

    return None, "NO MATCH"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually rename files (default: dry run)",
    )
    parser.add_argument(
        "--skip-non-ascii",
        action="store_true",
        help="Skip files where new name still has non-ASCII chars",
    )
    args = parser.parse_args()

    if not VIDEO_DIR.exists():
        print(f"ERROR: Video directory not found: {VIDEO_DIR}")
        sys.exit(1)
    if not NEW_MAP.exists():
        print(f"ERROR: Region map not found: {NEW_MAP}")
        sys.exit(1)

    # Load both maps; build lookups cross-referenced via Genotype
    new_rows, new_by_geno = load_map(NEW_MAP)
    old_rows, old_by_geno = load_map(OLD_MAP)

    # Lookups against new map (where target is always new map's Simplified Nickname)
    by_sn_new, by_osn_new, by_nick_new = build_lookup(new_rows, new_by_geno)
    # Lookups against old map, but resolving target via new map
    by_sn_old, by_osn_old, by_nick_old = build_lookup(old_rows, new_by_geno)

    # Get all compressed videos
    compressed = sorted(VIDEO_DIR.glob(f"*{SUFFIX}"))
    print(f"Found {len(compressed)} compressed videos in {VIDEO_DIR}\n")

    renames = []  # (old_path, new_path, old_stem, new_stem, how)
    already_ok = []  # stems that need no change
    no_match = []  # stems with no CSV match

    for vid in compressed:
        stem = vid.name[: -len(SUFFIX)]

        # Check manual overrides first
        if stem in MANUAL_OVERRIDES:
            override = MANUAL_OVERRIDES[stem]
            if override is None:
                already_ok.append(stem)
                continue
            new_sn, how = override, "manual override"
        else:
            new_sn, how = find_match(
                stem,
                by_sn_old,
                by_osn_old,
                by_nick_old,
                by_sn_new,
                by_osn_new,
                by_nick_new,
            )

        if new_sn is None:
            no_match.append((stem, vid))
            continue

        safe_sn = sanitize_for_dataverse(new_sn)
        target_path = VIDEO_DIR / f"{safe_sn}{SUFFIX}"

        if stem == safe_sn:
            already_ok.append(stem)
        else:
            renames.append((vid, target_path, stem, safe_sn, new_sn, how))

    # ── Report ──────────────────────────────────────────────────────────────
    W = 72

    print("=" * W)
    print(
        f"RENAMING PLAN  ({len(renames)} renames | "
        f"{len(already_ok)} already correct | {len(no_match)} unmatched)"
    )
    print("=" * W)
    print(f"{'CURRENT NAME':<44}  {'NEW NAME (Dataverse-safe)'}")
    print("-" * W)
    for old_path, new_path, old_stem, new_stem, csv_sn, how in renames:
        conflict = (
            "  *** CONFLICT: target already exists! ***"
            if new_path.exists() and new_path != old_path
            else ""
        )
        # Show what changed: csv nickname → sanitised name
        note = (
            f"  [{how}]"
            if csv_sn == new_stem
            else f"  [{how}  →  sanitised from '{csv_sn}']"
        )
        print(f"  {old_stem:<44}→  {new_stem}{conflict}")
        print(f"  {'':<44}   {note}")
        leftover = remaining_issues(new_stem)
        if leftover:
            print(f"  {'':44}   ⚠ REMAINING ISSUES: {', '.join(leftover)}")
        print()

    print("=" * W)
    print(f"ALREADY CORRECT ({len(already_ok)} files — no rename needed)")
    print("-" * W)
    for stem in already_ok:
        print(f"  {stem}")

    print()
    print("=" * W)
    print(f"NO CSV MATCH ({len(no_match)} files — kept as-is, check manually)")
    print("-" * W)
    for stem, vid in no_match:
        issues = remaining_issues(stem)
        flag = f"  ⚠ {', '.join(issues)}" if issues else ""
        print(f"  {stem}{flag}")

    print()
    if not args.execute:
        print("[DRY RUN] No files were renamed. Run with --execute to apply.")
        return

    # ── Execute renames ──────────────────────────────────────────────────────
    print("Executing renames...")
    success = 0
    errors = 0
    for old_path, new_path, old_stem, new_stem, csv_sn, how in renames:
        if new_path.exists() and new_path != old_path:
            print(f"  SKIP (conflict): {old_path.name} → {new_path.name}")
            errors += 1
            continue
        try:
            old_path.rename(new_path)
            print(f"  ✓ {old_path.name} → {new_path.name}")
            success += 1
        except Exception as e:
            print(f"  ✗ {old_path.name}: {e}")
            errors += 1

    print(f"\nDone: {success} renamed, {errors} skipped/failed.")


if __name__ == "__main__":
    main()
