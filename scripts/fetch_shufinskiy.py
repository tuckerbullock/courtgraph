#!/usr/bin/env python3
"""Fetch SRC-SHUFINSKIY dataset archives from a pinned commit.

``shufinskiy/nba_data`` re-packages ``stats.nba.com`` / ``data.nba.com`` payloads
as flat CSVs, one ``<name>.tar.xz`` per (provider surface, season).
``DATA_SOURCES.md`` designates it the local-dev-only bulk source (SRC-SHUFINSKIY).
This script downloads a chosen set of those archives **from GitHub raw at a
pinned commit only** -- no NBA endpoint is contacted -- verifies each against any
recorded sha256, extracts the CSV, and records the checksum.

Trust model: the commit is pinned; checksums are recorded on first fetch
(trust-on-first-use) and re-verified on every subsequent run. A mismatch aborts.

Usage::

    python scripts/fetch_shufinskiy.py --plan cycle1        # the curated set
    python scripts/fetch_shufinskiy.py --names datanba_2016 nbastats_2016 \
        --dest data/nba_snapshots/_shufinskiy_rs_2016_2019

Nothing here is imported by the package; it is an operator tool.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tarfile
from pathlib import Path

PINNED_COMMIT = "e829d4678be1e075f99e5d41a1c5f97089be446b"
RAW_BASE = "https://github.com/shufinskiy/nba_data/raw"
REPO_ROOT = Path(__file__).resolve().parent.parent
SNAP_ROOT = REPO_ROOT / "data" / "nba_snapshots"

# Curated acquisition plan (plan file: maximal data acquisition). Maps each
# archive name to the local directory it belongs in. Regular-season 2020-24 of
# nbastats/datanba/shotdetail is already present and not re-listed. The
# pbpstats.com surface and every wnba_* archive are deliberately excluded
# (SRC-PBPSTATS rejected; WNBA out of scope). The 2024-25 playoffs stay in
# _shufinskiy_src/ (held out) and are not listed here.
PLANS: dict[str, dict[str, str]] = {
    "cycle1": {
        # earlier regular seasons 2016-17 .. 2019-20
        **{
            f"{surface}_{yr}": "_shufinskiy_rs_2016_2019"
            for surface in ("datanba", "nbastats", "shotdetail")
            for yr in (2016, 2017, 2018, 2019)
        },
        # new surfaces for the regular seasons already held (2020-24) + matchups
        # back to its first available season (2017)
        **{f"cdnnba_{yr}": "_shufinskiy_rs_2020_2024" for yr in range(2020, 2025)},
        **{f"nbastatsv3_{yr}": "_shufinskiy_rs_2020_2024" for yr in range(2020, 2025)},
        **{f"matchups_{yr}": "_shufinskiy_rs_2020_2024" for yr in range(2017, 2025)},
        # 2025-26: no datanba/nbastats upstream; cdnnba + v3 + shots + matchups
        **{
            f"{surface}_2025": "_shufinskiy_2025"
            for surface in ("cdnnba", "nbastatsv3", "shotdetail", "matchups")
        },
        # playoffs 2016 .. 2023 (2024 held out)
        **{
            f"{surface}_po_{yr}": "_shufinskiy_po_2016_2023"
            for surface in ("datanba", "nbastats", "shotdetail")
            for yr in range(2016, 2024)
        },
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_sums(dest: Path) -> dict[str, str]:
    sums_file = dest / "SHA256SUMS.txt"
    out: dict[str, str] = {}
    if sums_file.is_file():
        for line in sums_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, _, name = line.partition("  ")
            out[name.strip()] = digest.strip()
    return out


def _write_sums(dest: Path, sums: dict[str, str]) -> None:
    body = "".join(f"{sums[name]}  {name}\n" for name in sorted(sums))
    (dest / "SHA256SUMS.txt").write_text(body)


def _download(name: str, dest: Path, *, force: bool) -> tuple[bool, str]:
    """Return (fetched_now, sha256). Extracts the single CSV in the archive."""

    dest.mkdir(parents=True, exist_ok=True)
    archives = dest / "_archives"
    archives.mkdir(exist_ok=True)
    tar_path = archives / f"{name}.tar.xz"
    recorded = _read_sums(archives)

    if not tar_path.is_file() or force:
        url = f"{RAW_BASE}/{PINNED_COMMIT}/datasets/{name}.tar.xz"
        tmp = tar_path.with_suffix(".tar.xz.part")
        # curl, matching the recipe recorded in the existing SOURCE.md files;
        # the repo has no runtime deps and urllib's CA handling is unreliable here.
        result = subprocess.run(
            ["curl", "-sSL", "--fail", "--max-time", "300", "-o", str(tmp), url],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not tmp.is_file():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{name}: curl failed ({result.stderr.strip()})")
        tmp.replace(tar_path)
        fetched = True
    else:
        fetched = False

    digest = _sha256(tar_path)
    if name in recorded and recorded[name] != digest:
        tar_path.unlink()
        raise RuntimeError(
            f"{name}: sha256 mismatch\n  recorded {recorded[name]}\n  got      {digest}"
        )
    recorded[name] = digest
    _write_sums(archives, recorded)

    with tarfile.open(tar_path, "r:xz") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        if len(members) != 1 or not members[0].name.endswith(".csv"):
            raise RuntimeError(f"{name}: expected exactly one .csv in the archive")
        member = members[0]
        member.name = Path(member.name).name  # flatten
        tf.extract(member, dest, filter="data")

    return fetched, digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--plan", choices=sorted(PLANS), help="a curated acquisition set")
    src.add_argument("--names", nargs="+", help="explicit archive names")
    parser.add_argument(
        "--dest", help="destination dir (required with --names; relative to CWD)"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if the archive exists"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print what would be fetched"
    )
    args = parser.parse_args(argv)

    if args.names:
        if not args.dest:
            parser.error("--dest is required with --names")
        targets = {name: args.dest for name in args.names}
    else:
        targets = {name: str(SNAP_ROOT / rel) for name, rel in PLANS[args.plan].items()}

    print(f"pinned commit: {PINNED_COMMIT}")
    print(f"{len(targets)} archive(s) -> {len(set(targets.values()))} dir(s)\n")
    fetched = skipped = 0
    for name, dest_str in sorted(targets.items()):
        dest = Path(dest_str)
        if args.dry_run:
            print(f"  would fetch {name} -> {dest}")
            continue
        try:
            was_fetched, digest = _download(name, dest, force=args.force)
        except Exception as exc:  # noqa: BLE001 - operator tool, report and continue
            print(f"  FAIL {name}: {exc}", file=sys.stderr)
            return 1
        if was_fetched:
            fetched += 1
            print(f"  fetched  {name}  {digest[:12]}  -> {dest.name}/")
        else:
            skipped += 1
            print(f"  ok       {name}  {digest[:12]}  (cached)")

    if not args.dry_run:
        print(f"\ndone: {fetched} fetched, {skipped} already present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
