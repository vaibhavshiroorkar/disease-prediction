"""Fetch the raw data files into data/raw/.

Run before training:

    python -m scripts.download_data                     # everything, about 450 MB
    python -m scripts.download_data --only heart_va diabetes

Files already on disk are left alone unless --force is passed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request

from ml.config import RAW_DIR, SOURCES

USER_AGENT = "disease-prediction-system/1.0 (dataset fetcher)"


def fetch(url: str, destination, timeout: int = 60) -> int:
    """Stream the body to disk and return its size in bytes.

    The malaria images are 353 MB, so the body never sits in memory whole. It
    lands in a .part file first, so a dropped connection cannot leave a
    truncated file that the next run would skip as already present.
    """
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    partial.replace(destination)
    return destination.stat().st_size


def download(force: bool = False, only: list[str] | None = None) -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0

    for source in SOURCES.values():
        if only and source.key not in only:
            continue
        if source.path.exists() and not force:
            size_kb = source.path.stat().st_size / 1024
            print(f"skip  {source.filename:28s} ({size_kb:.0f} KB already present)")
            continue

        print(f"get   {source.filename:28s} <- {source.url}")
        try:
            size = fetch(source.url, source.path)
        except Exception as exc:  # noqa: BLE001 - report and continue to the next file
            print(f"FAIL  {source.filename}: {exc}", file=sys.stderr)
            failures += 1
            continue
        print(f"ok    {source.filename:28s} ({size / 1024:.0f} KB)")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    parser.add_argument("--only", nargs="+", choices=list(SOURCES), metavar="KEY",
                        help=f"fetch only these sources: {', '.join(SOURCES)}")
    args = parser.parse_args()

    failures = download(force=args.force, only=args.only)
    if failures:
        print(f"\n{failures} download(s) failed.", file=sys.stderr)
        return 1

    print(f"\nRaw data ready in {RAW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
