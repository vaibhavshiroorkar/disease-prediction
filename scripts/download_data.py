"""Fetch the raw CSVs into data/raw/.

Run before training:

    python -m scripts.download_data

Files already on disk are left alone unless --force is passed.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request

from ml.config import RAW_DIR, SOURCES

USER_AGENT = "disease-prediction-system/1.0 (dataset fetcher)"


def fetch(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download(force: bool = False) -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0

    for source in SOURCES.values():
        if source.path.exists() and not force:
            size_kb = source.path.stat().st_size / 1024
            print(f"skip  {source.filename:24s} ({size_kb:.0f} KB already present)")
            continue

        print(f"get   {source.filename:24s} <- {source.url}")
        try:
            payload = fetch(source.url)
        except Exception as exc:  # noqa: BLE001 - report and continue to the next file
            print(f"FAIL  {source.filename}: {exc}", file=sys.stderr)
            failures += 1
            continue

        source.path.write_bytes(payload)
        print(f"ok    {source.filename:24s} ({len(payload) / 1024:.0f} KB)")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args()

    failures = download(force=args.force)
    if failures:
        print(f"\n{failures} download(s) failed.", file=sys.stderr)
        return 1

    print(f"\nRaw data ready in {RAW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
