"""What produced a result: code commit, data hashes, library versions, seed."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from ml.config import RANDOM_SEED, ROOT, SOURCES

CODE_DIRS = ("study", "ml", "app")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, encoding="utf-8"
    ).stdout.strip()


def provenance() -> dict[str, object]:
    """Code, data, library versions and seed behind the numbers."""
    prefixes = ("heart_", "dengue_", "diabetes", "liver")
    used = [s for k, s in SOURCES.items() if k.startswith(prefixes)]
    return {
        "commit": _git("rev-parse", "HEAD"),
        "uncommitted_code": bool(_git("status", "--porcelain", "--", *CODE_DIRS)),
        "seed": RANDOM_SEED,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "data_sha256": {s.filename: hashlib.sha256(s.path.read_bytes()).hexdigest() for s in used},
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def committed_provenance() -> dict[str, object]:
    """provenance(), refusing to run when the code differs from the commit it would name."""
    receipt = provenance()
    if receipt["uncommitted_code"]:
        raise SystemExit(f"commit {', '.join(CODE_DIRS)} first, so the result names the code that made it")
    return receipt


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")
