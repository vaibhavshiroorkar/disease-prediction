"""Run the transfer study and write results/transfer/.

    python -m study.run                 # everything
    python -m study.run --family heart  # quick look, prints only

Protocol: docs/research/protocol.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from ml.clinical_eval import out_of_fold_proba
from ml.clinical_model import fixed, untuned
from ml.config import MODELS_DIR, RANDOM_SEED, ROOT, SOURCES
from study.audits import load_all, missingness_audit, site_audit
from study.transfer import run_dengue, run_heart, verdicts

OUT = ROOT / "results" / "transfer"


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
        "uncommitted_code": bool(_git("status", "--porcelain", "--", "study", "ml", "app")),
        "seed": RANDOM_SEED,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "data_sha256": {s.filename: hashlib.sha256(s.path.read_bytes()).hexdigest() for s in used},
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--family", choices=["heart", "dengue"], help="one family, nothing written")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    started = time.perf_counter()
    receipt = provenance()
    if receipt["uncommitted_code"] and not args.family:
        raise SystemExit("commit study/, ml/ and app/ first, so the result names the code that made it")

    data = load_all()
    reports, rows = [], []
    for family, run in (("heart", run_heart), ("dengue", run_dengue)):
        if args.family in (None, family):
            for report, repeat_rows in run(data[family], args.jobs):
                reports.append(report)
                rows.append(repeat_rows)
                print(f"{time.perf_counter() - started:6.0f}s {report['name']}: AUC "
                      f"{report['source']['auc']:.3f} -> {report['target']['auc']:.3f}, predicted "
                      f"{report['mean_predicted']:.3f}, real {report['target_rate']:.3f}", flush=True)
    if args.family:
        return 0

    print("audits", flush=True)
    params = json.loads((MODELS_DIR / "dengue_summary.json").read_text(encoding="utf-8"))["metrics"]
    models = {
        "heart": untuned("logistic", data["heart"].features, data["heart"].missing_indicators),
        "dengue": fixed("boosting", data["dengue"].features, data["dengue"].missing_indicators,
                        params["chosen_params"]),
    }
    audits = {
        "missingness": missingness_audit(data),
        "site": {
            name: site_audit(data[name], out_of_fold_proba(model, data[name].X, data[name].y))
            for name, model in models.items()
        },
    }
    receipt["seconds"] = round(time.perf_counter() - started)
    _write_json(OUT / "report.json",
                {"provenance": receipt, "verdicts": verdicts(reports), "transfers": reports})
    _write_json(OUT / "audits.json", {"provenance": receipt, **audits})
    pd.concat(rows).round(6).to_csv(OUT / "repeats.csv.gz", index=False)
    print(f"wrote {OUT} in {receipt['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
