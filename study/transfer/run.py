"""Run the transfer study and write results/transfer/.

    python -m study.transfer.run                 # everything
    python -m study.transfer.run --family heart  # quick look, prints only

Protocol: docs/research/protocol.md
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from ml.clinical_eval import out_of_fold_proba
from ml.clinical_model import untuned
from ml.config import ROOT
from study.shared.models import dengue_model
from study.shared.provenance import committed_provenance, provenance, write_json
from study.transfer.audits import load_all, missingness_audit, site_audit
from study.transfer.experiment import run_dengue, run_heart, verdicts

OUT = ROOT / "results" / "transfer"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--family", choices=["heart", "dengue"], help="one family, nothing written")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    started = time.perf_counter()
    receipt = provenance() if args.family else committed_provenance()

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
    models = {
        "heart": untuned("logistic", data["heart"].features, data["heart"].missing_indicators),
        "dengue": dengue_model(data["dengue"].features, data["dengue"].missing_indicators),
    }
    audits = {
        "missingness": missingness_audit(data),
        "site": {
            name: site_audit(data[name], out_of_fold_proba(model, data[name].X, data[name].y))
            for name, model in models.items()
        },
    }
    receipt["seconds"] = round(time.perf_counter() - started)
    report = {"provenance": receipt, "verdicts": verdicts(reports), "transfers": reports}
    write_json(OUT / "report.json", report)
    write_json(OUT / "audits.json", {"provenance": receipt, **audits})
    pd.concat(rows).round(6).to_csv(OUT / "repeats.csv.gz", index=False)
    print(f"wrote {OUT} in {receipt['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
