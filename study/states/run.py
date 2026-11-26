"""Run the state-by-state dengue study and write results/states/.

    python -m study.states.run              # the full study, resumable
    python -m study.states.run --smoke DIR  # 2 targets per stage, 3 repeats, into DIR

Stage A (2023) runs first. The EM guard threshold is then written to
guard.json, and stage B (2024, 2025) runs against that fixed threshold. A
finished target is saved at once, so a stopped run picks up where it left
off, as long as the code has not changed.

Protocol: docs/research/protocol-states.md
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import ROOT
from study.shared.provenance import committed_provenance, provenance, write_json
from study.states.data import (
    SOURCE_YEAR,
    STAGE_A,
    STAGE_B,
    choose_targets,
    count_cases,
    read_years,
    target_rows,
)
from study.states.experiment import REPEATS, Sources, run_target
from study.states.verdicts import guard_threshold, verdicts

OUT = ROOT / "results" / "states"


def _log(started: float, message: str) -> None:
    print(f"{time.perf_counter() - started:7.0f}s {message}", flush=True)


def saved_file(path: Path, commit: str) -> dict[str, Any] | None:
    """A finished file from this commit, or None. Another commit's file is an error."""
    if not path.exists():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved["commit"] != commit:
        raise SystemExit(f"{path} came from commit {saved['commit']}. Move {path.parent} away to rerun.")
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--smoke", type=Path, metavar="DIR", help="tiny run into DIR, for checking the code")
    args = parser.parse_args()

    started = time.perf_counter()
    receipt = provenance() if args.smoke else committed_provenance()
    out = args.smoke or OUT
    repeats = 3 if args.smoke else REPEATS
    commit = receipt["commit"]

    years = read_years({SOURCE_YEAR, *STAGE_A, *STAGE_B})
    _log(started, "read " + ", ".join(f"{y}: {len(d.y)} cases" for y, d in years.items()))
    chosen, excluded = choose_targets(count_cases(years))
    if args.smoke:
        chosen = [t for t in chosen if t[0] in STAGE_A][:2] + [t for t in chosen if t[0] in STAGE_B][:2]
    write_json(out / "targets.json", {"targets": chosen, "excluded": excluded})
    _log(started, f"{len(chosen)} targets, {len(excluded)} state-years left out")
    sources = Sources(years[SOURCE_YEAR])

    def run_stage(stage: str, stage_years: tuple[int, ...]) -> list[dict[str, Any]]:
        reports = []
        for year, state in [t for t in chosen if t[0] in stage_years]:
            path = out / "targets" / f"{year}_{state}.json"
            report = saved_file(path, commit)
            if report is None:
                X_t, y_t = target_rows(years[year], state)
                report, rows = run_target(year, state, stage, sources(state), X_t, y_t, args.jobs, repeats)
                report["commit"] = commit
                path.parent.mkdir(parents=True, exist_ok=True)
                rows.round(6).to_csv(path.with_name(path.stem + ".csv.gz"), index=False)
                write_json(path, report)
            reports.append(report)
            _log(started, f"{report['name']}: AUC {report['source']['auc']:.3f} -> "
                          f"{report['target']['auc']:.3f}, O/E {report['observed_over_expected']:.2f}, "
                          f"shift {report['shift_score']:.3f}")
        return reports

    stage_a = run_stage("A", STAGE_A)
    guard = saved_file(out / "guard.json", commit)
    if guard is None:
        guard = {"commit": commit, "threshold": guard_threshold(stage_a),
                 "fitted_on": [r["name"] for r in stage_a]}
        write_json(out / "guard.json", guard)
    _log(started, f"EM guard threshold {guard['threshold']:.4f}, fixed before stage B")

    stage_b = run_stage("B", STAGE_B)
    receipt["seconds"] = round(time.perf_counter() - started)
    write_json(out / "report.json", {
        "provenance": receipt,
        "verdicts": verdicts(stage_a, stage_b, guard["threshold"]),
        "targets": stage_a + stage_b,
    })
    frames = [pd.read_csv(p) for p in sorted((out / "targets").glob("*.csv.gz"))]
    pd.concat(frames).to_csv(out / "repeats.csv.gz", index=False)
    _log(started, f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
