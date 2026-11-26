"""The frozen 2026 retest of the state study's findings.

    python -m study.states.retest

Protocol: docs/research/protocol-2026.md. The pass marks below are copied from
it and must not change after the 2026 file has been read. Needs the 2026 file
(python -m scripts.download_data) and the saved state-study results, including
results/states/bbse.json for T7.

Writes results/retest-2026/. A finished target is saved at once, so a stopped
run picks up where it left off, as long as the code has not changed.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ml.config import ROOT
from study.shared.provenance import committed_provenance, write_json
from study.states import bbse, true_rate
from study.states.data import (
    SOURCE_YEAR,
    choose_targets,
    count_cases,
    read_years,
    target_name,
    target_rows,
    unclassified_share,
)
from study.states.experiment import REPEATS, Sources, run_target
from study.states.explore import cases_within_cells, paired_intercept
from study.states.run import saved_file
from study.states.verdicts import bootstrap_by_state, em_change

YEAR = 2026
OUT = ROOT / "results" / "retest-2026"
MIN_TARGETS = 8
SHARE = 0.75
T2_MIN_HURT = 4
T3_RHO = 0.5
BAND = (0.85, 1.12)
T4_MIN_SIDE, T4_GAP = 3, 0.25
T5_RATIO, T5_MIN, T5_SHARE = 2.0, 5, 0.10
T6_MIN_CELLS, T6_SHARE = 20, 0.6

TESTS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")

Target = dict[str, Any]


def _rule(supported: bool, **detail: Any) -> dict[str, Any]:
    return {"supported": bool(supported), "testable": True, **detail}


def _untestable(reason: str) -> dict[str, Any]:
    return {"supported": False, "testable": False, "reason": reason}


def _em_error(t: Target) -> float:
    return abs(float(np.log(max(t["em_rate_over_true"], 1e-9))))


def _error_spearman(targets: list[Target]) -> float | None:
    errors, harm = [_em_error(t) for t in targets], [t["change"]["em"] for t in targets]
    if len(set(errors)) < 2 or len(set(harm)) < 2:
        return None
    return float(spearmanr(errors, harm).statistic)


def retest_verdicts(targets: list[Target], cells: dict[str, Any], bbse_before: str) -> dict[str, Any]:
    """T1 to T7 of the protocol. Each target holds name, state, oe, repaired, change, em_rate_over_true."""
    if len(targets) < MIN_TARGETS:
        return {t: _untestable(f"only {len(targets)} targets") for t in TESTS}
    total = len(targets)
    hurt = [t for t in targets if t["change"]["em"] > 0]
    out: dict[str, Any] = {"T1": _rule(len(hurt) >= SHARE * total, count=len(hurt), of=total)}

    fixed = [t for t in hurt if t["change"]["true_prior"] < 0]
    out["T2"] = (_untestable(f"EM hurt in only {len(hurt)}") if len(hurt) < T2_MIN_HURT
                 else _rule(len(fixed) >= SHARE * len(hurt), count=len(fixed), of=len(hurt)))

    rho = _error_spearman(targets)
    ci = bootstrap_by_state(targets, _error_spearman)
    out["T3"] = _rule(rho is not None and rho >= T3_RHO and ci["ci"] is not None and ci["ci"][0] > 0,
                      spearman=None if rho is None else round(rho, 4), **ci)

    inside = [t for t in targets if BAND[0] <= t["oe"] <= BAND[1]]
    outside = [t for t in targets if not BAND[0] <= t["oe"] <= BAND[1]]
    if min(len(inside), len(outside)) < T4_MIN_SIDE:
        out["T4"] = _untestable(f"{len(inside)} inside the band, {len(outside)} outside")
    else:
        share_in = sum(t["repaired"] for t in inside) / len(inside)
        share_out = sum(t["repaired"] for t in outside) / len(outside)
        out["T4"] = _rule(share_out - share_in >= T4_GAP, repaired_inside=round(share_in, 4),
                          repaired_outside=round(share_out, 4), inside=len(inside), outside=len(outside))

    modest = [t for t in targets if t["oe"] <= T5_RATIO]
    if len(modest) < T5_MIN:
        out["T5"] = _untestable(f"only {len(modest)} targets with O/E at most {T5_RATIO}")
    else:
        em_helped = sum(t["change"]["em"] < 0 for t in modest)
        out["T5"] = _rule(em_helped <= T5_SHARE * len(modest), count=em_helped, of=len(modest))

    out["T6"] = (_untestable(f"only {cells['cells']} cells") if cells["cells"] < T6_MIN_CELLS
                 else _rule(cells["share_negative"] >= T6_SHARE, **cells))

    after = bbse.summarize([t["bbse"] for t in targets])["answer"]
    out["T7"] = _rule(after == bbse_before, before=bbse_before, after=after)
    return out


def _target_summary(report: dict[str, Any], rate: dict[str, Any], estimates: dict[str, Any]) -> Target:
    return {
        "name": report["name"], "state": report["state"],
        "oe": report["observed_over_expected"],
        "repaired": report["repair_size"]["intercept"] is not None,
        "change": {"em": em_change(report), "true_prior": rate["change_from_none"]["true_prior"]},
        "em_rate_over_true": estimates["rate_over_true"]["em"],
        "bbse": estimates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    started = time.perf_counter()
    receipt = committed_provenance()
    commit = receipt["commit"]
    bbse_before = json.loads((ROOT / "results" / "states" / "bbse.json").read_text(encoding="utf-8"))
    bbse_before = bbse_before["summary"]["answer"]

    years = read_years({SOURCE_YEAR, YEAR})
    chosen, excluded = choose_targets(count_cases({YEAR: years[YEAR]}))
    write_json(OUT / "targets.json", {"targets": chosen, "excluded": excluded,
                                       "unclassified_share": round(unclassified_share(YEAR), 4)})
    print(f"{time.perf_counter() - started:5.0f}s {len(chosen)} targets", flush=True)
    sources = Sources(years[SOURCE_YEAR])

    summaries, frames = [], []
    for year, state in chosen:
        path = OUT / "targets" / f"{year}_{state}.json"
        saved = saved_file(path, commit)
        if saved is None:
            source = sources(state)
            X_t, y_t = target_rows(years[year], state)
            report, rows = run_target(year, state, "retest", source, X_t, y_t, args.jobs, REPEATS)
            p = source.model.predict_proba(X_t)[:, 1]
            saved = {"commit": commit, "report": report,
                     "true_rate": true_rate.run_target(report, p, y_t, source),
                     "bbse": bbse.run_target(report, p, y_t, source)}
            path.parent.mkdir(parents=True, exist_ok=True)
            rows.round(6).to_csv(path.with_name(path.stem + ".csv.gz"), index=False)
            write_json(path, saved)
        summaries.append(_target_summary(saved["report"], saved["true_rate"], saved["bbse"]))
        frames.append(pd.read_csv(path.with_name(path.stem + ".csv.gz")))
        print(f"{time.perf_counter() - started:5.0f}s {target_name(year, state)}: "
              f"{summaries[-1]['change']}", flush=True)

    rows = pd.concat(frames) if frames else pd.DataFrame()
    cells = cases_within_cells(paired_intercept(rows)) if len(rows) else {"cells": 0, "share_negative": None}
    receipt["seconds"] = round(time.perf_counter() - started)
    write_json(OUT / "report.json", {"provenance": receipt,
                                     "verdicts": retest_verdicts(summaries, cells, bbse_before),
                                     "targets": summaries})
    if len(rows):
        rows.to_csv(OUT / "repeats.csv.gz", index=False)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
