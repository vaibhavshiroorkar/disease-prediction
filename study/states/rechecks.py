"""Shared machinery for checks that rerun saved state-study draws with new methods.

Each check refits the source models, redraws the saved test halves and local
samples, and must reproduce the saved `none` and `em` numbers before it may
report anything new.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ml.config import ROOT
from study.shared.provenance import committed_provenance, write_json
from study.states.data import SOURCE_YEAR, read_years, target_rows
from study.states.experiment import Source, Sources
from study.states.verdicts import em_change

OUT = ROOT / "results" / "states"
# The refitted model must give the saved run's numbers, or a check compares
# against a different model.
REPRODUCE_TOL = 1e-4

Report = dict[str, Any]
PerTarget = Callable[[Report, np.ndarray, np.ndarray, Source], dict[str, Any]]


def check_reproduces(report: Report, mean: dict[str, float]) -> None:
    """Stop unless this check's `none` and `em` means match the saved run."""
    n = report["sizes"][-1]
    saved_none = report["curves"]["none"][str(n)]["log_loss"]["mean"]
    if abs(mean["none"] - saved_none) > REPRODUCE_TOL:
        raise RuntimeError(
            f"{report['name']}: refitted model gives {mean['none']:.5f}, saved {saved_none:.5f}"
        )
    if abs((mean["em"] - mean["none"]) - em_change(report)) > REPRODUCE_TOL:
        raise RuntimeError(f"{report['name']}: EM change does not match the saved run")


def run_check(
    filename: str, per_target: PerTarget, summarize: Callable[[list[dict[str, Any]]], dict[str, Any]],
    oof: bool,
) -> int:
    """Run per_target on every saved target and write results/states/<filename>."""
    started = time.perf_counter()
    receipt = committed_provenance()
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    years = read_years({SOURCE_YEAR, *(t["year"] for t in report["targets"])})
    sources = Sources(years[SOURCE_YEAR], oof=oof)
    print(f"{time.perf_counter() - started:5.0f}s read data", flush=True)

    results = []
    for target in sorted(report["targets"], key=lambda t: (t["state"], t["year"])):
        source = sources(target["state"])
        X_t, y_t = target_rows(years[target["year"]], target["state"])
        result = per_target(target, source.model.predict_proba(X_t)[:, 1], y_t, source)
        results.append(result)
        elapsed = time.perf_counter() - started
        print(f"{elapsed:5.0f}s {result['name']}: {result['change_from_none']}", flush=True)

    receipt["seconds"] = round(time.perf_counter() - started)
    out = {"provenance": receipt, "commit_of_results": report["provenance"]["commit"],
           "summary": summarize(results), "targets": results}
    write_json(OUT / filename, out)
    print(json.dumps(out["summary"], indent=1))
    return 0
