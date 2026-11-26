"""Does EM fail because its rate estimate is wrong, or because the correction is?

    python -m study.true_rate

Exploratory, run after the state study's results were seen. It refits each
state's source model as run_states does, redraws the same test halves and EM
samples, and adds two corrections that are given the target's true rate:

- true_prior: EM's own correction (Saerens 2002) with the true rate in place of
  EM's estimate.
- true_mean: shift every logit by one amount, so the mean prediction on the
  pool half equals the true rate.

Both use labels the EM method never sees, so they are bounds, not methods.

Decision rule, written before this code was first run. Among the targets where
EM was worse than no update at the largest n:
- if true_prior beats no update in at least 75% of them, the estimate is what
  failed;
- if in at most 25%, the correction itself fails under this shift;
- otherwise the answer is mixed.

Writes results/states/true_rate.json. See docs/research/findings-states.md.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.metrics import log_loss

from ml.clinical_data import DENGUE_FEATURES
from ml.clinical_model import fixed
from ml.config import MODELS_DIR, RANDOM_SEED, ROOT
from study.run import _write_json, provenance
from study.states import SOURCE_YEAR, em_change, read_year, source_rows
from study.transfer import split_target
from study.updating import em_rate, reweight, to_logit

OUT = ROOT / "results" / "states"
ESTIMATE_FAILED, CORRECTION_FAILED = 0.75, 0.25
METHODS = ("none", "em", "true_prior", "true_mean")
# The refitted model must give the saved run's numbers, or this compares
# against a different model.
REPRODUCE_TOL = 1e-4


def mean_matching_shift(p: np.ndarray, rate: float) -> float:
    """The logit shift that makes the mean of p equal rate."""
    z = to_logit(p)
    return brentq(lambda s: expit(z + s).mean() - rate, -30.0, 30.0)


def _loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))


def one_draw(p: np.ndarray, y: np.ndarray, source_rate: float, n: int, repeat: int) -> dict[str, float]:
    """The test half and EM sample of one_repeat in study.transfer, plus the two bounds."""
    seed = RANDOM_SEED + repeat
    pool, test = split_target(y, seed)
    local = np.random.default_rng(seed).permutation(pool)[:n]
    true_rate = float(y.mean())
    estimate = em_rate(p[local], source_rate)
    p_test = p[test]
    updated = {
        "none": p_test,
        "em": reweight(p_test, source_rate, estimate),
        "true_prior": reweight(p_test, source_rate, true_rate),
        "true_mean": expit(to_logit(p_test) + mean_matching_shift(p[pool], true_rate)),
    }
    row = {m: _loss(y[test], updated[m]) for m in METHODS}
    row["em_rate"] = estimate
    return row


def run_target(report: dict[str, Any], p: np.ndarray, y: np.ndarray, source_rate: float) -> dict[str, Any]:
    n = report["sizes"][-1]
    draws = [one_draw(p, y, source_rate, n, r) for r in range(report["repeats"])]
    mean = {m: float(np.mean([d[m] for d in draws])) for m in METHODS}
    saved_none = report["curves"]["none"][str(n)]["log_loss"]["mean"]
    if abs(mean["none"] - saved_none) > REPRODUCE_TOL:
        raise RuntimeError(
            f"{report['name']}: refitted model gives {mean['none']:.5f}, saved {saved_none:.5f}"
        )
    if abs((mean["em"] - mean["none"]) - em_change(report)) > REPRODUCE_TOL:
        raise RuntimeError(f"{report['name']}: EM change does not match the saved run")
    return {
        "name": report["name"],
        "n": n,
        "observed_over_expected": report["observed_over_expected"],
        "source_rate": round(source_rate, 5),
        "true_rate": round(float(y.mean()), 5),
        "mean_predicted": round(float(p.mean()), 5),
        "em_rate_median": round(float(np.median([d["em_rate"] for d in draws])), 5),
        "change_from_none": {m: round(mean[m] - mean["none"], 5) for m in METHODS[1:]},
    }


def summarize(targets: list[dict[str, Any]]) -> dict[str, Any]:
    em_hurt = [t for t in targets if t["change_from_none"]["em"] > 0]

    def helped(group: list[dict[str, Any]], method: str) -> int:
        return sum(t["change_from_none"][method] < 0 for t in group)

    share = helped(em_hurt, "true_prior") / len(em_hurt) if em_hurt else None
    if share is None:
        answer = "no target where EM hurt"
    elif share >= ESTIMATE_FAILED:
        answer = "the estimate failed"
    elif share <= CORRECTION_FAILED:
        answer = "the correction failed"
    else:
        answer = "mixed"
    return {
        "targets": len(targets),
        "em_hurt": len(em_hurt),
        "true_prior_helped_where_em_hurt": helped(em_hurt, "true_prior"),
        "true_mean_helped_where_em_hurt": helped(em_hurt, "true_mean"),
        "helped_of_all": {m: helped(targets, m) for m in METHODS[1:]},
        "answer": answer,
    }


def main() -> int:
    started = time.perf_counter()
    receipt = provenance()
    if receipt["uncommitted_code"]:
        raise SystemExit("commit study/, ml/ and app/ first, so the result names the code that made it")
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    params = json.loads((MODELS_DIR / "dengue_summary.json").read_text(encoding="utf-8"))["metrics"][
        "chosen_params"
    ]
    years = {year: read_year(year) for year in sorted({SOURCE_YEAR, *(t["year"] for t in report["targets"])})}
    print(f"{time.perf_counter() - started:5.0f}s read data", flush=True)

    results = []
    by_state: dict[str, list[dict[str, Any]]] = {}
    for target in report["targets"]:
        by_state.setdefault(target["state"], []).append(target)
    for state, targets in sorted(by_state.items()):
        X_s, y_s = source_rows(years[SOURCE_YEAR], state)
        model = fixed("boosting", DENGUE_FEATURES, True, params).fit(X_s, y_s)
        for target in targets:
            year = years[target["year"]]
            held = year.state == state
            p = model.predict_proba(year.X[held].reset_index(drop=True))[:, 1]
            result = run_target(target, p, year.y[held], float(y_s.mean()))
            results.append(result)
            print(f"{time.perf_counter() - started:5.0f}s {result['name']}: {result['change_from_none']}",
                  flush=True)

    receipt["seconds"] = round(time.perf_counter() - started)
    out = {"provenance": receipt, "commit_of_results": report["provenance"]["commit"],
           "summary": summarize(results), "targets": results}
    _write_json(OUT / "true_rate.json", out)
    print(json.dumps(out["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
