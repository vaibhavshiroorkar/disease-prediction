"""Does EM fail because its rate estimate is wrong, or because the correction is?

    python -m study.states.true_rate

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

from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from study.shared.repeats import clipped_log_loss, draw
from study.shared.updating import em_rate, reweight, to_logit
from study.states.experiment import Source
from study.states.rechecks import check_reproduces, run_check

ESTIMATE_FAILED, CORRECTION_FAILED = 0.75, 0.25
METHODS = ("none", "em", "true_prior", "true_mean")


def mean_matching_shift(p: np.ndarray, rate: float) -> float:
    """The logit shift that makes the mean of p equal rate."""
    z = to_logit(p)
    return brentq(lambda s: expit(z + s).mean() - rate, -30.0, 30.0)


def one_draw(p: np.ndarray, y: np.ndarray, source_rate: float, n: int, repeat: int) -> dict[str, float]:
    """The test half and EM sample of one_repeat in study.shared.repeats, plus the two bounds."""
    pool, test, order = draw(y, repeat)
    local = order[:n]
    true_rate = float(y.mean())
    estimate = em_rate(p[local], source_rate)
    p_test = p[test]
    updated = {
        "none": p_test,
        "em": reweight(p_test, source_rate, estimate),
        "true_prior": reweight(p_test, source_rate, true_rate),
        "true_mean": expit(to_logit(p_test) + mean_matching_shift(p[pool], true_rate)),
    }
    row = {m: clipped_log_loss(y[test], updated[m]) for m in METHODS}
    row["em_rate"] = estimate
    return row


def run_target(report: dict[str, Any], p: np.ndarray, y: np.ndarray, source: Source) -> dict[str, Any]:
    n = report["sizes"][-1]
    draws = [one_draw(p, y, source.rate, n, r) for r in range(report["repeats"])]
    mean = {m: float(np.mean([d[m] for d in draws])) for m in METHODS}
    check_reproduces(report, mean)
    return {
        "name": report["name"],
        "n": n,
        "observed_over_expected": report["observed_over_expected"],
        "source_rate": round(source.rate, 5),
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
    return run_check("true_rate.json", run_target, summarize, oof=False)


if __name__ == "__main__":
    raise SystemExit(main())
