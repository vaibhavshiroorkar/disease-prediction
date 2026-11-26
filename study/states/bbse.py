"""Can a different label-free rate estimate fix EM's failure?

    python -m study.states.bbse

Exploratory, run after the state study and the true-rate check
(results/states/true_rate.json), which found that EM's correction works when
given the true rate, and that EM's estimate of the rate is what fails.

This check keeps EM's correction and swaps the estimate for black-box shift
estimation (BBSE, Lipton et al. 2018). BBSE learns from the source model's
out-of-fold predictions how often it flags real cases (TPR) and non-cases
(FPR), then solves `flagged share = rate * TPR + (1 - rate) * FPR` on the
same unlabelled local sample EM saw.

- bbse: flags at the source threshold, as in Lipton et al. Named in advance as
  the method the rule judges.
- bbse_soft: the same with mean probabilities in place of flags. Secondary.

Decision rule, written before this code was first run, judged at the largest n:
- bbse is a usable label-free repair if it beats no update in at least 75% of
  the 47 targets (the bar R1 used for the labelled shift);
- bbse is a better estimate than EM if it beats EM in at least 75%;
- if neither, BBSE fails here too, and the next thing to try is not another
  estimator that assumes only the rate changed.

Writes results/states/bbse.json. See docs/research/findings-states.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from study.shared.repeats import clipped_log_loss, draw
from study.shared.updating import EPS, em_rate, reweight
from study.states.experiment import Source
from study.states.rechecks import check_reproduces, run_check

USABLE, BETTER = 0.75, 0.75
METHODS = ("none", "em", "bbse", "bbse_soft")
ESTIMATORS = ("em", "bbse", "bbse_soft")


def solve_rate(share: float, on_cases: float, on_non_cases: float) -> float:
    """The rate r with share = r * on_cases + (1 - r) * on_non_cases, kept inside (0, 1)."""
    if on_cases <= on_non_cases:
        raise ValueError("the source model does not separate cases from non-cases")
    return float(np.clip((share - on_non_cases) / (on_cases - on_non_cases), EPS, 1 - EPS))


def bbse_rates(source: Source, p_local: np.ndarray) -> dict[str, float]:
    """Hard and soft BBSE estimates of the local rate, from source out-of-fold predictions."""
    y_s, oof, cut = source.y.to_numpy(), source.oof, source.threshold
    flagged = oof >= cut
    tpr, fpr = flagged[y_s == 1].mean(), flagged[y_s == 0].mean()
    return {
        "bbse": solve_rate(float((p_local >= cut).mean()), tpr, fpr),
        "bbse_soft": solve_rate(float(p_local.mean()), oof[y_s == 1].mean(), oof[y_s == 0].mean()),
    }


def one_draw(p: np.ndarray, y: np.ndarray, source: Source, n: int, repeat: int) -> dict[str, float]:
    """The saved draw's test half and local sample; each estimate feeds EM's correction."""
    _, test, order = draw(y, repeat)
    p_local, p_test = p[order[:n]], p[test]
    rates = {"em": em_rate(p_local, source.rate), **bbse_rates(source, p_local)}
    row = {"none": clipped_log_loss(y[test], p_test)}
    for name, rate in rates.items():
        row[name] = clipped_log_loss(y[test], reweight(p_test, source.rate, rate))
        row[f"{name}_rate"] = rate
    return row


def run_target(report: dict[str, Any], p: np.ndarray, y: np.ndarray, source: Source) -> dict[str, Any]:
    n = report["sizes"][-1]
    draws = [one_draw(p, y, source, n, r) for r in range(report["repeats"])]
    mean = {m: float(np.mean([d[m] for d in draws])) for m in METHODS}
    check_reproduces(report, mean)
    true_rate = float(y.mean())
    return {
        "name": report["name"],
        "n": n,
        "observed_over_expected": report["observed_over_expected"],
        "true_rate": round(true_rate, 5),
        "rate_over_true": {e: round(float(np.median([d[f"{e}_rate"] for d in draws])) / true_rate, 4)
                           for e in ESTIMATORS},
        "change_from_none": {m: round(mean[m] - mean["none"], 5) for m in METHODS[1:]},
    }


def summarize(targets: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(targets)

    def count(rule) -> int:
        return sum(bool(rule(t["change_from_none"])) for t in targets)

    beats_none = {m: count(lambda c, m=m: c[m] < 0) for m in METHODS[1:]}
    beats_em = {m: count(lambda c, m=m: c[m] < c["em"]) for m in ("bbse", "bbse_soft")}
    close = {e: sum(0.8 <= t["rate_over_true"][e] <= 1.25 for t in targets) for e in ESTIMATORS}
    usable = beats_none["bbse"] >= USABLE * total
    better = beats_em["bbse"] >= BETTER * total
    if usable:
        answer = "bbse is a usable label-free repair"
    elif better:
        answer = "bbse is a better estimate than EM, but not a usable repair"
    else:
        answer = "bbse fails here too"
    return {"targets": total, "beats_none": beats_none, "beats_em": beats_em,
            "rate_within_0.8_to_1.25": close, "answer": answer}


def main() -> int:
    return run_check("bbse.json", run_target, summarize, oof=True)


if __name__ == "__main__":
    raise SystemExit(main())
