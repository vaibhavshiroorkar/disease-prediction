"""The state study's frozen pass rules, from docs/research/protocol-states.md.

Each function takes per-target reports only, so the rules can be tested on
made-up reports and rerun from the saved files.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from ml import config

MIN_DRAWS = 20
S1_SHARE = 0.75
R1_SIZE, R1_SHARE, R2_SHARE = 500, 0.75, 0.25
STATE_BOOTSTRAPS = 1_000

Report = dict[str, Any]


def em_change(report: Report) -> float:
    """EM's mean paired log-loss change from `none`, at the target's largest n."""
    return report["curves"]["em"][str(report["sizes"][-1])]["diff_from_none"]


def guard_threshold(stage_a: list[Report]) -> float:
    """The shift-score cut that minimizes summed log loss of "EM below it, none above".

    Candidates are 0 (never EM) and just above each target's score. Ties go to
    the lowest cut, the one that uses EM least.
    """
    scores = [r["shift_score"] for r in stage_a]
    candidates = [0.0, *sorted(s + 1e-9 for s in scores)]

    # Summing the change from `none` ranks cuts the same as summing log loss.
    def total(cut: float) -> float:
        return sum(em_change(r) for r in stage_a if r["shift_score"] < cut)

    best = min(total(c) for c in candidates)
    return min(c for c in candidates if total(c) == best)


def bootstrap_by_state(
    reports: list[Report], statistic: Callable[[list[Report]], float | None], seed: int = config.RANDOM_SEED
) -> dict[str, Any]:
    """95% interval that resamples states, keeping each state's years together."""
    by_state: dict[str, list[Report]] = {}
    for r in reports:
        by_state.setdefault(r["state"], []).append(r)
    states = sorted(by_state)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(STATE_BOOTSTRAPS):
        drawn = [t for s in rng.choice(states, len(states)) for t in by_state[s]]
        value = statistic(drawn)
        if value is not None and np.isfinite(value):
            values.append(value)
    if not values:
        return {"ci": None, "valid": 0}
    low, high = np.percentile(values, [2.5, 97.5])
    return {"ci": [round(float(low), 4), round(float(high), 4)], "valid": len(values)}


def _cv(values: list[float]) -> float | None:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or values.mean() == 0:
        return None
    return float(values.std() / values.mean())


def cv_ratio(reports: list[Report]) -> float | None:
    """S2: spread of repair sizes in local cases over their spread in patients."""
    repaired = [r for r in reports if r["repair_size"]["intercept"] is not None]
    patients = _cv([r["repair_size"]["intercept"] for r in repaired])
    cases = _cv([r["mean_local_cases"][str(r["repair_size"]["intercept"])] for r in repaired])
    if patients is None or cases is None or patients == 0:
        return None
    return cases / patients


def shift_spearman(reports: list[Report]) -> float | None:
    """S3: rank correlation of shift score with EM's harm."""
    if len({r["shift_score"] for r in reports}) < 2 or len({em_change(r) for r in reports}) < 2:
        return None
    return float(spearmanr([r["shift_score"] for r in reports], [em_change(r) for r in reports]).statistic)


def _share_rule(names: list[str], of: int, needed: float) -> dict[str, Any]:
    return {"count": len(names), "of": of, "needed_share": needed, "targets": names,
            "supported": of > 0 and len(names) >= needed * of}


def verdicts(stage_a: list[Report], stage_b: list[Report], guard: float) -> dict[str, Any]:
    every = stage_a + stage_b

    eligible = [r for r in every
                if r["cases_split"]["low_draws"] >= MIN_DRAWS and r["cases_split"]["high_draws"] >= MIN_DRAWS]
    s1 = [r["name"] for r in eligible
          if r["cases_split"]["low_hurt_share"] > r["cases_split"]["high_hurt_share"]]

    ratio = cv_ratio(every)
    ratio_ci = bootstrap_by_state(every, cv_ratio)
    rho = shift_spearman(every)
    rho_ci = bootstrap_by_state(every, shift_spearman)

    em_worse = [r["name"] for r in stage_b if em_change(r) > 0]
    guarded = [r for r in stage_b if r["shift_score"] < guard]
    guard_worse = [r["name"] for r in guarded if em_change(r) > 0]
    guard_helps = [r["name"] for r in guarded if em_change(r) < 0]

    r1_pool = [r for r in every if R1_SIZE in r["sizes"]]
    r1 = [r["name"] for r in r1_pool if r["curves"]["intercept"][str(R1_SIZE)]["diff_from_none"] < 0]
    r2 = [r["name"] for r in every if em_change(r) > 0]

    return {
        "S1": {**_share_rule(s1, len(eligible), S1_SHARE),
               "eligible": [r["name"] for r in eligible]},
        "S2": {"cv_ratio": None if ratio is None else round(ratio, 4), **ratio_ci,
               "repaired": [r["name"] for r in every if r["repair_size"]["intercept"] is not None],
               "supported": ratio is not None and ratio < 1 and ratio_ci["ci"] is not None
               and ratio_ci["ci"][1] < 1},
        "S3": {"spearman": None if rho is None else round(rho, 4), **rho_ci, "of": len(every),
               "supported": rho is not None and rho > 0 and rho_ci["ci"] is not None and rho_ci["ci"][0] > 0},
        "S4": {"threshold": guard, "em_worse": em_worse, "guard_worse": guard_worse,
               "guard_helps": guard_helps, "of": len(stage_b),
               "supported": len(guard_worse) < len(em_worse) and len(guard_helps) >= 1},
        "R1": _share_rule(r1, len(r1_pool), R1_SHARE),
        "R2": _share_rule(r2, len(every), R2_SHARE),
    }
