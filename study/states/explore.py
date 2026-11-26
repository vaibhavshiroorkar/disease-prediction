"""Exploratory checks on the state study, written after its results were seen.

    python -m study.states.explore

Reads results/states/ and writes results/states/exploratory.json. Nothing here
is a test of a frozen hypothesis. See docs/research/findings-states.md.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ml.config import ROOT
from study.shared.provenance import write_json
from study.states.verdicts import em_change

OUT = ROOT / "results" / "states"
KEYS = ["target", "repeat", "n"]


def paired_intercept(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per draw: its local cases and intercept's log-loss change from `none`."""
    mine = rows[rows.method == "intercept"].set_index(KEYS)
    base = rows[rows.method == "none"].set_index(KEYS)["log_loss"]
    return pd.DataFrame(
        {"cases": mine["local_cases"], "diff": mine["log_loss"] - base.loc[mine.index]}
    ).reset_index()


def cases_within_cells(draws: pd.DataFrame) -> dict[str, object]:
    """S1 pooled every n, so few cases mostly meant few patients. Hold target and n fixed."""
    rhos = []
    for (_, n), cell in draws.groupby(["target", "n"]):
        if cell["cases"].nunique() > 2:
            rhos.append({"n": int(n), "rho": spearmanr(cell["cases"], cell["diff"]).statistic})
    table = pd.DataFrame(rhos).dropna()
    by_n = table.groupby("n")["rho"]
    return {
        "cells": len(table),
        "share_negative": round(float((table["rho"] < 0).mean()), 3),
        "median_rho": round(float(table["rho"].median()), 3),
        "by_n": {str(n): {"cells": int(by_n.size()[n]), "median_rho": round(float(by_n.median()[n]), 3),
                          "share_negative": round(float((g < 0).mean()), 3)}
                 for n, g in by_n},
    }


def miscalibration_and_repair(targets: list[dict]) -> dict[str, object]:
    """How far the source was off (|log O/E|) against whether any repair worked."""
    frame = pd.DataFrame([
        {"name": t["name"], "oe": t["observed_over_expected"],
         "repaired": t["repair_size"]["intercept"] is not None, "em": em_change(t)}
        for t in targets
    ])
    frame["off"] = np.log(frame["oe"]).abs()
    unrepaired, repaired = frame[~frame["repaired"]], frame[frame["repaired"]]
    return {
        "unrepaired": unrepaired["name"].tolist(),
        "unrepaired_oe_range": [float(unrepaired["oe"].min()), float(unrepaired["oe"].max())],
        "largest_off_unrepaired": round(float(unrepaired["off"].max()), 3),
        "smallest_off_repaired": round(float(repaired["off"].min()), 3),
        "em_helped": frame.loc[frame["em"] < 0, "name"].tolist(),
        "oe_above_2": int((frame["oe"] > 2).sum()),
        "em_helped_with_oe_at_most_2": int(((frame["em"] < 0) & (frame["oe"] <= 2)).sum()),
    }


def main() -> int:
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    draws = paired_intercept(pd.read_csv(OUT / "repeats.csv.gz"))
    result = {
        "commit_of_results": report["provenance"]["commit"],
        "cases_within_cells": cases_within_cells(draws),
        "miscalibration_and_repair": miscalibration_and_repair(report["targets"]),
    }
    write_json(OUT / "exploratory.json", result)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
