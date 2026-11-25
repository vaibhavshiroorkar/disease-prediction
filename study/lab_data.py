"""Build app/static/lab-data.json, the numbers behind the website's lab page.

    python -m study.lab_data

Scores the committed models on their own held-out rows and stops if the AUC
differs from the committed summary. The transfer numbers come straight from
results/transfer/. The file holds counts and averages only, no patient rows.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ml.clinical_data import LOADERS
from ml.clinical_eval import calibration_table
from ml.config import MODELS_DIR, RANDOM_SEED, ROOT
from ml.malaria import load_malaria
from ml.train import _split

TARGET = ROOT / "app" / "static" / "lab-data.json"
RESULTS = ROOT / "results" / "transfer"
GRID = 150
HISTOGRAM_BINS = 24
CURVE_METRICS = ("log_loss", "gap", "auc")


def threshold_grid(p: np.ndarray, operating: float) -> np.ndarray:
    """Cutoffs spread over the scores in use, always including the served one."""
    quantiles = np.quantile(p, np.linspace(0, 1, GRID))
    return np.unique(np.concatenate([[0.0], np.round(quantiles, 5), [operating], [1.0]]))


def threshold_counts(y: np.ndarray, p: np.ndarray, grid: np.ndarray) -> dict[str, list[int]]:
    """True and false positives when everything at or above each cutoff is flagged."""
    pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
    return {
        "tp": (len(pos) - np.searchsorted(pos, grid, side="left")).tolist(),
        "fp": (len(neg) - np.searchsorted(neg, grid, side="left")).tolist(),
    }


def histogram(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    top = float(np.quantile(p, 0.995))
    edges = np.linspace(0, max(top, 0.01), HISTOGRAM_BINS + 1)
    clipped = np.clip(p, 0, edges[-1])
    return {
        "edges": np.round(edges, 5).tolist(),
        "positive": np.histogram(clipped[y == 1], edges)[0].tolist(),
        "negative": np.histogram(clipped[y == 0], edges)[0].tolist(),
    }


def held_out(name: str) -> tuple[Any, np.ndarray]:
    """The same test rows that training set aside."""
    if name == "malaria":
        cells = load_malaria()
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        _, test = next(splitter.split(cells.X, cells.y, cells.groups))
        return cells.X[test], cells.y[test]
    _, X_test, _, y_test = _split(LOADERS[name]())
    return X_test, y_test.to_numpy()


def model_panel(name: str) -> dict[str, Any]:
    # Safe only because this is our own committed artifact. See docs/SECURITY.md.
    bundle = joblib.load(MODELS_DIR / f"{name}.joblib")
    summary = json.loads((MODELS_DIR / f"{name}_summary.json").read_text(encoding="utf-8"))["metrics"]
    X, y = held_out(name)
    p = bundle["model"].predict_proba(X)[:, 1]
    auc = float(roc_auc_score(y, p))
    if abs(auc - summary["test"]["roc_auc"]) > 1e-4:
        raise RuntimeError(f"{name}: scored AUC {auc:.4f}, summary says {summary['test']['roc_auc']}")
    grid = threshold_grid(p, float(bundle["threshold"]))
    families = [{k: row[k] for k in ("family", "roc_auc", "roc_auc_std")}
                for row in summary["comparison"] if not row.get("ablation")]
    labels = {spec["name"]: spec["label"] for spec in bundle.get("feature_specs", [])}
    importance = [{"label": labels[row["feature"]], "auc_drop": row["auc_drop"]}
                  for row in summary.get("permutation_importance", [])[:8]]
    return {
        "auc": round(auc, 4),
        "auc_ci": summary["test_ci95"]["roc_auc"],
        "threshold": float(bundle["threshold"]),
        "positives": int(y.sum()),
        "negatives": int(len(y) - y.sum()),
        "grid": grid.tolist(),
        **threshold_counts(y, p, grid),
        "histogram": histogram(y, p),
        "calibration": calibration_table(y, p),
        "families": families,
        "importance": importance,
    }


def slim_transfer(t: dict[str, Any]) -> dict[str, Any]:
    keep = ("name", "family", "sizes", "source_rows", "target_rows", "target_minority", "target_rate",
            "mean_predicted", "calibration_slope", "auc_drop_ci", "oe_ci", "repair_size")
    out = {k: t[k] for k in keep}
    out["source_auc"], out["target_auc"] = t["source"]["auc"], t["target"]["auc"]
    out["sensitivity"] = t["target"]["sensitivity"]
    out["curves"] = {
        method: {n: {m: row[m] for m in CURVE_METRICS} for n, row in by_n.items()}
        for method, by_n in t["curves"].items()
    }
    return out


def main() -> int:
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    audits = json.loads((RESULTS / "audits.json").read_text(encoding="utf-8"))
    data = {
        "study_commit": report["provenance"]["commit"],
        "models": {name: model_panel(name) for name in (*LOADERS, "malaria")},
        "verdicts": report["verdicts"],
        "transfers": [slim_transfer(t) for t in report["transfers"]],
        "audits": {k: audits[k] for k in ("missingness", "site")},
    }
    TARGET.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {TARGET.relative_to(ROOT).as_posix()}, {TARGET.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
