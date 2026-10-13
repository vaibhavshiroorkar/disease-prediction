"""Malaria from blood smear images: is this red blood cell infected?

The data is the NIH collection of thin blood smear cells: 27,558 crops of
single red blood cells, half of them carrying P. falciparum. The parasite
takes up the Giemsa stain, so an infected cell shows a dark purple ring or dot
against the pale pink of a healthy one.

There is no neural network here. Each cell becomes 181 hand-built numbers:
colour statistics, a hue histogram, how much of the cell is much darker than
its own median (the stained parasite), and gradient-orientation histograms on
a 4x4 grid (a small HOG) for shape. The extractor is in app/cells.py, because
the API runs it on every upload. scikit-learn models take it from there.

The split is by patient. Cells from one patient share a staining batch, a
microscope and a camera, so a model can learn "this patient's slide" instead of
"this cell's parasite". A random split by cell lets that leak in and flatters
the score. Training measures exactly how much.

    python -m ml.train --model malaria
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.cells import FEATURE_NAMES, FEATURE_VERSION, cell_features
from ml import config


def patient_of(path: str) -> str:
    """'.../C100P61ThinF_IMG_..._cell_162.png' -> 'C100'.

    The C number is the slide, one per patient. The rest of the prefix is
    spelled several ways in the data (C37BP2_thinF, C42P5thin_original,
    C12NThinF), but the C number is always first.
    """
    match = re.match(r"C\d+", path.rsplit("/", 1)[-1])
    if not match:
        raise ValueError(f"no patient id in {path}")
    return match.group(0)


@dataclass
class CellData:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    paths: np.ndarray
    notes: list[str] = field(default_factory=list)


def _features_for(args: tuple[str, list[str]]) -> np.ndarray:
    zip_path, members = args
    with zipfile.ZipFile(zip_path) as archive:
        return np.stack([cell_features(Image.open(io.BytesIO(archive.read(m)))) for m in members])


def load_malaria(workers: int = 8) -> CellData:
    """Read every cell from the zip, without unpacking it, and cache the features.

    Extraction takes about 15 seconds on 8 cores. The cache is keyed on the
    zip's size and the feature version, so a new download or a changed
    extractor rebuilds it.
    """
    source = config.SOURCES["malaria_cells"]
    if not source.path.exists():
        raise FileNotFoundError(f"{source.path} is missing. Run: python -m scripts.download_data")
    cache = config.RAW_DIR / f"malaria_features_v{FEATURE_VERSION}_{source.path.stat().st_size}.npz"
    notes: list[str] = []

    if cache.exists():
        stored = np.load(cache)
        X, y, paths = stored["X"], stored["y"], stored["paths"]
        notes.append(f"features read from {cache.name}")
    else:
        with zipfile.ZipFile(source.path) as archive:
            paths = np.array(sorted(n for n in archive.namelist() if n.endswith(".png")))
        y = np.array([p.split("/")[-2] == "Parasitized" for p in paths], dtype=int)
        started = time.perf_counter()
        chunks = [list(paths[i::workers * 4]) for i in range(workers * 4)]
        with ProcessPoolExecutor(workers) as pool:
            parts = list(pool.map(_features_for, [(str(source.path), c) for c in chunks]))
        X = np.zeros((len(paths), len(FEATURE_NAMES)), dtype=np.float32)
        for i, part in enumerate(parts):
            X[i::workers * 4] = part
        notes.append(f"extracted {X.shape[1]} features from {len(paths)} cells in "
                     f"{time.perf_counter() - started:.0f}s")
        np.savez_compressed(cache, X=X, y=y, paths=paths)

    groups = np.array([patient_of(p) for p in paths])
    healthy = [g for g in np.unique(groups) if y[groups == g].sum() == 0]
    notes.append(
        f"{len(paths)} cells, {y.mean():.0%} parasitized, from {len(np.unique(groups))} patients "
        f"({len(healthy)} with no parasitized cell)"
    )
    return CellData(X=X, y=y, groups=groups, paths=paths, notes=notes)


# Training ----------------------------------------------------------------

FOLDS = 5


def _families(seed: int) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "dummy": (Pipeline([("model", DummyClassifier(strategy="prior"))]), {}),
        "logistic": (
            Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=5000))]),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "boosting": (
            Pipeline([("model", HistGradientBoostingClassifier(
                random_state=seed, early_stopping=False, learning_rate=0.1, l2_regularization=1.0))]),
            {"model__max_iter": [200, 400], "model__max_leaf_nodes": [15, 31]},
        ),
    }


def _search(pipeline, grid, X, y, groups):
    """Tune on whole patients: the inner folds are grouped too."""
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV, GroupKFold

    if not grid:
        return clone(pipeline).fit(X, y), {}
    search = GridSearchCV(clone(pipeline), grid, scoring="roc_auc", cv=GroupKFold(n_splits=3), n_jobs=1)
    search.fit(X, y, groups=groups)
    return search.best_estimator_, {k.removeprefix("model__"): v for k, v in search.best_params_.items()}


def _outer_fold(family, pipeline, grid, X, y, groups, train, test):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    started = time.perf_counter()
    model, _ = _search(pipeline, grid, X[train], y[train], groups[train])
    proba = model.predict_proba(X[test])[:, 1]
    return family, {
        "roc_auc": roc_auc_score(y[test], proba),
        "pr_auc": average_precision_score(y[test], proba),
        "brier": brier_score_loss(y[test], proba),
        "seconds": time.perf_counter() - started,
    }


def compare_families(X, y, groups, seed: int = config.RANDOM_SEED) -> list[dict[str, Any]]:
    """Nested CV with patients kept whole in both loops, best first."""
    from joblib import Parallel, delayed
    from sklearn.model_selection import StratifiedGroupKFold

    outer = list(StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed).split(X, y, groups))
    jobs = [
        delayed(_outer_fold)(family, pipeline, grid, X, y, groups, train, test)
        for family, (pipeline, grid) in _families(seed).items()
        for train, test in outer
    ]
    results = Parallel(n_jobs=-1)(jobs)
    rows = []
    for family in _families(seed):
        folds = [r for f, r in results if f == family]
        auc = [f["roc_auc"] for f in folds]
        rows.append({
            "family": family,
            "roc_auc": round(float(np.mean(auc)), 4),
            "roc_auc_std": round(float(np.std(auc)), 4),
            "pr_auc": round(float(np.mean([f["pr_auc"] for f in folds])), 4),
            "brier": round(float(np.mean([f["brier"] for f in folds])), 4),
            "fit_seconds": round(float(np.sum([f["seconds"] for f in folds])), 1),
            "ablation": False,
        })
    return sorted(rows, key=lambda row: row["roc_auc"], reverse=True)


def _calibrated(best, X, y, groups, seed: int):
    """Sigmoid calibration on grouped folds, so calibration never sees a patient twice."""
    from sklearn.base import clone
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GroupKFold

    splits = list(GroupKFold(n_splits=FOLDS).split(X, y, groups))
    return CalibratedClassifierCV(clone(best), method="sigmoid", cv=splits).fit(X, y)


def grouped_oof(best, X, y, groups, seed: int = config.RANDOM_SEED) -> np.ndarray:
    """Calibrated probabilities for each training cell from a model that never saw its patient."""
    from joblib import Parallel, delayed
    from sklearn.model_selection import GroupKFold

    def fold(train, test):
        model = _calibrated(best, X[train], y[train], groups[train], seed)
        return test, model.predict_proba(X[test])[:, 1]

    oof = np.zeros(len(y))
    splits = GroupKFold(n_splits=FOLDS).split(X, y, groups)
    for test, proba in Parallel(n_jobs=FOLDS)(delayed(fold)(tr, te) for tr, te in splits):
        oof[test] = proba
    return oof


def split_leak(best, X, y, groups, seed: int = config.RANDOM_SEED) -> dict[str, float]:
    """The same model scored two ways: cells split at random, and patients kept whole."""
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score

    random_split = cross_val_score(best, X, y, scoring="roc_auc", n_jobs=FOLDS,
                                   cv=StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed))
    by_patient = cross_val_score(best, X, y, groups=groups, scoring="roc_auc", n_jobs=FOLDS,
                                 cv=GroupKFold(n_splits=FOLDS))
    return {
        "random_cells_roc_auc": round(float(np.mean(random_split)), 4),
        "whole_patients_roc_auc": round(float(np.mean(by_patient)), 4),
    }


def per_patient(y, proba, groups, threshold) -> dict[str, Any]:
    """What the cell scores mean for a slide.

    A real smear is read by counting cells across many fields. Two numbers
    matter for that: on a healthy patient's slide, what share of cells get
    flagged (each one a false alarm), and on an infected patient's slide, what
    share of the truly infected cells get caught.
    """
    flagged = proba >= threshold
    healthy_rates, caught_rates = [], []
    for patient in np.unique(groups):
        rows = groups == patient
        if y[rows].sum() == 0:
            healthy_rates.append(float(flagged[rows].mean()))
        else:
            infected = rows & (y == 1)
            caught_rates.append(float(flagged[infected].mean()))

    def describe(values):
        if not values:
            return None
        return {
            "patients": len(values),
            "median": round(float(np.median(values)), 4),
            "worst": round(float(max(values) if values is healthy_rates else min(values)), 4),
        }

    return {
        "healthy_patients_cells_flagged": describe(healthy_rates),
        "infected_patients_parasitized_cells_caught": describe(caught_rates),
    }


def example_grid(paths, y, proba, threshold, out_dir: Path, per_row: int = 6) -> Path:
    """The worst misses and the worst false alarms on the test set, as pictures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = config.SOURCES["malaria_cells"].path
    missed = [i for i in np.argsort(proba) if y[i] == 1 and proba[i] < threshold][:per_row]
    alarms = [i for i in np.argsort(-proba) if y[i] == 0 and proba[i] >= threshold][:per_row]
    caught = [i for i in np.argsort(-proba) if y[i] == 1][:per_row]
    rows = [("Caught", caught), ("Missed", missed), ("False alarm", alarms)]

    fig, axes = plt.subplots(len(rows), per_row, figsize=(per_row * 1.6, len(rows) * 1.9))
    with zipfile.ZipFile(source) as archive:
        for r, (title, indices) in enumerate(rows):
            for c in range(per_row):
                ax = axes[r][c]
                ax.axis("off")
                if c < len(indices):
                    i = indices[c]
                    ax.imshow(Image.open(io.BytesIO(archive.read(str(paths[i])))).convert("RGB"))
                    ax.set_title(f"{proba[i]:.2f}", fontsize=8)
            axes[r][0].text(-0.15, 0.5, title, transform=axes[r][0].transAxes, ha="right", va="center",
                            fontsize=9)
    fig.suptitle("malaria: test cells with their predicted probability of infection", fontsize=10)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "malaria_examples.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def train_malaria(figures_dir: Path) -> dict[str, Any]:
    import sklearn
    from sklearn.model_selection import StratifiedGroupKFold

    from ml.clinical_eval import (
        TARGET_SENSITIVITY,
        binary_scores,
        bootstrap_ci,
        calibration_table,
        choose_threshold,
        figures,
        rounded,
    )
    from ml.clinical_model import pick

    seed = config.RANDOM_SEED
    print("[malaria] loading")
    data = load_malaria()
    for note in data.notes:
        print(f"    - {note}")
    started = time.perf_counter()

    # One fold of five, by patient, is the test set: about 40 patients.
    train, test = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                       .split(data.X, data.y, data.groups))
    X_train, y_train, g_train = data.X[train], data.y[train], data.groups[train]
    X_test, y_test, g_test = data.X[test], data.y[test], data.groups[test]
    print(f"    - split by patient: {len(np.unique(g_train))} patients ({len(train)} cells) train, "
          f"{len(np.unique(g_test))} patients ({len(test)} cells) test")

    print("[malaria] comparing model families (nested CV, patients kept whole)")
    table = compare_families(X_train, y_train, g_train, seed)
    for row in table:
        print(f"    {row['family']:10s} AUC {row['roc_auc']:.4f} +/- {row['roc_auc_std']:.4f}"
              f"  PR-AUC {row['pr_auc']:.4f}  Brier {row['brier']:.4f}")
    family = pick(table)
    print(f"    chose {family}")

    pipeline, grid = _families(seed)[family]
    best, params = _search(pipeline, grid, X_train, y_train, g_train)
    print(f"    settings {params}")

    print("[malaria] how much a random split by cell would flatter the score")
    leak = split_leak(best, X_train, y_train, g_train, seed)
    print(f"    {leak}")

    print("[malaria] choosing the threshold on grouped out-of-fold predictions")
    oof = grouped_oof(best, X_train, y_train, g_train, seed)
    threshold = choose_threshold(y_train, oof, TARGET_SENSITIVITY)
    print(f"    flag at p >= {threshold:.3f} to catch {TARGET_SENSITIVITY:.0%} of infected cells")

    model = _calibrated(best, X_train, y_train, g_train, seed)
    proba = model.predict_proba(X_test)[:, 1]

    print("[malaria] scoring the held-out patients")
    scores = rounded(binary_scores(y_test, proba, threshold))
    intervals = bootstrap_ci(y_test, proba, threshold, groups=g_test)
    calibration = calibration_table(y_test, proba)
    for key in ["roc_auc", "pr_auc", "brier", "sensitivity", "specificity", "ppv", "npv"]:
        low, high = intervals[key]
        print(f"    {key:12s} {scores[key]:.3f}  [{low:.3f}, {high:.3f}]")
    print(f"    ece          {calibration['ece']:.3f}")
    patients = per_patient(y_test, proba, g_test, threshold)
    print(f"    {patients}")

    for path in [
        *figures("malaria", y_test, proba, threshold, figures_dir, unit="cells"),
        example_grid(data.paths[test], y_test, proba, threshold, figures_dir),
    ]:
        print(f"    drew {path.relative_to(config.ROOT).as_posix()}")

    metrics = {
        "comparison": table,
        "chosen_family": family,
        "chosen_params": params,
        "threshold": {
            "value": round(threshold, 4),
            "target_sensitivity": TARGET_SENSITIVITY,
            "chosen_on": "grouped out-of-fold predictions on the training patients",
        },
        "test": {**scores, "rows": int(len(y_test)), "patients": int(len(np.unique(g_test))),
                 "prevalence": round(float(y_test.mean()), 4)},
        "test_ci95": intervals,
        "test_ci95_method": "bootstrap over patients, not cells",
        "calibration": calibration,
        "per_patient": patients,
        "split_leak": leak,
    }
    source = config.SOURCES["malaria_cells"]
    return {
        "artifact_version": config.ARTIFACT_VERSION,
        "name": "malaria",
        "kind": "image",
        "model": model,
        "feature_names": FEATURE_NAMES,
        "feature_version": FEATURE_VERSION,
        "classes": [0, 1],
        "target": "red blood cell parasitized by P. falciparum",
        "threshold": threshold,
        "metrics": metrics,
        "dataset": {
            "sources": [source.url],
            "rows": int(len(data.y)),
            "rows_train": int(len(train)),
            "rows_test": int(len(test)),
            "patients": int(len(np.unique(data.groups))),
            "prevalence": round(float(data.y.mean()), 4),
            "cleaning": data.notes,
        },
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "train_seconds": round(time.perf_counter() - started, 1),
    }

