"""Load and clean the raw CSVs into model-ready frames.

Each loader returns tidy data plus the quirks it had to fix, so the training
script can print what was changed instead of silently reshaping the input.
"""


from dataclasses import dataclass, field

import pandas as pd

from ml.config import SOURCES


@dataclass
class Dataset:
    """Features, labels, and a record of what cleaning did."""

    X: pd.DataFrame
    y: pd.Series
    name: str
    notes: list[str] = field(default_factory=list)

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    def __repr__(self) -> str:
        return f"<Dataset {self.name} X={self.X.shape} classes={self.y.nunique()}>"


def _require(key: str) -> pd.DataFrame:
    source = SOURCES[key]
    if not source.path.exists():
        raise FileNotFoundError(
            f"{source.path} is missing. Run: python -m scripts.download_data"
        )
    return pd.read_csv(source.path)


def normalize_symptom_name(raw: str) -> str:
    """Fix the whitespace damage in the published header.

    The source ships names like 'dischromic _patches' and 'spotting_ urination'.
    Collapsing spaces and underscores keeps the API's symptom keys predictable.
    """
    cleaned = raw.strip().lower().replace(" ", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def load_symptoms(split: str = "train") -> Dataset:
    """Binary symptom matrix -> disease label.

    The published file has an unnamed trailing column from a stray comma and a
    duplicated 'fluid_overload' that pandas renames to 'fluid_overload.1'. The
    all-zero copy carries no signal, so constant columns are dropped.
    """
    key = "symptom_train" if split == "train" else "symptom_test"
    df = _require(key)
    notes: list[str] = []

    stray = [c for c in df.columns if c.startswith("Unnamed")]
    if stray:
        df = df.drop(columns=stray)
        notes.append(f"dropped {len(stray)} unnamed trailing column(s)")

    df.columns = [c if c == "prognosis" else normalize_symptom_name(c) for c in df.columns]

    y = df["prognosis"].str.strip()
    X = df.drop(columns=["prognosis"]).astype("int8")

    constant = [c for c in X.columns if X[c].nunique() == 1]
    if constant and split == "train":
        X = X.drop(columns=constant)
        notes.append(f"dropped {len(constant)} constant column(s): {', '.join(constant)}")

    notes.append(f"{len(X)} rows, {X.shape[1]} symptoms, {y.nunique()} diseases")
    return Dataset(X=X, y=y, name=f"symptoms_{split}", notes=notes)


def unique_symptom_patterns(dataset: Dataset) -> Dataset:
    """Collapse the symptom data to its distinct (pattern, disease) rows.

    The published file repeats each disease's handful of symptom patterns until
    every class has exactly 120 rows. 4920 rows reduce to ~300 distinct ones, so
    a random split puts identical vectors on both sides and scores ~100%. Every
    honest estimate in this project is measured after this collapse.
    """
    combined = dataset.X.copy()
    combined["__label__"] = dataset.y.to_numpy()
    deduped = combined.drop_duplicates().reset_index(drop=True)

    y = deduped["__label__"]
    X = deduped.drop(columns=["__label__"])
    notes = [f"collapsed {len(dataset.X)} rows to {len(X)} unique symptom patterns"]
    return Dataset(X=X, y=y, name=f"{dataset.name}_unique", notes=notes)
