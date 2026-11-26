"""SINAN dengue by state and year: reading, the inclusion rule, source rows.

Protocol: docs/research/protocol-states.md. The constants here are copied from
it and must not change after a run has started.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml import config
from ml.clinical_data import DENGUE_COLUMNS, DENGUE_FEATURES, DENGUE_SEVERE, dengue_confirmed, dengue_inputs

SOURCE_YEAR = 2023
STAGE_A = (2023,)
STAGE_B = (2024, 2025)
MIN_CASES, MIN_SEVERE = 1_000, 100
SOURCE_SAMPLE = 200_000
CHUNK_ROWS = 500_000

# IBGE state codes, for readable target names.
STATES = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
    "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL",
    "28": "SE", "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP", "41": "PR",
    "42": "SC", "43": "RS", "50": "MS", "51": "MT", "52": "GO", "53": "DF",
}
FEATURE_NAMES = [f.name for f in DENGUE_FEATURES]


@dataclass
class Year:
    X: pd.DataFrame
    y: np.ndarray
    state: np.ndarray


def target_name(year: int, state: str) -> str:
    return f"{STATES.get(state, state)} {year}"


def read_year(year: int) -> Year:
    """Lab-confirmed cases of one year, built chunk by chunk to keep memory low."""
    source = config.SOURCES[f"dengue_{year}"]
    if not source.path.exists():
        raise FileNotFoundError(f"{source.path} is missing. Run: python -m scripts.download_data")
    parts = []
    for chunk in pd.read_csv(source.path, usecols=DENGUE_COLUMNS, dtype=str, chunksize=CHUNK_ROWS):
        confirmed = dengue_confirmed(chunk)
        X = dengue_inputs(confirmed)[FEATURE_NAMES].astype("float32")
        X["severe"] = confirmed["CLASSI_FIN"].isin(DENGUE_SEVERE).astype("int8")
        X["state"] = confirmed["SG_UF_NOT"]
        parts.append(X)
    rows = pd.concat(parts, ignore_index=True)
    return Year(rows[FEATURE_NAMES], rows["severe"].to_numpy(), rows["state"].to_numpy(dtype=str))


def read_years(years: set[int] | list[int] | tuple[int, ...]) -> dict[int, Year]:
    return {year: read_year(year) for year in sorted(set(years))}


def choose_targets(counts: pd.DataFrame) -> tuple[list[tuple[int, str]], list[dict[str, Any]]]:
    """Apply the inclusion rule to a table with year, state, cases and severe columns."""
    keep = (counts["cases"] >= MIN_CASES) & (counts["severe"] >= MIN_SEVERE)
    chosen = [(int(r.year), str(r.state)) for r in counts[keep].itertuples()]
    excluded = [
        {"year": int(r.year), "state": STATES.get(str(r.state), str(r.state)),
         "cases": int(r.cases), "severe": int(r.severe)}
        for r in counts[~keep].itertuples()
    ]
    return sorted(chosen), excluded


def count_cases(years: dict[int, Year]) -> pd.DataFrame:
    frames = [
        pd.DataFrame({"year": year, "state": data.state, "severe": data.y})
        .groupby(["year", "state"], as_index=False)
        .agg(cases=("severe", "size"), severe=("severe", "sum"))
        for year, data in years.items()
    ]
    return pd.concat(frames, ignore_index=True)


def source_rows(source: Year, state: str, seed: int = config.RANDOM_SEED) -> tuple[pd.DataFrame, pd.Series]:
    """Every other state, stratified down to SOURCE_SAMPLE rows."""
    away = np.flatnonzero(source.state != state)
    if len(away) > SOURCE_SAMPLE:
        away, _ = train_test_split(away, train_size=SOURCE_SAMPLE, stratify=source.y[away], random_state=seed)
        away = np.sort(away)
    return source.X.iloc[away].reset_index(drop=True), pd.Series(source.y[away])


def target_rows(year: Year, state: str) -> tuple[pd.DataFrame, np.ndarray]:
    held = year.state == state
    return year.X[held].reset_index(drop=True), year.y[held]


def unclassified_share(year: int) -> float:
    """Share of notified rows with no final class yet. High in a preliminary file."""
    source = config.SOURCES[f"dengue_{year}"]
    rows = missing = 0
    for chunk in pd.read_csv(source.path, usecols=["CLASSI_FIN"], dtype=str, chunksize=CHUNK_ROWS):
        rows += len(chunk)
        missing += int(chunk["CLASSI_FIN"].isna().sum())
    return missing / rows
