"""Tests for loading and cleaning.

These encode the specific defects found in the raw files. If a loader silently
stops fixing one of them, the model degrades in a way that is hard to see from
the accuracy number alone.
"""


import pandas as pd
import pytest

from ml.config import SOURCES
from ml.datasets import Dataset, load_symptoms, normalize_symptom_name, unique_symptom_patterns

raw_present = pytest.mark.skipif(
    not all(source.path.exists() for source in SOURCES.values()),
    reason="raw data missing, run: python -m scripts.download_data",
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("dischromic _patches", "dischromic_patches"),
        ("spotting_ urination", "spotting_urination"),
        ("  Itching  ", "itching"),
        ("foul_smell_of urine", "foul_smell_of_urine"),
    ],
)
def test_normalize_symptom_name_fixes_published_header(raw: str, expected: str) -> None:
    assert normalize_symptom_name(raw) == expected


@raw_present
def test_symptom_loader_drops_stray_and_constant_columns() -> None:
    data = load_symptoms("train")
    assert not [c for c in data.feature_names if c.startswith("Unnamed")]
    # 'fluid_overload' is all zeros in the published file; its duplicate carries the signal.
    assert "fluid_overload" not in data.feature_names
    assert "fluid_overload.1" in data.feature_names
    assert data.X.shape == (4920, 131)
    assert data.y.nunique() == 41


@raw_present
def test_symptom_matrix_is_binary() -> None:
    data = load_symptoms("train")
    assert set(pd.unique(data.X.to_numpy().ravel())) <= {0, 1}


@raw_present
def test_deduplication_collapses_the_repeated_patterns() -> None:
    """The published file pads every disease to 120 rows by repeating patterns.

    Splitting before this collapse puts identical vectors in train and test,
    which is what produces the 100% accuracy this dataset is known for.
    """
    data = load_symptoms("train")
    unique = unique_symptom_patterns(data)
    assert len(unique.X) < len(data.X) / 10
    assert unique.y.nunique() == 41
    assert not unique.X.duplicated().any() or len(unique.X.drop_duplicates()) == len(unique.X)


@raw_present
def test_every_disease_survives_deduplication_with_enough_rows_for_cv() -> None:
    unique = unique_symptom_patterns(load_symptoms("train"))
    counts = unique.y.value_counts()
    assert counts.min() >= 5, "5-fold stratified CV needs at least 5 patterns per class"


@raw_present
def test_test_split_columns_align_with_training() -> None:
    """The 42-row holdout must reindex onto the training feature order."""
    train = unique_symptom_patterns(load_symptoms("train"))
    test = load_symptoms("test")
    aligned = test.X.reindex(columns=train.feature_names, fill_value=0)
    assert list(aligned.columns) == train.feature_names
    assert len(aligned) == 42


def test_dataset_repr_is_informative() -> None:
    data = Dataset(X=pd.DataFrame({"a": [0, 1]}), y=pd.Series(["x", "y"]), name="demo")
    assert "demo" in repr(data)
