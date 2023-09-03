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
