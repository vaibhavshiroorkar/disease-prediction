"""Tests for the clinical loaders.

Each test writes a few rows that reproduce one defect in the published files,
then checks the loader fixes it. No network and no downloaded data needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml import config
from ml.clinical_data import (
    DIABETES_FEATURES,
    HEART_FEATURES,
    LIVER_FEATURES,
    LOADERS,
    load_diabetes,
    load_heart,
    load_liver,
)

raw_present = pytest.mark.skipif(
    not all(config.SOURCES[key].path.exists() for key in ["diabetes", "liver", "heart_cleveland"]),
    reason="raw data missing, run: python -m scripts.download_data",
)


@pytest.fixture()
def raw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    return tmp_path


def _write_heart(raw_dir, rows_by_site: dict[str, list[str]]) -> None:
    for site in config.HEART_SITES:
        (raw_dir / f"heart_{site}.data").write_text("\n".join(rows_by_site.get(site, [])) + "\n")


HEART_ROW = "63,1,1,145,233,1,2,150,0,2.3,3,0,6,0"


def test_heart_marks_question_marks_and_zero_cholesterol_as_missing(raw_dir) -> None:
    _write_heart(
        raw_dir,
        {
            "cleveland": [HEART_ROW, "67,1,4,160,286,0,2,108,1,1.5,2,3,3,2"],
            "switzerland": ["32,1,1,95,0,?,0,127,0,.7,1,?,?,1"],
            "va": ["63,1,4,?,260,0,1,112,1,3,2,?,?,2"],
        },
    )
    data = load_heart()
    swiss = data.X[data.groups == "switzerland"].iloc[0]
    assert np.isnan(swiss["chol"])
    assert np.isnan(swiss["fbs"])
    assert np.isnan(data.X[data.groups == "va"].iloc[0]["trestbps"])


def test_heart_collapses_severity_to_presence(raw_dir) -> None:
    rows = [HEART_ROW.rsplit(",", 1)[0] + f",{num}" for num in range(5)]
    rows = [r.replace("63,", f"{40 + i},", 1) for i, r in enumerate(rows)]  # keep rows distinct
    _write_heart(raw_dir, {"cleveland": rows})
    assert load_heart().y.tolist() == [0, 1, 1, 1, 1]


def test_heart_drops_the_columns_that_identify_the_hospital(raw_dir) -> None:
    _write_heart(raw_dir, {"cleveland": [HEART_ROW]})
    data = load_heart()
    assert not {"slope", "ca", "thal", "num", "site"} & set(data.feature_names)
    assert data.missing_indicators is False


def test_heart_keeps_the_site_of_every_row(raw_dir) -> None:
    _write_heart(raw_dir, {"cleveland": [HEART_ROW], "hungarian": ["28,1,2,130,132,0,2,185,0,0,?,?,?,0"]})
    assert load_heart().groups.tolist() == ["cleveland", "hungarian"]


def test_heart_drops_exact_duplicates(raw_dir) -> None:
    _write_heart(raw_dir, {"cleveland": [HEART_ROW, HEART_ROW]})
    data = load_heart()
    assert len(data.X) == 1
    assert any("duplicate" in note for note in data.notes)


def test_diabetes_treats_impossible_zeros_as_missing(raw_dir) -> None:
    (raw_dir / "pima_diabetes.csv").write_text("6,148,72,35,0,33.6,0.627,50,1\n1,0,66,29,94,0,0.351,31,0\n")
    data = load_diabetes()
    assert np.isnan(data.X.loc[0, "insulin"])
    assert np.isnan(data.X.loc[1, "glucose"])
    assert np.isnan(data.X.loc[1, "bmi"])
    # Zero pregnancies is a real answer and must survive.
    (raw_dir / "pima_diabetes.csv").write_text("0,148,72,35,80,33.6,0.627,50,1\n")
    assert load_diabetes().X.loc[0, "pregnancies"] == 0


def test_liver_maps_labels_and_sex(raw_dir) -> None:
    (raw_dir / "ilpd_liver.csv").write_text(
        "65,Female,0.7,0.1,187,16,18,6.8,3.3,0.9,1\n62,Male,10.9,5.5,699,64,100,7.5,3.2,,2\n"
    )
    data = load_liver()
    assert data.y.tolist() == [1, 0]
    assert data.X["sex"].tolist() == [0, 1]
    assert np.isnan(data.X.loc[1, "ag_ratio"])


def test_liver_drops_exact_duplicates(raw_dir) -> None:
    row = "65,Female,0.7,0.1,187,16,18,6.8,3.3,0.9,1"
    (raw_dir / "ilpd_liver.csv").write_text(f"{row}\n{row}\n")
    assert len(load_liver().X) == 1


def test_liver_rejects_an_unknown_sex(raw_dir) -> None:
    (raw_dir / "ilpd_liver.csv").write_text("65,Other,0.7,0.1,187,16,18,6.8,3.3,0.9,1\n")
    with pytest.raises(ValueError, match="sex"):
        load_liver()


def test_missing_file_says_how_to_fetch_it(raw_dir) -> None:
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_diabetes()


@pytest.mark.parametrize("features", [HEART_FEATURES, DIABETES_FEATURES, LIVER_FEATURES])
def test_specs_are_well_formed(features) -> None:
    names = [spec.name for spec in features]
    assert len(names) == len(set(names))
    for spec in features:
        assert spec.kind in {"numeric", "binary", "category"}
        if spec.kind == "numeric":
            assert spec.minimum is not None and spec.maximum is not None and spec.minimum < spec.maximum
        else:
            assert spec.choices


@raw_present
@pytest.mark.parametrize("name", list(LOADERS))
def test_every_real_value_fits_the_published_ranges(name: str) -> None:
    """The API validates against the spec ranges, so they must admit the training data."""
    data = LOADERS[name]()
    for spec in data.features:
        column = data.X[spec.name].dropna()
        if spec.kind == "numeric":
            assert column.min() >= spec.minimum, spec.name
            assert column.max() <= spec.maximum, spec.name
        else:
            assert set(column.astype(int)) <= set(spec.choices), spec.name


@raw_present
def test_real_heart_data_shape() -> None:
    data = load_heart()
    assert len(data.X) == 918
    assert set(data.groups) == set(config.HEART_SITES)
    assert data.X.loc[data.groups == "switzerland", "chol"].isna().all()
