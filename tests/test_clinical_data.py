"""Tests for the clinical loaders.

Each test writes a few rows that reproduce one defect in the published files,
then checks the loader fixes it. No network and no downloaded data needed.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest

from ml import config
from ml.clinical_data import (
    DENGUE_COLUMNS,
    DENGUE_FEATURES,
    DIABETES_FEATURES,
    HEART_FEATURES,
    LIVER_FEATURES,
    LOADERS,
    load_dengue,
    load_diabetes,
    load_heart,
    load_liver,
)

raw_present = pytest.mark.skipif(
    not all(source.path.exists() for source in config.SOURCES.values()),
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


@pytest.mark.parametrize("features", [HEART_FEATURES, DIABETES_FEATURES, LIVER_FEATURES, DENGUE_FEATURES])
def test_specs_are_well_formed(features) -> None:
    names = [spec.name for spec in features]
    assert len(names) == len(set(names))
    for spec in features:
        assert spec.kind in {"numeric", "binary", "category"}
        if spec.kind == "numeric":
            assert spec.minimum is not None and spec.maximum is not None and spec.minimum < spec.maximum
        else:
            assert spec.choices


def _sinan_row(**overrides: str) -> dict[str, str]:
    """One notification as SINAN writes it: every sign answered 'no' (2)."""
    row = {column: "2" for column in DENGUE_COLUMNS}
    row.update(NU_IDADE_N="4031", CS_SEXO="F", CS_GESTANT="5", SG_UF_NOT="35", DT_NOTIFIC="2023-03-10",
               DT_SIN_PRI="2023-03-07", CLASSI_FIN="10", CRITERIO="1")
    row.update(overrides)
    return row


def _write_sinan(raw_dir, year: int, rows: list[dict[str, str]]) -> None:
    path = raw_dir / f"dengue_sinan_{year}.csv.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"DENGBR{year % 100}.csv", pd.DataFrame(rows).to_csv(index=False))


def test_dengue_keeps_only_lab_confirmed_cases_and_labels_severity(raw_dir) -> None:
    _write_sinan(raw_dir, 2023, [
        _sinan_row(),                               # plain dengue
        _sinan_row(CLASSI_FIN="11"),                # warning signs
        _sinan_row(CLASSI_FIN="12"),                # severe
        _sinan_row(CRITERIO="2"),                   # confirmed on symptoms only: dropped
        _sinan_row(CLASSI_FIN="8"),                 # inconclusive: dropped
    ])
    data = load_dengue(sample=None)
    assert data.y.tolist() == [0, 1, 1]
    assert data.external is None
    assert data.feature_names == [spec.name for spec in DENGUE_FEATURES]


def test_dengue_decodes_sinan_fields(raw_dir) -> None:
    _write_sinan(raw_dir, 2023, [
        _sinan_row(NU_IDADE_N="3006", CS_SEXO="M", CS_GESTANT="6", SG_UF_NOT="13", FEBRE="1", LACO="9"),
        _sinan_row(CS_GESTANT="2", DT_SIN_PRI="2022-11-01", SG_UF_NOT="43"),
        _sinan_row(CS_GESTANT="9", NU_IDADE_N="4999"),
    ])
    data = load_dengue(sample=None)
    first, second, third = (data.X.iloc[i] for i in range(3))
    assert first["age"] == 0.5                     # 6 months
    assert first["sex"] == 1 and first["pregnant"] == 0
    assert first["fever"] == 1 and np.isnan(first["tourniquet_test"])
    assert first["days_since_onset"] == 3
    assert second["pregnant"] == 1
    assert np.isnan(second["days_since_onset"])    # 129 days is a data entry error
    assert np.isnan(third["pregnant"]) and np.isnan(third["age"])
    assert data.groups.tolist() == ["north", "south", "southeast"]


def test_dengue_later_year_becomes_the_external_test_set(raw_dir) -> None:
    _write_sinan(raw_dir, 2023, [_sinan_row(), _sinan_row(CLASSI_FIN="11")])
    _write_sinan(raw_dir, 2025, [_sinan_row(CLASSI_FIN="12"), _sinan_row(), _sinan_row()])
    data = load_dengue(sample=None, external_sample=None)
    assert data.external.label == "2025"
    assert data.external.y.tolist() == [1, 0, 0]
    assert list(data.external.X.columns) == data.feature_names


def test_dengue_sample_keeps_the_label_share(raw_dir) -> None:
    rows = [_sinan_row(CLASSI_FIN="11" if i % 10 == 0 else "10") for i in range(400)]
    _write_sinan(raw_dir, 2023, rows)
    data = load_dengue(sample=100)
    assert len(data.y) == 100
    assert data.y.sum() == 10


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
