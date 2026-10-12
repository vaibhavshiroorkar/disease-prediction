"""Load and clean the clinical datasets behind the binary risk models.

Same contract as ml/datasets.py: each loader returns tidy data plus a list of
what it had to fix, so training prints every change it made to the input.

Missing values stay as NaN here. Imputation belongs inside the model pipeline,
where it is fit on training folds only. Filling gaps in the loader would leak
statistics from the test rows into the training rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml import config


@dataclass(frozen=True)
class FeatureSpec:
    """One model input, described well enough to validate and to label in a UI.

    The API builds its request schemas from these, so the ranges here are the
    ranges a client is allowed to send. They are set wide enough to admit every
    value in the training data and narrow enough to reject typos like a
    cholesterol of 2400.
    """

    name: str
    label: str
    kind: str  # "numeric", "binary" or "category"
    description: str
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    choices: dict[int, str] | None = None
    required: bool = False
    # Long right tail (lab enzymes, insulin). The pipeline takes log1p first so
    # one patient with an ALT of 2000 does not dominate a linear model.
    skewed: bool = False

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "description": self.description,
            "required": self.required,
        }
        if self.unit:
            out["unit"] = self.unit
        if self.minimum is not None:
            out["minimum"] = self.minimum
        if self.maximum is not None:
            out["maximum"] = self.maximum
        if self.choices:
            out["choices"] = {str(k): v for k, v in self.choices.items()}
        return out


@dataclass
class ClinicalDataset:
    """Features, a 0/1 target, optional site groups, and a record of the cleaning."""

    X: pd.DataFrame
    y: pd.Series
    name: str
    target: str
    features: list[FeatureSpec]
    groups: pd.Series | None = None
    notes: list[str] = field(default_factory=list)
    # Whether the pipeline may add "this value was missing" columns. Off when
    # missingness is a proxy for something the model should not see.
    missing_indicators: bool = True

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    @property
    def prevalence(self) -> float:
        return float(self.y.mean())

    def __repr__(self) -> str:
        return f"<ClinicalDataset {self.name} X={self.X.shape} prevalence={self.prevalence:.2f}>"


def _read(key: str, **kwargs) -> pd.DataFrame:
    source = config.SOURCES[key]
    if not source.path.exists():
        raise FileNotFoundError(f"{source.path} is missing. Run: python -m scripts.download_data")
    return pd.read_csv(source.path, **kwargs)


def _zeros_to_nan(df: pd.DataFrame, columns: list[str], notes: list[str]) -> pd.DataFrame:
    """Treat 0 as 'not measured' where 0 is physically impossible."""
    for column in columns:
        zeros = int((df[column] == 0).sum())
        if zeros:
            df[column] = df[column].replace(0, np.nan)
            notes.append(f"{column}: {zeros} zero(s) treated as missing")
    return df


def _drop_duplicates(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """Drop exact repeats before any split, so a patient never sits on both sides."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < before:
        notes.append(f"dropped {before - len(df)} exact duplicate row(s)")
    return df


def _check_columns(features: list[FeatureSpec], df: pd.DataFrame) -> None:
    missing = [spec.name for spec in features if spec.name not in df.columns]
    if missing:
        raise ValueError(f"feature specs name columns the data does not have: {missing}")


HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

HEART_FEATURES = [
    FeatureSpec("age", "Age", "numeric", "Age in years.", "years", 18, 100, required=True),
    FeatureSpec("sex", "Sex", "binary", "1 for male, 0 for female.", choices={0: "female", 1: "male"},
                required=True),
    FeatureSpec(
        "cp", "Chest pain type", "category", "The kind of chest pain the patient reports.",
        choices={1: "typical angina", 2: "atypical angina", 3: "non-anginal pain", 4: "asymptomatic"},
        required=True,
    ),
    FeatureSpec("trestbps", "Resting blood pressure", "numeric", "Systolic, on admission.", "mmHg", 60, 250),
    FeatureSpec("chol", "Serum cholesterol", "numeric", "Total serum cholesterol.", "mg/dl", 80, 700),
    FeatureSpec("fbs", "High fasting blood sugar", "binary", "1 if fasting blood sugar is above 120 mg/dl.",
                choices={0: "no", 1: "yes"}),
    FeatureSpec(
        "restecg", "Resting ECG", "category", "Resting electrocardiogram result.",
        choices={0: "normal", 1: "ST-T wave abnormality", 2: "left ventricular hypertrophy"},
    ),
    FeatureSpec("thalach", "Max heart rate", "numeric", "Maximum heart rate reached in an exercise test.",
                "bpm", 50, 230),
    FeatureSpec("exang", "Exercise angina", "binary", "1 if exercise brought on angina.",
                choices={0: "no", 1: "yes"}),
    FeatureSpec("oldpeak", "ST depression", "numeric", "ST depression induced by exercise relative to rest.",
                "mm", -3, 10),
]


def load_heart() -> ClinicalDataset:
    """The four UCI heart disease sites, pooled, with the site kept as a group.

    Fixes applied:
    - '?' marks missing values in every file.
    - Cholesterol 0 means 'not recorded'. Switzerland has it for all 123
      patients and VA for 49. One VA row has a resting blood pressure of 0.
    - The target 'num' runs 0 to 4 (no disease, then increasing severity).
      Every paper on this data collapses it to 'num > 0', and so does this.
    - slope, ca and thal are dropped. They are mostly complete at Cleveland and
      50-99% missing everywhere else, so their missingness says which hospital
      a row came from. Prevalence differs by hospital (36% to 93%), so a model
      would learn the hospital rather than the patient.

    Cholesterol has the same problem in a milder form: missing for every Swiss
    patient. It is kept because it matters clinically, but the heart dataset
    sets `missing_indicators=False` so the pipeline imputes it without also
    handing the model a "cholesterol was missing" flag.
    """
    notes: list[str] = []
    frames = []
    for site in config.HEART_SITES:
        frame = _read(f"heart_{site}", names=HEART_COLUMNS, na_values="?")
        frame["site"] = site
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    notes.append(f"pooled {len(df)} patients from {len(frames)} sites")

    df = _zeros_to_nan(df, ["chol", "trestbps"], notes)
    df = df.drop(columns=["slope", "ca", "thal"])
    notes.append("dropped slope, ca, thal: their missingness identifies the hospital")
    df = _drop_duplicates(df, notes)

    _check_columns(HEART_FEATURES, df)
    names = [spec.name for spec in HEART_FEATURES]
    y = (df["num"] > 0).astype(int)
    notes.append(f"target: num > 0, prevalence {y.mean():.2f}")
    return ClinicalDataset(
        X=df[names].astype(float),
        y=y,
        name="heart",
        target="angiographic coronary artery disease (more than 50% narrowing in any major vessel)",
        features=HEART_FEATURES,
        groups=df["site"],
        notes=notes,
        missing_indicators=False,
    )


DIABETES_COLUMNS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "pedigree", "age", "outcome",
]

DIABETES_FEATURES = [
    FeatureSpec("pregnancies", "Pregnancies", "numeric", "Number of times pregnant.", None, 0, 20,
                required=True),
    FeatureSpec("glucose", "Glucose", "numeric", "Plasma glucose 2 hours into an oral glucose tolerance test.",
                "mg/dl", 40, 300),
    FeatureSpec("blood_pressure", "Diastolic blood pressure", "numeric", "Diastolic blood pressure.",
                "mmHg", 20, 140),
    FeatureSpec("skin_thickness", "Skin fold thickness", "numeric", "Triceps skin fold thickness.", "mm", 5, 100),
    FeatureSpec("insulin", "Insulin", "numeric", "2-hour serum insulin.", "uU/ml", 10, 900, skewed=True),
    FeatureSpec("bmi", "BMI", "numeric", "Body mass index.", "kg/m2", 10, 70),
    FeatureSpec("pedigree", "Diabetes pedigree", "numeric",
                "A score for family history of diabetes, from the original study.", None, 0.05, 2.5,
                skewed=True),
    FeatureSpec("age", "Age", "numeric", "Age in years.", "years", 18, 100, required=True),
]


def load_diabetes() -> ClinicalDataset:
    """Pima Indians Diabetes: women of Pima heritage aged 21 and over.

    The file has no header, and it uses 0 for 'not measured' in five columns
    where 0 cannot be a real reading. Insulin is missing for 374 of 768 women
    and skin thickness for 227. Leaving those zeros in teaches the model that
    "insulin = 0" is a real and very common value.
    """
    notes: list[str] = []
    df = _read("diabetes", header=None, names=DIABETES_COLUMNS)
    df = _zeros_to_nan(df, ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"], notes)
    df = _drop_duplicates(df, notes)

    _check_columns(DIABETES_FEATURES, df)
    names = [spec.name for spec in DIABETES_FEATURES]
    y = df["outcome"].astype(int)
    notes.append(f"{len(df)} patients, prevalence {y.mean():.2f}")
    return ClinicalDataset(
        X=df[names].astype(float),
        y=y,
        name="diabetes",
        target="diabetes diagnosed within five years of the examination",
        features=DIABETES_FEATURES,
        notes=notes,
    )


LIVER_COLUMNS = [
    "age", "sex", "total_bilirubin", "direct_bilirubin", "alk_phosphatase",
    "alt", "ast", "total_proteins", "albumin", "ag_ratio", "selector",
]

LIVER_FEATURES = [
    FeatureSpec("age", "Age", "numeric", "Age in years.", "years", 1, 100, required=True),
    FeatureSpec("sex", "Sex", "binary", "1 for male, 0 for female.", choices={0: "female", 1: "male"},
                required=True),
    FeatureSpec("total_bilirubin", "Total bilirubin", "numeric", "Total serum bilirubin.", "mg/dl", 0.1, 80,
                skewed=True),
    FeatureSpec("direct_bilirubin", "Direct bilirubin", "numeric", "Conjugated bilirubin.", "mg/dl", 0.05, 30,
                skewed=True),
    FeatureSpec("alk_phosphatase", "Alkaline phosphatase", "numeric", "ALP.", "IU/L", 20, 3000, skewed=True),
    FeatureSpec("alt", "ALT (SGPT)", "numeric", "Alanine aminotransferase.", "IU/L", 1, 3000, skewed=True),
    FeatureSpec("ast", "AST (SGOT)", "numeric", "Aspartate aminotransferase.", "IU/L", 1, 6000, skewed=True),
    FeatureSpec("total_proteins", "Total proteins", "numeric", "Total serum protein.", "g/dl", 2, 12),
    FeatureSpec("albumin", "Albumin", "numeric", "Serum albumin.", "g/dl", 0.5, 7),
    FeatureSpec("ag_ratio", "A/G ratio", "numeric", "Albumin to globulin ratio.", None, 0.1, 4),
]


def load_liver() -> ClinicalDataset:
    """Indian Liver Patient Dataset.

    No header. The label column is 1 for a liver patient and 2 for not, which
    is mapped to 1 and 0. Sex arrives as 'Male'/'Female'. 13 rows are exact
    duplicates. Four patients have no A/G ratio, which stays missing.

    The 'not a patient' group were not healthy volunteers in the usual sense,
    and the published accuracy on this data tops out around 0.70-0.75 AUC.
    Expect this to be the hardest of the three.
    """
    notes: list[str] = []
    df = _read("liver", header=None, names=LIVER_COLUMNS)
    df = _drop_duplicates(df, notes)

    df["sex"] = df["sex"].str.strip().str.lower().map({"male": 1, "female": 0})
    unknown_sex = int(df["sex"].isna().sum())
    if unknown_sex:
        raise ValueError(f"{unknown_sex} liver row(s) have a sex other than Male/Female")
    notes.append("sex: Male/Female mapped to 1/0")

    y = (df["selector"] == 1).astype(int)
    notes.append(f"label: selector 1 (patient) -> 1, 2 -> 0, prevalence {y.mean():.2f}")
    missing = int(df["ag_ratio"].isna().sum())
    if missing:
        notes.append(f"ag_ratio: {missing} missing value(s) left for the pipeline to impute")

    _check_columns(LIVER_FEATURES, df)
    names = [spec.name for spec in LIVER_FEATURES]
    return ClinicalDataset(
        X=df[names].astype(float),
        y=y,
        name="liver",
        target="referred to the clinic as a liver patient",
        features=LIVER_FEATURES,
        notes=notes,
    )


LOADERS = {
    "heart": load_heart,
    "diabetes": load_diabetes,
    "liver": load_liver,
}
