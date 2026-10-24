"""Request and response models.

Pydantic does the input validation at the edge, so a malformed symptom list never
reaches the model and every rejection comes back as a 422 naming the field.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SymptomRequest(BaseModel):
    symptoms: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Symptom keys as returned by GET /symptoms.",
        examples=[["itching", "skin_rash", "nodal_skin_eruptions"]],
    )
    top_k: int = Field(5, ge=1, le=10, description="How many ranked conditions to return.")

    @field_validator("symptoms")
    @classmethod
    def normalize(cls, values: list[str]) -> list[str]:
        """Accept 'Skin Rash', 'skin rash' and 'skin_rash' as the same key."""
        seen: list[str] = []
        for value in values:
            key = value.strip().lower().replace(" ", "_")
            while "__" in key:
                key = key.replace("__", "_")
            key = key.strip("_")
            if key and key not in seen:
                seen.append(key)
        if not seen:
            raise ValueError("no usable symptom names after normalization")
        return seen


class RankedCondition(BaseModel):
    disease: str
    probability: float = Field(
        ..., description="Symptom model score, 0 to 1; not a calibrated clinical risk.",
    )
    confidence: str = Field(..., description="Bucketed probability: low, moderate, high.")
    matched_symptoms: list[str] = Field(
        default_factory=list,
        description="Reported symptoms recorded for this disease in the training data.",
    )
    unmatched_symptoms: list[str] = Field(
        default_factory=list,
        description="Reported symptoms never recorded for this disease.",
    )


class SymptomResponse(BaseModel):
    predictions: list[RankedCondition]
    recognized_symptoms: list[str]
    unknown_symptoms: list[str] = Field(
        default_factory=list, description="Names that matched no known symptom key."
    )
    suggestions: dict[str, list[str]] = Field(
        default_factory=dict, description="Close matches for each unknown name."
    )
    prediction_id: int | None = Field(
        None,
        description="Row id in Postgres, for submitting a review later. Null when logging is off.",
    )
    model_metrics: dict[str, Any] = Field(default_factory=dict)
    disclaimer: str


class ReviewRequest(BaseModel):
    """A reviewer's verdict on a logged prediction."""

    was_correct: bool = Field(..., description="Whether the top-ranked condition was right.")
    actual_disease: str | None = Field(
        None,
        description="What the condition actually was. Required to retrain on this row.",
    )

    @field_validator("actual_disease")
    @classmethod
    def strip_blank(cls, value: str | None) -> str | None:
        cleaned = (value or "").strip()
        return cleaned or None


class ReviewResponse(BaseModel):
    prediction_id: int
    recorded: bool
    message: str


class LoggedPrediction(BaseModel):
    id: int
    created_at: str | None
    symptoms: list[str]
    predicted_disease: str | None
    confidence: float | None
    actual_disease: str | None
    was_correct: bool | None


class SymptomCatalogue(BaseModel):
    count: int
    symptoms: list[dict[str, str]]


class DiseaseCatalogue(BaseModel):
    count: int
    diseases: list[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    models: dict[str, Any]
    database: dict[str, Any] = Field(default_factory=dict)


# Risk model requests. The field names, ranges and choices mirror the
# feature_specs saved in each artifact, and tests/test_api.py checks that they
# still agree. Unknown fields are rejected rather than ignored: a client that
# sends "cholesterol" instead of "chol" should hear about it, not get a risk
# computed with cholesterol silently imputed.
#
# Optional fields may be left out. The model fills them with the training
# median, and the response lists which ones it filled.


class HeartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(..., ge=18, le=100, description="Age in years.", examples=[54])
    sex: int = Field(..., ge=0, le=1, description="1 for male, 0 for female.", examples=[1])
    cp: int = Field(
        ...,
        ge=1,
        le=4,
        description="Chest pain: 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic.",
        examples=[4],
    )
    trestbps: float | None = Field(None, ge=60, le=250, description="Resting systolic blood pressure, mmHg.")
    chol: float | None = Field(None, ge=80, le=700, description="Serum cholesterol, mg/dl.")
    fbs: int | None = Field(None, ge=0, le=1, description="1 if fasting blood sugar is above 120 mg/dl.")
    restecg: int | None = Field(
        None, ge=0, le=2, description="Resting ECG: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy."
    )
    thalach: float | None = Field(None, ge=50, le=230, description="Max heart rate in an exercise test, bpm.")
    exang: int | None = Field(None, ge=0, le=1, description="1 if exercise brought on angina.")
    oldpeak: float | None = Field(None, ge=-3, le=10, description="Exercise ST depression, mm.")


class DiabetesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pregnancies: float = Field(..., ge=0, le=20, description="Number of times pregnant.", examples=[2])
    glucose: float | None = Field(
        None, ge=40, le=300, description="Plasma glucose 2 hours into an oral glucose tolerance test, mg/dl."
    )
    blood_pressure: float | None = Field(None, ge=20, le=140, description="Diastolic blood pressure, mmHg.")
    skin_thickness: float | None = Field(None, ge=5, le=100, description="Triceps skin fold thickness, mm.")
    insulin: float | None = Field(None, ge=10, le=900, description="2-hour serum insulin, uU/ml.")
    bmi: float | None = Field(None, ge=10, le=70, description="Body mass index, kg/m2.")
    pedigree: float | None = Field(
        None, ge=0.05, le=2.5, description="Diabetes pedigree function, a family-history score."
    )
    age: float = Field(..., ge=18, le=100, description="Age in years.", examples=[45])


class LiverRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(..., ge=1, le=100, description="Age in years.", examples=[45])
    sex: int = Field(..., ge=0, le=1, description="1 for male, 0 for female.", examples=[1])
    total_bilirubin: float | None = Field(None, ge=0.1, le=80, description="Total bilirubin, mg/dl.")
    direct_bilirubin: float | None = Field(None, ge=0.05, le=30, description="Direct bilirubin, mg/dl.")
    alk_phosphatase: float | None = Field(None, ge=20, le=3000, description="Alkaline phosphatase, IU/L.")
    alt: float | None = Field(None, ge=1, le=3000, description="ALT (SGPT), IU/L.")
    ast: float | None = Field(None, ge=1, le=6000, description="AST (SGOT), IU/L.")
    total_proteins: float | None = Field(None, ge=2, le=12, description="Total serum protein, g/dl.")
    albumin: float | None = Field(None, ge=0.5, le=7, description="Serum albumin, g/dl.")
    ag_ratio: float | None = Field(None, ge=0.1, le=4, description="Albumin to globulin ratio.")


def _sign(description: str) -> Any:
    return Field(None, ge=0, le=1, description=f"1 if present. {description}")


class DengueRequest(BaseModel):
    """A patient already confirmed to have dengue, as seen at their first visit."""

    model_config = ConfigDict(extra="forbid")

    age: float = Field(..., ge=0, le=110, description="Age in years.", examples=[34])
    sex: int = Field(..., ge=0, le=1, description="1 for male, 0 for female.", examples=[0])
    pregnant: int | None = Field(None, ge=0, le=1, description="1 if pregnant. 0 for men.")
    days_since_onset: float | None = Field(
        None, ge=0, le=30, description="Days from the first symptom to this visit."
    )
    fever: int | None = _sign("Fever at presentation.")
    myalgia: int | None = _sign("Muscle pain.")
    headache: int | None = _sign("Headache.")
    rash: int | None = _sign("Skin rash (exanthema).")
    vomiting: int | None = _sign("Vomiting.")
    nausea: int | None = _sign("Nausea.")
    back_pain: int | None = _sign("Back pain.")
    conjunctivitis: int | None = _sign("Red, inflamed eyes.")
    arthritis: int | None = _sign("Swollen joints.")
    arthralgia: int | None = _sign("Intense joint pain.")
    petechiae: int | None = _sign("Small red or purple spots from bleeding under the skin.")
    leukopenia: int | None = _sign("Low white cell count on a blood test.")
    tourniquet_test: int | None = _sign("Petechiae appear under an inflated blood pressure cuff.")
    retro_orbital_pain: int | None = _sign("Pain behind the eyes.")
    diabetes: int | None = _sign("Has diabetes.")
    blood_disorder: int | None = _sign("Has a blood disorder, such as sickle cell disease.")
    liver_disease: int | None = _sign("Has chronic liver disease.")
    kidney_disease: int | None = _sign("Has chronic kidney disease.")
    hypertension: int | None = _sign("Has high blood pressure.")
    peptic_ulcer: int | None = _sign("Has a peptic ulcer.")
    autoimmune: int | None = _sign("Has an autoimmune disease.")


RISK_REQUESTS: dict[str, type[BaseModel]] = {
    "heart": HeartRequest,
    "diabetes": DiabetesRequest,
    "liver": LiverRequest,
    "dengue": DengueRequest,
}


class RiskFactor(BaseModel):
    feature: str
    label: str
    value: float
    typical_value: float = Field(..., description="Training median (or most common category).")
    effect: float = Field(
        ...,
        description=(
            "Risk with this value minus risk with the typical value instead. Positive means "
            "this input pushed the risk up."
        ),
    )


class RiskResponse(BaseModel):
    model: str
    condition: str = Field(..., description="What the probability is the probability of.")
    probability: float = Field(..., description="Calibrated risk, 0 to 1.")
    risk_band: str = Field(
        ...,
        description="low (under half the threshold), borderline (under the threshold), or elevated.",
    )
    flagged: bool = Field(..., description="True when probability >= threshold.")
    threshold: float = Field(
        ..., description="Chosen during training to catch 90% of true cases, at the cost of false alarms."
    )
    operating_point: dict[str, Any] = Field(
        default_factory=dict,
        description="Sensitivity and specificity this threshold reached on the held-out test set.",
    )
    factors: list[RiskFactor] = Field(
        default_factory=list, description="The supplied inputs that moved this risk the most."
    )
    imputed_fields: list[str] = Field(
        default_factory=list, description="Inputs left out and filled with the training median."
    )
    prediction_id: int | None = None
    model_metrics: dict[str, Any] = Field(default_factory=dict)
    disclaimer: str


class ImageInfo(BaseModel):
    format: str | None
    width: int
    height: int
    sha256: str = Field(..., description="Hash of the uploaded bytes, so a reviewer can match the image.")


class MalariaResponse(BaseModel):
    model: str
    condition: str
    probability: float = Field(..., description="Calibrated probability that this cell is parasitized.")
    flagged: bool = Field(..., description="True when probability >= threshold.")
    threshold: float = Field(
        ..., description="Chosen during training to catch 90% of infected cells, at the cost of false alarms."
    )
    operating_point: dict[str, Any] = Field(
        default_factory=dict,
        description="Sensitivity and specificity this threshold reached on the held-out patients.",
    )
    image: ImageInfo
    prediction_id: int | None = None
    model_metrics: dict[str, Any] = Field(default_factory=dict)
    disclaimer: str


class RiskReviewRequest(BaseModel):
    had_condition: bool = Field(..., description="Whether the patient turned out to have the condition.")


class LoggedRiskPrediction(BaseModel):
    id: int
    created_at: str | None
    model: str
    inputs: dict[str, Any]
    probability: float
    flagged: bool
    had_condition: bool | None
