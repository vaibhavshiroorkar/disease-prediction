"""Request and response models.

Pydantic does the input validation at the edge, so a malformed symptom list never
reaches the model and every rejection comes back as a 422 naming the field.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


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
    probability: float = Field(..., description="Forest confidence, 0 to 1.")
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
