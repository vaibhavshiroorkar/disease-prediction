"""Catalogue routes.

A client cannot build a symptom picker without knowing the exact keys the model
was trained on, so these endpoints publish them straight from the artifact
rather than from a hand-maintained list that could drift.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.registry import ModelNotAvailable, registry
from app.schemas import DiseaseCatalogue, SymptomCatalogue

router = APIRouter(tags=["catalogue"])


def display_label(key: str) -> str:
    """'nodal_skin_eruptions' -> 'Nodal skin eruptions'."""
    words = key.replace("_", " ").strip()
    return words[:1].upper() + words[1:]


@router.get("/symptoms", response_model=SymptomCatalogue, summary="Every symptom key the model accepts")
def list_symptoms() -> SymptomCatalogue:
    try:
        model = registry.get("symptom")
    except ModelNotAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    symptoms = [{"key": key, "label": display_label(key)} for key in sorted(model.feature_names)]
    return SymptomCatalogue(count=len(symptoms), symptoms=symptoms)


@router.get("/diseases", response_model=DiseaseCatalogue, summary="Every condition the model can rank")
def list_diseases() -> DiseaseCatalogue:
    try:
        model = registry.get("symptom")
    except ModelNotAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return DiseaseCatalogue(count=len(model.classes), diseases=sorted(model.classes))


@router.get("/diseases/{name}/symptoms", summary="Symptoms recorded for one condition")
def symptoms_for_disease(name: str) -> dict[str, object]:
    try:
        model = registry.get("symptom")
    except ModelNotAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Case-insensitive lookup so clients need not echo the dataset's exact casing.
    lookup = {key.lower(): key for key in model.disease_symptoms}
    resolved = lookup.get(name.strip().lower())
    if resolved is None:
        raise HTTPException(status_code=404, detail=f"unknown disease: {name}")

    keys = model.disease_symptoms[resolved]
    return {
        "disease": resolved,
        "count": len(keys),
        "symptoms": [{"key": k, "label": display_label(k)} for k in keys],
    }
