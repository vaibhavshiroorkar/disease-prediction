"""Catalogue routes.

A client cannot build a symptom picker without knowing the exact keys the model
was trained on, so these endpoints publish them straight from the artifact
rather than from a hand-maintained list that could drift.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.registry import MODEL_NAMES, LoadedModel, ModelNotAvailable, registry
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


def _summary(model: LoadedModel) -> dict[str, Any]:
    """The headline numbers for one model, in a shape that suits both kinds."""
    if model.kind == "binary":
        test = model.metrics.get("test", {})
        return {
            "condition": model.target,
            "chosen_family": model.metrics.get("chosen_family"),
            "test_roc_auc": test.get("roc_auc"),
            "test_roc_auc_ci95": model.metrics.get("test_ci95", {}).get("roc_auc"),
            "threshold": round(model.threshold, 4) if model.threshold is not None else None,
            "sensitivity_at_threshold": test.get("sensitivity"),
            "specificity_at_threshold": test.get("specificity"),
        }
    cv = model.metrics.get("cross_validation", {})
    return {
        "classes": len(model.classes),
        "cv_accuracy": cv.get("accuracy"),
        "cv_top_3_accuracy": cv.get("top_3_accuracy"),
        "noise_2_dropped_2_added": model.metrics.get("noise_2_dropped_2_added", {}).get("accuracy"),
    }


@router.get("/models", summary="Every model the service can run, with headline scores")
def list_models() -> dict[str, object]:
    models = []
    for name in MODEL_NAMES:
        try:
            model = registry.get(name)
        except ModelNotAvailable as exc:
            models.append({"name": name, "loaded": False, "error": str(exc)})
            continue
        models.append(
            {
                "name": name,
                "loaded": True,
                "kind": model.kind,
                "endpoint": f"/predict/{'symptoms' if name == 'symptom' else name}",
                "trained_at": model.trained_at,
                "features": len(model.feature_names),
                **_summary(model),
            }
        )
    return {"count": len(models), "models": models}


@router.get("/models/{name}", summary="One model's inputs, scores and training data")
def describe_model(name: str) -> dict[str, object]:
    """Everything the artifact knows about itself.

    For a risk model this is the input specs (names, units, ranges), the full
    test scores with their intervals, the model comparison, and where the data
    came from. The symptom model's per-class recall is left out; it is 41 rows
    long and lives in models/symptom_summary.json.
    """
    if name not in MODEL_NAMES:
        raise HTTPException(status_code=404, detail=f"unknown model: {name}")
    try:
        model = registry.get(name)
    except ModelNotAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    metrics = dict(model.metrics)
    if "holdout" in metrics:
        metrics["holdout"] = {k: v for k, v in metrics["holdout"].items() if k != "per_class_recall"}
    return {
        "name": name,
        "kind": model.kind,
        "trained_at": model.trained_at,
        "sklearn_version": model.sklearn_version,
        "summary": _summary(model),
        "inputs": model.feature_specs or [{"name": f} for f in model.feature_names],
        "metrics": metrics,
        "dataset": model.dataset,
    }
