"""The prediction route."""

from __future__ import annotations

import difflib
import logging
import time

from fastapi import APIRouter, HTTPException

from app import db
from app.config import DISCLAIMER
from app.registry import LoadedModel, ModelNotAvailable, rank_probabilities, registry
from app.schemas import RankedCondition, SymptomRequest, SymptomResponse

log = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["predictions"])


def _model(name: str) -> LoadedModel:
    try:
        return registry.get(name)
    except ModelNotAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def confidence_band(probability: float) -> str:
    if probability < 0.25:
        return "low"
    if probability < 0.60:
        return "moderate"
    return "high"


def _suggest(unknown: list[str], vocabulary: list[str], limit: int = 3) -> dict[str, list[str]]:
    """Closest known keys for each unrecognized name, by edit-distance ratio."""
    out: dict[str, list[str]] = {}
    for name in unknown:
        matches = difflib.get_close_matches(name, vocabulary, n=limit, cutoff=0.6)
        if matches:
            out[name] = matches
    return out


@router.post(
    "/symptoms",
    response_model=SymptomResponse,
    summary="Rank likely conditions from a list of symptoms",
)
def predict_symptoms(request: SymptomRequest) -> SymptomResponse:
    """Turn the reported symptoms into a binary feature vector, score every
    condition, and return the top_k with confidences.

    Unknown symptom names do not fail the request. A user misspelling one symptom
    out of six should still get a ranking, with the misses reported back and
    close matches offered so the client can correct them.
    """
    started = time.perf_counter()
    model = _model("symptom")
    known = set(model.feature_names)

    recognized = [s for s in request.symptoms if s in known]
    unknown = [s for s in request.symptoms if s not in known]

    if not recognized:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "none of the supplied symptoms are recognized",
                "unknown_symptoms": unknown,
                "suggestions": _suggest(unknown, model.feature_names),
                "hint": "GET /symptoms lists every valid key",
            },
        )

    # The feature vector is 1 for each reported symptom, 0 for the other ~127.
    vector = [0.0] * len(model.feature_names)
    for symptom in recognized:
        vector[model.feature_index[symptom]] = 1.0

    proba = model.estimator.predict_proba(model.frame(vector))[0]
    ranked = rank_probabilities(proba, model.classes, request.top_k)

    predictions = []
    for disease, probability in ranked:
        recorded = set(model.disease_symptoms.get(disease, []))
        predictions.append(
            RankedCondition(
                disease=disease,
                probability=round(probability, 4),
                confidence=confidence_band(probability),
                matched_symptoms=[s for s in recognized if s in recorded],
                unmatched_symptoms=[s for s in recognized if s not in recorded],
            )
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    prediction_id = db.log_prediction(
        symptoms=recognized,
        unknown=unknown,
        top_k=request.top_k,
        ranked=[p.model_dump() for p in predictions],
        model_trained_at=model.trained_at,
        response_ms=round(elapsed_ms, 2),
    )

    log.info(
        "symptom prediction: %d recognized -> %s (%.3f) in %.1fms id=%s",
        len(recognized),
        predictions[0].disease,
        predictions[0].probability,
        elapsed_ms,
        prediction_id,
    )

    return SymptomResponse(
        predictions=predictions,
        recognized_symptoms=recognized,
        unknown_symptoms=unknown,
        suggestions=_suggest(unknown, model.feature_names),
        prediction_id=prediction_id,
        model_metrics=model.metrics,
        disclaimer=DISCLAIMER,
    )
