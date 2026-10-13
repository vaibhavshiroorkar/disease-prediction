"""Risk prediction routes for the binary models, and their review loop.

The tabular routes take named measurements, fill any the client left out with
the training median, and return a calibrated probability. With it come the
threshold the model was tuned to and which inputs moved the answer.

The malaria route takes a picture of one blood cell instead.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from app import db
from app.config import DISCLAIMER
from app.registry import RISK_MODELS, LoadedModel, ModelNotAvailable, registry
from app.schemas import (
    DengueRequest,
    DiabetesRequest,
    HeartRequest,
    ImageInfo,
    LiverRequest,
    LoggedRiskPrediction,
    MalariaResponse,
    ReviewResponse,
    RiskFactor,
    RiskResponse,
    RiskReviewRequest,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["risk"])
review_router = APIRouter(prefix="/risk-predictions", tags=["review"])

MAX_FACTORS = 3


def _model(name: str) -> LoadedModel:
    try:
        return registry.get(name)
    except ModelNotAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def risk_band(probability: float, threshold: float) -> str:
    """Bands are relative to the threshold, because the thresholds differ a lot.

    The diabetes model flags at about 0.21 and the liver model at about 0.54.
    A fixed "0.3 is moderate" would mean different things for each.
    """
    if probability >= threshold:
        return "elevated"
    if probability >= threshold / 2:
        return "borderline"
    return "low"


def explain(model: LoadedModel, row: dict[str, float], supplied: list[str]) -> list[RiskFactor]:
    """Which supplied inputs moved the risk most, by swapping each for a typical value.

    For every input the client gave, rescore the patient with that one value
    replaced by the training median (or most common category). The gap between
    the two risks is that input's effect. Imputed inputs are skipped: they are
    the typical value already.

    One batched predict_proba call over at most ten rows, so it costs about as
    much as the prediction itself.
    """
    labels = {spec["name"]: spec["label"] for spec in model.feature_specs}
    swapped: list[str] = []
    rows = [row]
    for name in supplied:
        typical = model.reference_values.get(name)
        if typical is None or row[name] == typical:
            continue
        rows.append({**row, name: typical})
        swapped.append(name)
    if not swapped:
        return []

    frame = pd.DataFrame(rows, columns=model.feature_names, dtype=float)
    proba = model.estimator.predict_proba(frame)[:, 1]
    factors = [
        RiskFactor(
            feature=name,
            label=labels.get(name, name),
            value=row[name],
            typical_value=model.reference_values[name],
            effect=round(float(proba[0] - proba[i + 1]), 4),
        )
        for i, name in enumerate(swapped)
    ]
    factors.sort(key=lambda f: abs(f.effect), reverse=True)
    return [f for f in factors if f.effect != 0][:MAX_FACTORS]


def _headline(model: LoadedModel) -> tuple[dict[str, Any], dict[str, Any]]:
    """The scores worth sending with every response: how good the model is, and
    what its threshold does."""
    test = model.metrics.get("test", {})
    ci = model.metrics.get("test_ci95", {})
    metrics = {
        "chosen_family": model.metrics.get("chosen_family"),
        "test_rows": test.get("rows"),
        "roc_auc": test.get("roc_auc"),
        "roc_auc_ci95": ci.get("roc_auc"),
        "calibration_ece": model.metrics.get("calibration", {}).get("ece"),
    }
    operating = {
        key: test.get(key) for key in ["sensitivity", "specificity", "ppv", "npv"]
    } | {f"{key}_ci95": ci.get(key) for key in ["sensitivity", "specificity"]}
    return metrics, operating


def predict_risk(name: str, request: BaseModel) -> RiskResponse:
    started = time.perf_counter()
    model = _model(name)
    if model.threshold is None:
        raise HTTPException(status_code=503, detail=f"{name}.joblib has no threshold; retrain it")

    values = request.model_dump()
    supplied = [f for f in model.feature_names if values.get(f) is not None]
    imputed = [f for f in model.feature_names if values.get(f) is None]

    # NaN for the gaps: the pipeline's own imputer fills them with the training
    # median, exactly as it did during cross-validation.
    row = {f: float(values[f]) if values.get(f) is not None else math.nan for f in model.feature_names}
    frame = pd.DataFrame([row], columns=model.feature_names, dtype=float)
    probability = float(model.estimator.predict_proba(frame)[0, 1])
    if not np.isfinite(probability):
        raise HTTPException(status_code=500, detail="model returned a non-finite probability")

    flagged = probability >= model.threshold
    factors = explain(model, row, supplied)
    metrics, operating = _headline(model)
    elapsed_ms = (time.perf_counter() - started) * 1000

    prediction_id = db.log_risk_prediction(
        model=name,
        inputs={f: values[f] for f in supplied},
        imputed=imputed,
        probability=round(probability, 4),
        threshold=round(model.threshold, 4),
        flagged=flagged,
        model_trained_at=model.trained_at,
        response_ms=round(elapsed_ms, 2),
    )
    log.info(
        "%s risk %.3f (threshold %.3f, %d imputed) in %.1fms id=%s",
        name, probability, model.threshold, len(imputed), elapsed_ms, prediction_id,
    )

    return RiskResponse(
        model=name,
        condition=model.target or name,
        probability=round(probability, 4),
        risk_band=risk_band(probability, model.threshold),
        flagged=flagged,
        threshold=round(model.threshold, 4),
        operating_point=operating,
        factors=factors,
        imputed_fields=imputed,
        prediction_id=prediction_id,
        model_metrics=metrics,
        disclaimer=DISCLAIMER,
    )


@router.post("/heart", response_model=RiskResponse, summary="Risk of coronary artery disease")
def predict_heart(request: HeartRequest) -> RiskResponse:
    """Trained on 918 patients from four hospitals (Cleveland, Hungary, Switzerland, Long Beach VA)."""
    return predict_risk("heart", request)


@router.post("/diabetes", response_model=RiskResponse, summary="Risk of diabetes within five years")
def predict_diabetes(request: DiabetesRequest) -> RiskResponse:
    """Trained on 768 Pima women aged 21 and over. It has not seen men or other populations."""
    return predict_risk("diabetes", request)


@router.post("/liver", response_model=RiskResponse, summary="Likelihood of being a liver patient")
def predict_liver(request: LiverRequest) -> RiskResponse:
    """Trained on 570 patients from Andhra Pradesh. The weakest of the three models; see /models/liver."""
    return predict_risk("liver", request)


@router.post(
    "/dengue",
    response_model=RiskResponse,
    summary="Risk that a confirmed dengue case turns serious",
)
def predict_dengue(request: DengueRequest) -> RiskResponse:
    """For a patient whose dengue is already confirmed: the chance it becomes dengue with
    warning signs or severe dengue. Trained on 2023 lab-confirmed cases from Brazil's national
    register and checked on 2025. It does not say whether someone has dengue."""
    return predict_risk("dengue", request)


IMAGE_TYPES = {"image/png", "image/jpeg"}


@router.post(
    "/malaria",
    response_model=MalariaResponse,
    summary="Is this red blood cell infected with malaria?",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {kind: {"schema": {"type": "string", "format": "binary"}} for kind in IMAGE_TYPES},
        }
    },
)
async def predict_malaria(request: Request) -> MalariaResponse:
    """Send the image bytes as the body, with Content-Type image/png or image/jpeg:

        curl --data-binary @cell.png -H "Content-Type: image/png" localhost:8000/predict/malaria

    The image must be one red blood cell cropped from a Giemsa-stained thin smear, like the
    NIH training images. A whole field of view, a thick smear or a photo of a test strip will
    get a number back, but the number means nothing. One cell is also not a diagnosis: a
    microscopist reads hundreds of cells per slide.
    """
    # Imported here so the tabular routes never pay for Pillow.
    from app.cells import FEATURE_VERSION, cell_features, read_image

    started = time.perf_counter()
    model = _model("malaria")
    if model.threshold is None or model.feature_version != FEATURE_VERSION:
        raise HTTPException(
            status_code=503,
            detail=f"malaria.joblib was built with feature version {model.feature_version}, "
            f"the code is at {FEATURE_VERSION}; retrain it",
        )

    kind = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if kind not in IMAGE_TYPES:
        raise HTTPException(status_code=415, detail=f"send image/png or image/jpeg, not {kind or 'nothing'}")
    data = await request.body()
    if not data:
        raise HTTPException(status_code=422, detail="the request body is empty; send the image bytes")
    try:
        image = read_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    features = cell_features(image).reshape(1, -1)
    probability = float(model.estimator.predict_proba(features)[0, 1])
    flagged = probability >= model.threshold
    info = ImageInfo(
        format=image.format, width=image.width, height=image.height, sha256=hashlib.sha256(data).hexdigest()
    )
    metrics, operating = _headline(model)
    metrics["test_patients"] = model.metrics.get("test", {}).get("patients")
    elapsed_ms = (time.perf_counter() - started) * 1000

    prediction_id = db.log_risk_prediction(
        model="malaria",
        inputs=info.model_dump(),
        imputed=[],
        probability=round(probability, 4),
        threshold=round(model.threshold, 4),
        flagged=flagged,
        model_trained_at=model.trained_at,
        response_ms=round(elapsed_ms, 2),
    )
    log.info("malaria cell p=%.3f (threshold %.3f) in %.1fms id=%s",
             probability, model.threshold, elapsed_ms, prediction_id)

    return MalariaResponse(
        model="malaria",
        condition=model.target or "malaria",
        probability=round(probability, 4),
        flagged=flagged,
        threshold=round(model.threshold, 4),
        operating_point=operating,
        image=info,
        prediction_id=prediction_id,
        model_metrics=metrics,
        disclaimer=DISCLAIMER,
    )


@review_router.get("", response_model=list[LoggedRiskPrediction], summary="Recently logged risk predictions")
def list_risk_predictions(
    limit: int = Query(20, ge=1, le=200),
    model: str | None = Query(None, description="Only this model."),
) -> list[LoggedRiskPrediction]:
    if model is not None and model not in RISK_MODELS:
        raise HTTPException(status_code=404, detail=f"unknown risk model: {model}")
    return [LoggedRiskPrediction(**row) for row in db.recent_risk_predictions(limit, model)]


@review_router.get("/stats", summary="Per-model counts, live calibration and threshold performance")
def risk_prediction_stats() -> dict[str, object]:
    return db.risk_stats()


@review_router.post(
    "/{prediction_id}/review", response_model=ReviewResponse, summary="Record the real outcome"
)
def review_risk_prediction(prediction_id: int, request: RiskReviewRequest) -> ReviewResponse:
    if not db.enabled():
        raise HTTPException(
            status_code=503,
            detail="prediction logging is not active; set DATABASE_URL to review predictions",
        )
    if not db.record_risk_review(prediction_id, request.had_condition):
        raise HTTPException(status_code=404, detail=f"no logged risk prediction with id {prediction_id}")
    outcome = "had" if request.had_condition else "did not have"
    return ReviewResponse(
        prediction_id=prediction_id, recorded=True, message=f"recorded: the patient {outcome} the condition."
    )
