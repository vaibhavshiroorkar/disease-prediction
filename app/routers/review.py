"""Routes over the logged predictions.

This is the loop the logging exists for: look at what the model returned, mark
the wrong ones, and feed the corrections back into training with
`python -m ml.train --with-feedback`.

Every route degrades to an empty result rather than an error when DATABASE_URL
is unset, so the API behaves the same with or without a database attached.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app import db
from app.registry import ModelNotAvailable, registry
from app.schemas import LoggedPrediction, ReviewRequest, ReviewResponse
from app.security import require_admin

router = APIRouter(prefix="/predictions", tags=["review"], dependencies=[Depends(require_admin)])


@router.get("", response_model=list[LoggedPrediction], summary="Recently logged predictions")
def list_predictions(limit: int = Query(20, ge=1, le=200)) -> list[LoggedPrediction]:
    return [LoggedPrediction(**row) for row in db.recent_predictions(limit)]


@router.get(
    "/wrong",
    response_model=list[LoggedPrediction],
    summary="Predictions a reviewer marked wrong",
)
def list_wrong(limit: int = Query(50, ge=1, le=200)) -> list[LoggedPrediction]:
    """The retraining queue.

    Rows here with an actual_disease are the ones `ml.train --with-feedback`
    picks up; a row marked wrong without one records the error but cannot be
    trained on, because nothing says what the right answer was.
    """
    return [LoggedPrediction(**row) for row in db.wrong_predictions(limit)]


@router.get("/stats", summary="Counts and observed accuracy over reviewed predictions")
def prediction_stats() -> dict[str, object]:
    return db.stats()


@router.post("/{prediction_id}/review", response_model=ReviewResponse, summary="Record a verdict")
def review_prediction(prediction_id: int, request: ReviewRequest) -> ReviewResponse:
    if request.actual_disease is not None:
        try:
            known = registry.get("symptom").classes
        except ModelNotAvailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if request.actual_disease not in known:
            # A typo here would become a new class the next time the model retrains.
            raise HTTPException(
                status_code=422,
                detail=f"'{request.actual_disease}' is not one of the {len(known)} conditions "
                "the model knows",
            )

    if not db.enabled():
        raise HTTPException(
            status_code=503,
            detail="prediction logging is not active; set DATABASE_URL to review predictions",
        )

    recorded = db.record_review(prediction_id, request.actual_disease, request.was_correct)
    if not recorded:
        raise HTTPException(status_code=404, detail=f"no logged prediction with id {prediction_id}")

    if not request.was_correct and request.actual_disease is None:
        message = "recorded as wrong. Add actual_disease to make it usable for retraining."
    elif not request.was_correct:
        message = f"recorded as wrong, correct answer '{request.actual_disease}' saved for retraining."
    else:
        message = "recorded as correct."

    return ReviewResponse(prediction_id=prediction_id, recorded=True, message=message)
