"""PostgreSQL logging for every prediction the service returns.

Two rules shape this module:

1. Logging never breaks a prediction. The database is a reviewing tool, not part
   of serving. Every write is wrapped so a dead connection costs a log line, not
   the user's response.
2. The service runs without it. With DATABASE_URL unset the whole layer disables
   itself, so a fresh clone and the test suite work with no database at all.

The point of the log is the retraining loop: a reviewer marks what the condition
actually was, and `python -m ml.train --with-feedback` folds those corrections
back into the training set.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app.config import DATABASE_POOL_MAX, DATABASE_POOL_MIN, DATABASE_URL

log = logging.getLogger(__name__)

_pool: Any = None
_initialized = False


def configured() -> bool:
    """Whether a connection string was supplied at all."""
    return bool(DATABASE_URL)


def enabled() -> bool:
    """Whether logging is actually live, i.e. the pool opened successfully."""
    return _pool is not None


def init_pool(url: str | None = None) -> None:
    """Open the connection pool and apply the schema.

    Called once at startup, and by the test suite with an explicit url. Failures
    are logged and swallowed: the API must come up and serve predictions whether
    or not Postgres is reachable.
    """
    global _pool, _initialized
    target = url or DATABASE_URL
    if _initialized or not target:
        return
    _initialized = True

    try:
        from psycopg_pool import ConnectionPool
    except ImportError as exc:
        log.warning(
            "DATABASE_URL is set but the Postgres driver is missing (%s); prediction logging "
            "is off. Install it with: pip install 'psycopg[binary,pool]'",
            exc,
        )
        return

    try:
        _pool = ConnectionPool(
            target,
            min_size=DATABASE_POOL_MIN,
            max_size=DATABASE_POOL_MAX,
            open=True,
            timeout=10,
        )
        _pool.wait(timeout=10)
        apply_schema()
        log.info("prediction logging enabled (pool %d-%d)", DATABASE_POOL_MIN, DATABASE_POOL_MAX)
    except Exception as exc:  # noqa: BLE001 - never fatal
        log.warning("could not connect to Postgres, prediction logging is off: %s", exc)
        _pool = None


def close_pool() -> None:
    global _pool, _initialized
    if _pool is not None:
        _pool.close()
        _pool = None
    _initialized = False


@contextmanager
def cursor() -> Iterator[Any]:
    """Yield a cursor, or None when logging is unavailable.

    Callers check for None rather than catching, which keeps the "no database"
    path explicit at every call site.
    """
    if _pool is None:
        yield None
        return
    # Acquire outside the yield: a contextmanager that raises before yielding
    # turns into "generator didn't yield" and fails the caller's request.
    try:
        connection_context = _pool.connection(timeout=2)
        connection = connection_context.__enter__()
    except Exception as exc:  # noqa: BLE001 - a logging failure is not a request failure
        log.warning("could not get a database connection: %s", exc)
        yield None
        return
    failure: Exception | None = None
    try:
        with connection.cursor() as cur:
            yield cur
    except Exception as exc:  # noqa: BLE001 - a logging failure is not a request failure
        log.warning("database operation failed: %s", exc)
        failure = exc
    try:
        # Passing the failure lets the pool roll back instead of committing.
        if failure is None:
            connection_context.__exit__(None, None, None)
        else:
            connection_context.__exit__(type(failure), failure, failure.__traceback__)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not return the database connection: %s", exc)


SCHEMA = """
create table if not exists predictions (
    id                bigserial primary key,
    created_at        timestamptz not null default now(),
    symptoms          text[]      not null,
    unknown_symptoms  text[]      not null default '{}',
    top_k             smallint    not null,
    predicted_disease text        not null,
    confidence        real        not null,
    ranked            jsonb       not null,
    model_trained_at  text,
    response_ms       real,
    -- Filled in by a reviewer afterwards. actual_disease is what it really was;
    -- was_correct is stored rather than derived so a reviewer can mark a
    -- prediction wrong without yet knowing the right answer.
    actual_disease    text,
    was_correct       boolean,
    reviewed_at       timestamptz
);

create index if not exists predictions_created_idx on predictions (created_at desc);

-- Partial index: the review queue only ever scans unreviewed rows, and they are
-- a shrinking minority of the table.
create index if not exists predictions_unreviewed_idx
    on predictions (created_at desc) where reviewed_at is null;

create index if not exists predictions_wrong_idx
    on predictions (created_at desc) where was_correct is false;

-- The binary risk models log here. A separate table because the inputs are
-- named measurements rather than a symptom list, and the follow-up question is
-- "did they have it" rather than "what did they have".
create table if not exists risk_predictions (
    id                bigserial primary key,
    created_at        timestamptz not null default now(),
    model             text        not null,
    inputs            jsonb       not null,
    imputed           text[]      not null default '{}',
    probability       real        not null,
    threshold         real        not null,
    flagged           boolean     not null,
    model_trained_at  text,
    response_ms       real,
    had_condition     boolean,
    reviewed_at       timestamptz
);

create index if not exists risk_predictions_model_idx on risk_predictions (model, created_at desc);
"""


def apply_schema() -> None:
    with cursor() as cur:
        if cur is None:
            return
        cur.execute(SCHEMA)


def log_prediction(
    symptoms: list[str],
    unknown: list[str],
    top_k: int,
    ranked: list[dict[str, Any]],
    model_trained_at: str | None = None,
    response_ms: float | None = None,
) -> int | None:
    """Record one prediction. Returns its id, or None if logging is off."""
    if not ranked:
        return None

    with cursor() as cur:
        if cur is None:
            return None
        cur.execute(
            """
            insert into predictions
                (symptoms, unknown_symptoms, top_k, predicted_disease,
                 confidence, ranked, model_trained_at, response_ms)
            values (%s, %s, %s, %s, %s, %s, %s, %s)
            returning id
            """,
            (
                symptoms,
                unknown,
                top_k,
                ranked[0]["disease"],
                ranked[0]["probability"],
                json.dumps(ranked),
                model_trained_at,
                response_ms,
            ),
        )
        row = cur.fetchone()
        return int(row[0]) if row else None
    return None


def record_review(prediction_id: int, actual_disease: str | None, was_correct: bool) -> bool:
    """Attach a reviewer's verdict to a logged prediction."""
    with cursor() as cur:
        if cur is None:
            return False
        cur.execute(
            """
            update predictions
               set actual_disease = %s, was_correct = %s, reviewed_at = now()
             where id = %s
            """,
            (actual_disease, was_correct, prediction_id),
        )
        return cur.rowcount > 0
    return False


def recent_predictions(limit: int = 20) -> list[dict[str, Any]]:
    with cursor() as cur:
        if cur is None:
            return []
        cur.execute(
            """
            select id, created_at, symptoms, predicted_disease, confidence,
                   actual_disease, was_correct
              from predictions
             order by created_at desc
             limit %s
            """,
            (limit,),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]
    return []


def wrong_predictions(limit: int = 50) -> list[dict[str, Any]]:
    """The cases a reviewer marked wrong, which is the retraining queue."""
    with cursor() as cur:
        if cur is None:
            return []
        cur.execute(
            """
            select id, created_at, symptoms, predicted_disease, confidence,
                   actual_disease, was_correct
              from predictions
             where was_correct is false
             order by created_at desc
             limit %s
            """,
            (limit,),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]
    return []


def stats() -> dict[str, Any]:
    """Counts for the review dashboard, and live accuracy where it is known."""
    with cursor() as cur:
        if cur is None:
            return {"logging_enabled": False, "configured": configured()}
        cur.execute(
            """
            select count(*),
                   count(*) filter (where reviewed_at is not null),
                   count(*) filter (where was_correct is true),
                   count(*) filter (where was_correct is false)
              from predictions
            """
        )
        total, reviewed, correct, wrong = cur.fetchone()
        return {
            "logging_enabled": True,
            "total": int(total),
            "reviewed": int(reviewed),
            "correct": int(correct),
            "wrong": int(wrong),
            "observed_accuracy": round(correct / reviewed, 4) if reviewed else None,
        }
    return {"logging_enabled": False, "configured": configured()}


def fetch_training_feedback() -> list[tuple[list[str], str]]:
    """Corrected predictions, as (symptoms, actual_disease) pairs.

    Used by `ml.train --with-feedback`. Only rows with a known correct answer are
    returned: a prediction marked wrong without an actual_disease says the model
    erred but not what it should have said, so it cannot be trained on.
    """
    with cursor() as cur:
        if cur is None:
            return []
        cur.execute(
            """
            select symptoms, actual_disease
              from predictions
             where actual_disease is not null
               and array_length(symptoms, 1) > 0
            """
        )
        return [(list(symptoms), actual) for symptoms, actual in cur.fetchall()]
    return []


def log_risk_prediction(
    model: str,
    inputs: dict[str, Any],
    imputed: list[str],
    probability: float,
    threshold: float,
    flagged: bool,
    model_trained_at: str | None = None,
    response_ms: float | None = None,
) -> int | None:
    """Record one risk prediction. Returns its id, or None if logging is off."""
    with cursor() as cur:
        if cur is None:
            return None
        cur.execute(
            """
            insert into risk_predictions
                (model, inputs, imputed, probability, threshold, flagged, model_trained_at, response_ms)
            values (%s, %s, %s, %s, %s, %s, %s, %s)
            returning id
            """,
            (
                model, json.dumps(inputs), imputed, probability, threshold, flagged,
                model_trained_at, response_ms,
            ),
        )
        row = cur.fetchone()
        return int(row[0]) if row else None
    return None


def record_risk_review(prediction_id: int, had_condition: bool) -> bool:
    """Attach the real outcome to a logged risk prediction."""
    with cursor() as cur:
        if cur is None:
            return False
        cur.execute(
            "update risk_predictions set had_condition = %s, reviewed_at = now() where id = %s",
            (had_condition, prediction_id),
        )
        return cur.rowcount > 0
    return False


def recent_risk_predictions(limit: int = 20, model: str | None = None) -> list[dict[str, Any]]:
    with cursor() as cur:
        if cur is None:
            return []
        cur.execute(
            """
            select id, created_at, model, inputs, probability, flagged, had_condition
              from risk_predictions
             where %(model)s::text is null or model = %(model)s
             order by created_at desc
             limit %(limit)s
            """,
            {"model": model, "limit": limit},
        )
        return [
            {
                "id": r[0],
                "created_at": r[1].isoformat() if r[1] else None,
                "model": r[2],
                "inputs": r[3],
                "probability": float(r[4]),
                "flagged": r[5],
                "had_condition": r[6],
            }
            for r in cur.fetchall()
        ]
    return []


def risk_stats() -> dict[str, Any]:
    """Per model: how many predictions, how many have an outcome, and how the
    threshold is doing on the reviewed ones.

    Comparing mean predicted risk with the observed rate is the cheapest live
    calibration check there is. If they drift apart, the population the API is
    seeing no longer looks like the training data.
    """
    with cursor() as cur:
        if cur is None:
            return {"logging_enabled": False, "configured": configured()}
        cur.execute(
            """
            select model,
                   count(*),
                   count(*) filter (where had_condition is not null),
                   avg(probability) filter (where had_condition is not null),
                   avg(had_condition::int) filter (where had_condition is not null),
                   count(*) filter (where flagged and had_condition),
                   count(*) filter (where had_condition),
                   count(*) filter (where not flagged and had_condition is false),
                   count(*) filter (where had_condition is false)
              from risk_predictions
             group by model
             order by model
            """
        )
        models = {}
        for name, total, reviewed, mean_p, observed, tp, positives, tn, negatives in cur.fetchall():
            models[name] = {
                "total": int(total),
                "reviewed": int(reviewed),
                "mean_predicted": round(float(mean_p), 4) if mean_p is not None else None,
                "observed_rate": round(float(observed), 4) if observed is not None else None,
                "sensitivity": round(tp / positives, 4) if positives else None,
                "specificity": round(tn / negatives, 4) if negatives else None,
            }
        return {"logging_enabled": True, "models": models}
    return {"logging_enabled": False, "configured": configured()}


def _row_to_dict(row: tuple) -> dict[str, Any]:
    return {
        "id": row[0],
        "created_at": row[1].isoformat() if row[1] else None,
        "symptoms": list(row[2]) if row[2] else [],
        "predicted_disease": row[3],
        "confidence": float(row[4]) if row[4] is not None else None,
        "actual_disease": row[5],
        "was_correct": row[6],
    }
