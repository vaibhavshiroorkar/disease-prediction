# API reference

Base URL is the service root, `http://localhost:8000` when run with uvicorn.
Interactive docs are at `/docs`.

Every prediction response carries a `disclaimer` field. It is part of the
contract: a client cannot render a result without the caveat travelling with it.

## GET /health

Returns 200 whenever the process is alive. `status` is `degraded` only when the
model artifact could not be loaded. A database that is down does not make the
service unhealthy, it just stops recording.

```json
{
  "status": "ok",
  "version": "1.0.0",
  "models": {
    "symptom": {
      "loaded": true,
      "trained_at": "2024-02-15T09:12:44+00:00",
      "sklearn_version": "1.9.0",
      "features": 131,
      "classes": 41
    }
  },
  "database": {
    "logging_enabled": true,
    "total": 128,
    "reviewed": 40,
    "correct": 36,
    "wrong": 4,
    "observed_accuracy": 0.9
  }
}
```

## GET /symptoms

The 131 keys the model accepts, published straight from the artifact so a
client's picker cannot drift from what the model was trained on.

```json
{"count": 131, "symptoms": [{"key": "abdominal_pain", "label": "Abdominal pain"}]}
```

## GET /diseases

```json
{"count": 41, "diseases": ["AIDS", "Acne", "Alcoholic hepatitis"]}
```

## GET /diseases/{name}/symptoms

Case-insensitive. Returns 404 for an unknown name.

```json
{"disease": "Fungal infection", "count": 5, "symptoms": [{"key": "itching", "label": "Itching"}]}
```

## POST /predict/symptoms

```json
{"symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"], "top_k": 5}
```

`symptoms` takes 1 to 50 entries, `top_k` is 1 to 10 and defaults to 5.

Names are normalized before lookup, so `Skin Rash`, `skin rash` and `skin_rash`
are the same key. Duplicates are collapsed. The recognized names become a binary
vector over the model's 131 columns.

```json
{
  "predictions": [
    {
      "disease": "Fungal infection",
      "probability": 0.8733,
      "confidence": "high",
      "matched_symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"],
      "unmatched_symptoms": []
    }
  ],
  "recognized_symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"],
  "unknown_symptoms": [],
  "suggestions": {},
  "prediction_id": 42,
  "model_metrics": {"cross_validation": {"accuracy": 1.0, "macro_recall": 1.0}},
  "disclaimer": "Educational project. ..."
}
```

`matched_symptoms` are the reported symptoms recorded for that disease in the
training data, and `unmatched_symptoms` the ones never recorded for it. Together
they say why a condition ranks where it does, which is steadier and far cheaper
than running feature attribution per request.

`confidence` buckets the probability: `low` below 0.25, `moderate` below 0.60,
`high` at or above it.

`prediction_id` is the row id in Postgres, needed to review the prediction
later. It is `null` when logging is off, which is the normal state with no
`DATABASE_URL` set.

### Unknown symptoms

One bad name does not fail the request. A user misspelling one symptom out of
six still gets a ranking, with the misses reported back and close matches
offered:

```json
{
  "unknown_symptoms": ["hedache"],
  "suggestions": {"hedache": ["headache"]}
}
```

If *every* name is unrecognized there is nothing to predict from, and the
request is a 422:

```json
{
  "detail": {
    "message": "none of the supplied symptoms are recognized",
    "unknown_symptoms": ["headake"],
    "suggestions": {"headake": ["headache"]},
    "hint": "GET /symptoms lists every valid key"
  }
}
```

## Reviewing logged predictions

These routes exist to close the loop: find what the model got wrong, record the
right answer, retrain on it. With no database attached they return empty lists
rather than errors, except the review write itself which returns 503.

### GET /predictions

Newest first. `limit` is 1 to 200, default 20.

```json
[
  {
    "id": 42,
    "created_at": "2024-02-15T09:20:11+00:00",
    "symptoms": ["itching", "skin_rash"],
    "predicted_disease": "Fungal infection",
    "confidence": 0.8733,
    "actual_disease": null,
    "was_correct": null
  }
]
```

### GET /predictions/wrong

Same shape, filtered to rows a reviewer marked wrong. This is the retraining
queue. `limit` is 1 to 200, default 50.

### GET /predictions/stats

```json
{
  "logging_enabled": true,
  "total": 128,
  "reviewed": 40,
  "correct": 36,
  "wrong": 4,
  "observed_accuracy": 0.9
}
```

`observed_accuracy` is `correct / reviewed`, so it is `null` until something has
been reviewed. It measures the model against reality, unlike the training scores
in `model_metrics`.

### POST /predictions/{id}/review

```json
{"was_correct": false, "actual_disease": "Acne"}
```

`actual_disease` is optional but is what makes a row usable for retraining. A
row marked wrong without one is counted and kept, but skipped by
`ml.train --with-feedback`, because nothing records what the answer should have
been. Blank or whitespace-only values are stored as null rather than becoming a
training label.

```json
{
  "prediction_id": 42,
  "recorded": true,
  "message": "recorded as wrong, correct answer 'Acne' saved for retraining."
}
```

Returns 404 for an unknown id, and 503 when logging is not active.

## Errors

| Status | Meaning |
|---|---|
| 404 | Unknown disease name, or unknown prediction id |
| 422 | Request failed validation, or no symptom was recognized |
| 503 | The model artifact is missing, corrupt, or trained on another scikit-learn; or a review was attempted with logging off |

Validation happens at the edge through Pydantic, so a malformed request never
reaches the model and the 422 names the offending field.
