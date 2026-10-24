# API reference

Base URL is the service root, `http://localhost:8000` when run with uvicorn.
Interactive docs are at `/docs`.

Every prediction response carries a `disclaimer` field. It is part of the
contract: a client cannot render a result without the caveat travelling with it.

## Access and request limits

Prediction and model catalogue routes are public. All `/predictions` and
`/risk-predictions` routes require an administrator key in `X-API-Key`:

```bash
curl -H "X-API-Key: $ADMIN_API_KEY" http://localhost:8000/predictions
```

Set `ADMIN_API_KEY` on the server to enable administrator access. Without it,
protected routes return 503. Missing or incorrect keys return 401 when access
is configured. `/docs` also accepts the key through its `Authorize` control.

Prediction and administrator requests share a per-client limit of 60 requests
per minute by default. Exceeding it returns 429 with a `Retry-After` header.
Request bodies may not exceed 2,100,000 bytes. Prediction and administrator
responses use `Cache-Control: no-store`.

See [deployment settings](deployment.md#environment-variables) to configure
origins, proxy trust, request limits and optional database logging.

## GET / and /api

`/` serves the web interface. `/api` returns the service name, version, API
documentation URL and endpoint list.

## GET /ready

Returns 200 with `{"ready": true}` when all six model artifacts are loaded.
Returns 503 with `{"ready": false}` if any model is unavailable. Database
logging is optional and does not affect readiness.

## GET /health

Returns 200 whenever the process is alive. `status` is `degraded` only when a
model artifact could not be loaded. A database that is down does not make the
service unhealthy, it just stops recording. `models` has one entry per model
(symptom, heart, diabetes, liver, dengue, malaria); the example shows only the
first.

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
    "configured": true
  }
}
```

Review statistics require administrator access through `/predictions/stats`
or `/risk-predictions/stats`.

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

The symptom estimator is logistic regression. Its scores rank conditions from
the training data and are not calibrated clinical probabilities.

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

## GET /models

Every model the service runs, with its headline scores, read from the artifacts.
The catalogue contains six entries; this example shows only the heart model.

```json
{
  "count": 6,
  "models": [
    {
      "name": "heart",
      "loaded": true,
      "kind": "binary",
      "endpoint": "/predict/heart",
      "features": 10,
      "condition": "angiographic coronary artery disease (more than 50% narrowing in any major vessel)",
      "chosen_family": "logistic",
      "test_roc_auc": 0.8656,
      "test_roc_auc_ci95": [0.808, 0.9185],
      "threshold": 0.4032,
      "sensitivity_at_threshold": 0.8725,
      "specificity_at_threshold": 0.6829
    }
  ]
}
```

## GET /models/{name}

One model in full: its inputs (name, label, unit, range, whether required), every
score from training with its 95% interval, the model comparison, and where the
data came from. For the risk models, this is the place to read a field's valid
range. Returns 404 for an unknown name.

## POST /predict/heart, /predict/diabetes, /predict/liver, /predict/dengue

Each takes named clinical measurements and returns a calibrated risk.

```json
{"age": 61, "sex": 1, "cp": 4, "thalach": 115, "exang": 1, "oldpeak": 2.5}
```

| Model | Required | Optional |
|---|---|---|
| heart | `age`, `sex` (1 male), `cp` (chest pain 1-4) | `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak` |
| diabetes | `pregnancies`, `age` | `glucose`, `blood_pressure`, `skin_thickness`, `insulin`, `bmi`, `pedigree` |
| liver | `age`, `sex` (1 male) | `total_bilirubin`, `direct_bilirubin`, `alk_phosphatase`, `alt`, `ast`, `total_proteins`, `albumin`, `ag_ratio` |
| dengue | `age`, `sex` (1 male) | `pregnant`, `days_since_onset`, then 0/1 for each sign: `fever`, `myalgia`, `headache`, `rash`, `vomiting`, `nausea`, `back_pain`, `conjunctivitis`, `arthritis`, `arthralgia`, `petechiae`, `leukopenia`, `tourniquet_test`, `retro_orbital_pain`, and each chronic condition: `diabetes`, `blood_disorder`, `liver_disease`, `kidney_disease`, `hypertension`, `peptic_ulcer`, `autoimmune` |

The dengue model is for patients whose dengue is already confirmed. Its
probability is the chance the case becomes dengue with warning signs or severe
dengue. It cannot say whether someone has dengue in the first place: the public
data it learned from has no negative cases.

Units and ranges are in `/docs` and `/models/{name}`. A value outside its range
is a 422. So is a field the model does not know. If you send `cholesterol`
instead of `chol`, you get an error rather than a risk computed with cholesterol
quietly filled in.

Optional fields you leave out are filled with the training median, the same way
the model handled missing values during training. The response lists them.

```json
{
  "model": "heart",
  "condition": "angiographic coronary artery disease (more than 50% narrowing in any major vessel)",
  "probability": 0.9589,
  "risk_band": "elevated",
  "flagged": true,
  "threshold": 0.4032,
  "operating_point": {
    "sensitivity": 0.8725, "sensitivity_ci95": [0.8039, 0.9314],
    "specificity": 0.6829, "specificity_ci95": [0.5732, 0.7805],
    "ppv": 0.7739, "npv": 0.8116
  },
  "factors": [
    {"feature": "oldpeak", "label": "ST depression", "value": 2.5, "typical_value": 0.6, "effect": 0.0671},
    {"feature": "exang", "label": "Exercise angina", "value": 1.0, "typical_value": 0.0, "effect": 0.0348},
    {"feature": "thalach", "label": "Max heart rate", "value": 115.0, "typical_value": 140.0, "effect": 0.0176}
  ],
  "imputed_fields": ["trestbps", "chol", "fbs", "restecg"],
  "prediction_id": 1,
  "model_metrics": {"chosen_family": "logistic", "test_rows": 184, "roc_auc": 0.8656,
                    "roc_auc_ci95": [0.808, 0.9185], "calibration_ece": 0.04},
  "disclaimer": "Educational project. ..."
}
```

`threshold` was chosen in training to catch 90% of real cases.
`operating_point` is what that threshold actually did on the held-out test set.
`flagged` is `probability >= threshold`.

`risk_band` is relative to the threshold, because the thresholds differ a lot
between models (about 0.21 for diabetes, 0.54 for liver). It is `low` below
half the threshold, `borderline` below the threshold, and `elevated` at or
above it.

`factors` are the three supplied inputs that moved the risk most. For each
input, the patient is rescored with that one value swapped for the training
median, and `effect` is the difference. Two limits:
- An input already at the typical value has no effect, even if it matters. For
  heart, the most common chest pain type is 4 (asymptomatic), so `cp: 4` never
  shows up as a factor.
- Effects shrink when the risk is close to 0 or 1, because the calibrated
  probability flattens out there.

## POST /predict/malaria

Takes a picture of one red blood cell from a Giemsa-stained thin blood smear
and returns the probability that a malaria parasite is inside it. Send the
image bytes as the request body, not JSON and not a form:

```bash
curl --data-binary @cell.png -H "Content-Type: image/png" http://localhost:8000/predict/malaria
```

PNG and JPEG are accepted, up to 2,000,000 bytes and 4,000,000 pixels.

```json
{
  "model": "malaria",
  "condition": "red blood cell parasitized by P. falciparum",
  "probability": 0.9973,
  "flagged": true,
  "threshold": 0.8715,
  "operating_point": {
    "sensitivity": 0.9255, "sensitivity_ci95": [0.9028, 0.9414],
    "specificity": 0.9844, "specificity_ci95": [0.9786, 0.9898],
    "ppv": 0.9837, "npv": 0.9284
  },
  "image": {"format": "PNG", "width": 142, "height": 148, "sha256": "f53ac02c4149..."},
  "prediction_id": 2,
  "model_metrics": {"chosen_family": "boosting", "test_rows": 5555, "roc_auc": 0.9915,
                    "roc_auc_ci95": [0.9882, 0.9942], "calibration_ece": 0.0118, "test_patients": 40},
  "disclaimer": "Educational project. ..."
}
```

`test_rows` counts cells. The intervals come from resampling the 40 test
patients, not the cells. The model was trained on single-cell crops like the NIH images: one cell,
centred, on a black background. A whole microscope field, a thick smear or a
photo of a rapid test will still get a number back, but that number means
nothing. One cell is also not a diagnosis. A microscopist reads hundreds of
cells per slide.

The logged inputs are the image's format, size and SHA-256 hash, not the image
itself. A review records whether that cell really was parasitized.

| Status | Meaning |
|---|---|
| 413 | Request body exceeds 2,100,000 bytes |
| 415 | Content-Type is not image/png or image/jpeg |
| 422 | Empty or unreadable image, or image exceeds its byte or pixel limit |
| 503 | Artifact missing, or trained with a different version of the feature extractor |

## Reviewing risk predictions

The risk models log to their own table. A review here records what really
happened, which is the only way to see whether the calibration holds outside
the training data.

Every route in this section requires `X-API-Key`.

### GET /risk-predictions

Newest first. `limit` is 1 to 200, default 20. `model` filters to one model.

```json
[
  {
    "id": 1,
    "created_at": "2026-10-12T16:31:02+00:00",
    "model": "heart",
    "inputs": {"age": 61.0, "sex": 1, "cp": 4, "thalach": 115.0, "exang": 1, "oldpeak": 2.5},
    "probability": 0.9589,
    "flagged": true,
    "had_condition": null
  }
]
```

### GET /risk-predictions/stats

Per model. `mean_predicted` and `observed_rate` cover reviewed rows only. If they
drift apart, the people the API sees no longer look like the training data.
`sensitivity` and `specificity` are the threshold's live record.

```json
{
  "logging_enabled": true,
  "models": {
    "heart": {"total": 1, "reviewed": 1, "mean_predicted": 0.9589, "observed_rate": 1.0,
              "sensitivity": 1.0, "specificity": null}
  }
}
```

### POST /risk-predictions/{id}/review

```json
{"had_condition": true}
```

Returns 404 for an unknown id, and 503 when logging is not active. The clinical
models do not retrain on these yet. They are recorded so that calibration can
be checked against real outcomes.

## Reviewing logged predictions

These routes exist to close the loop: find what the model got wrong, record the
right answer, retrain on it. Every route requires `X-API-Key`. After authentication,
history lists are empty when logging is off. Review writes return 503.

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
| 401 | Administrator key is missing or incorrect |
| 404 | Unknown disease, model or prediction id |
| 413 | Request body exceeds 2,100,000 bytes |
| 415 | `/predict/malaria` got something other than image/png or image/jpeg |
| 422 | Invalid inputs, no recognized symptoms, or an unreadable or oversized image |
| 429 | Per-client request limit exceeded; retry after the `Retry-After` interval |
| 503 | Model unavailable, readiness failed, administrator access unconfigured, or a review attempted with logging off |

Validation happens at the edge through Pydantic, so a malformed request never
reaches the model and the 422 names the offending field.
