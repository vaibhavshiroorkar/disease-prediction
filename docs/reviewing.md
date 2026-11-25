# Reviewing predictions

## Enable prediction logging

The API can make predictions without a database. To store them, start the local
Postgres container and set `DATABASE_URL` in the same shell as the API.

In PowerShell:

```powershell
docker compose up -d
$env:DATABASE_URL = "postgresql://dps:devpass@localhost:5432/disease_predictions"
python -m uvicorn app.main:app --reload
```

In Bash:

```bash
docker compose up -d
export DATABASE_URL=postgresql://dps:devpass@localhost:5432/disease_predictions
python -m uvicorn app.main:app --reload
```

The API creates its tables on startup. If it cannot connect to Postgres, it logs
a warning and keeps serving predictions without recording them.

`docker-compose.yml` starts only the development database.
`compose.production.yml` runs the application. History, review and statistics
routes require an administrator key in the `X-API-Key` header. Set `ADMIN_API_KEY`
on the server to enable them; they return 503 when no key is configured.

## Add a review

With database logging and administrator access enabled, you can attach a review to a stored prediction.
For symptoms, record whether the result was correct and, if known, the actual condition:

```bash
curl -X POST http://localhost:8000/predictions/42/review \
  -H "X-API-Key: $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"was_correct": false, "actual_disease": "Acne"}'
```

Running `python -m ml.train --with-feedback` adds corrected symptom predictions
to the training data. These rows do not receive extra weight. A result marked
wrong without an `actual_disease` counts in the review statistics but cannot
become a training example because it has no corrected label.

For the binary models, record whether the positive outcome occurred:

```bash
curl -X POST http://localhost:8000/risk-predictions/7/review \
  -H "X-API-Key: $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"had_condition": true}'
```

`/risk-predictions/stats` compares predicted probabilities with recorded
outcomes for each model. A gap can point to calibration drift, though the
reviewed cases may not represent all requests. The risk models do not retrain
on these outcomes; the records support later evaluation.
