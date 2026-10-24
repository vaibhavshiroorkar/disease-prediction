# Deploy the demo

The Docker image serves the frontend and API together. You do not need Node.js,
a separate frontend host, or the raw training datasets. The image includes the
six trained artifacts and a hash-locked Python runtime.

## Build and run

```bash
docker build -t disease-prediction .
docker run --rm -p 127.0.0.1:8000:8000 disease-prediction
```

Open http://localhost:8000. `/docs` provides interactive API documentation;
`/api` returns the service metadata that previously lived at `/`.

For a managed Docker host, expose its assigned port and configure HTTP health
checks on `/ready`. The process reads `PORT` and listens on `0.0.0.0`. It uses one
worker so its in-memory rate limit applies consistently within the container.

For a server behind a reverse proxy:

```bash
docker compose -f compose.production.yml up -d --build
```

This binds port 8000 only to the server's loopback interface. Put an HTTPS reverse
proxy in front of it. The compose service runs without root, drops Linux
capabilities, and uses a read-only filesystem with a temporary `/tmp` mount.

The default image runs with prediction logging off and administrator access
disabled. You can host an educational demo with those defaults.

## Environment variables

Copy [.env.example](../.env.example) to `.env` to configure Docker Compose. For a
managed host, set the same values in its environment settings. Running Python
directly does not automatically load `.env`; export those variables in your shell.

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | HTTP listening port; production Compose fixes the internal port to 8000 |
| `ADMIN_API_KEY` | empty | Secret for all history, statistics and review routes; empty disables access |
| `DATABASE_URL` | empty | Postgres connection string; empty disables prediction logging |
| `RATE_LIMIT_PER_MINUTE` | `60` | Shared request budget per client for prediction and administrator routes |
| `ALLOWED_ORIGINS` | empty | Comma-separated browser origins allowed by CORS; unnecessary for this frontend |
| `FORWARDED_ALLOW_IPS` | `127.0.0.1` | Reverse proxy addresses Uvicorn trusts for forwarded client IPs |
| `PRELOAD_MODELS` | `1` | Load every artifact during startup |
| `MODELS_DIR` | `models/` in the project | Override the artifact directory when running Python directly |

For correct per-client limits behind a proxy, set `FORWARDED_ALLOW_IPS` to that
proxy's actual addresses and prevent direct access to the application port.
Do not trust arbitrary forwarded headers from public clients. If the proxy is
not trusted, its requests share one rate-limit budget instead.

The limiter returns 429 with `Retry-After` when a client exceeds its budget.
It bounds client storage to 4,096 entries. Multiple replicas need a shared edge
limiter or a shared store; each process otherwise has its own budget.

## Administrator access

Generate a server-side secret:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Set that value as `ADMIN_API_KEY`. Never put it in frontend code or a public
repository. Use the `Authorize` control in `/docs`, or send it in `X-API-Key`:

```bash
curl -H "X-API-Key: $ADMIN_API_KEY" http://localhost:8000/predictions
```

Without a configured key, protected routes return 503. With a configured key,
missing or incorrect credentials return 401. No login cookie or browser storage
contains the key. The public frontend only calls public prediction and catalogue
routes; it has no administrator session.

## Optional Postgres logging

Use a separate Postgres database for this deployment and set `DATABASE_URL`.
The API creates its tables on startup. Use a TLS connection when required by
your database host. Set a strong database password rather than the credentials
in the development compose file.

Logging stores submitted measurements or symptoms, model outputs, and later
reviews. For malaria, it stores image metadata and a hash, not the image bytes.
The frontend displays whether logging is currently enabled. Avoid submitting
identifying information. Only authenticated administrators can read the history.

If the database is unavailable, predictions continue without recording. `/health`
reports that logging is off. `/ready` checks the model load state, not database
availability, because prediction serving does not require Postgres.

## Verify a deployment

Check `/ready` for a 200 response and `{"ready": true}`. It returns 503 if any
model failed to load. `/health` remains a liveness endpoint with model status
and database connection flags; it does not expose review statistics.

Try the symptom example and one clinical example in the frontend. Confirm that
`/predictions` rejects unauthenticated requests. Use `/models` to check that all
six artifacts loaded and the symptom model reports `chosen_family: logistic`.

Prediction and administrator responses use `Cache-Control: no-store`. The frontend
uses a same-origin content security policy. The API limits request bodies to
2.1 MB, and malaria accepts PNG or JPEG files up to 2 MB.

## Checks before changes ship

[The CI workflow](../.github/workflows/ci.yml) runs lint and the Python suite on
Python 3.12 and 3.13, with a disposable Postgres service. It also runs Chromium
flows and builds and starts the Docker image on an assigned port.

To run browser checks locally after installing development requirements:

```bash
python -m playwright install chromium
RUN_BROWSER_TESTS=1 python -m pytest tests/browser -ra
```

In PowerShell, set `$env:RUN_BROWSER_TESTS = "1"` before the pytest command.
The browser tests start and stop their own local server with logging disabled.

Regenerate the runtime lock after changing its input file:

```bash
uv pip compile requirements-runtime.in --universal --generate-hashes --output-file requirements-runtime.txt
```

Verify model loading and the full test suite after dependency changes. A
scikit-learn version change also requires retraining the matching artifacts.
