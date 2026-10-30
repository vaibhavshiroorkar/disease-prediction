"""Runtime settings, read from the environment.

Deliberately plain os.environ rather than a settings library: the API has a
handful of knobs and adding a dependency to read them is not worth it.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))

# Comma-separated list, or "*" to allow everything.
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "").strip()
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))
if RATE_LIMIT_PER_MINUTE < 1:
    raise ValueError("RATE_LIMIT_PER_MINUTE must be positive")
MAX_REQUEST_BYTES = 2_100_000

# How many proxies in front of the app append to X-Forwarded-For. With 1, the
# rate limiter keys on the last entry, the one the host's own proxy wrote; the
# entries before it come from the client and can be forged. 0 uses the socket
# address, which behind a proxy is the proxy itself.
TRUSTED_PROXY_HOPS = int(os.environ.get("TRUSTED_PROXY_HOPS", "0"))
if TRUSTED_PROXY_HOPS < 0:
    raise ValueError("TRUSTED_PROXY_HOPS cannot be negative")
STATIC_DIR = ROOT / "app" / "static"

# Load the artifact during startup instead of on first request, so a cold start
# pays the ~1s load in the health check rather than inside a user's request.
PRELOAD_MODELS = os.environ.get("PRELOAD_MODELS", "1") not in {"0", "false", "False"}

PORT = int(os.environ.get("PORT", "8000"))

# Postgres connection string, e.g.
#   postgresql://dps:devpass@localhost:5432/disease_predictions
# Unset means prediction logging is disabled and the service runs without a
# database, which is what a fresh clone and the test suite do.
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
DATABASE_POOL_MIN = int(os.environ.get("DATABASE_POOL_MIN", "1"))
DATABASE_POOL_MAX = int(os.environ.get("DATABASE_POOL_MAX", "5"))

API_TITLE = "Disease Prediction API"
API_VERSION = "1.0.0"

# Returned on every prediction. The model is trained on a small public research
# dataset and is not a diagnostic device; saying so in the payload means a client
# cannot render a result without the caveat travelling with it.
DISCLAIMER = (
    "Educational project. These predictions come from a model trained on a small public "
    "dataset and are not medical advice. Consult a qualified clinician for diagnosis."
)
