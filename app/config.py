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
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

# Load the artifact during startup instead of on first request, so a cold start
# pays the ~1s load in the health check rather than inside a user's request.
PRELOAD_MODELS = os.environ.get("PRELOAD_MODELS", "1") not in {"0", "false", "False"}

PORT = int(os.environ.get("PORT", "8000"))

API_TITLE = "Disease Prediction API"
API_VERSION = "1.0.0"

# Returned on every prediction. The model is trained on a small public research
# dataset and is not a diagnostic device; saying so in the payload means a client
# cannot render a result without the caveat travelling with it.
DISCLAIMER = (
    "Educational project. These predictions come from a model trained on a small public "
    "dataset and are not medical advice. Consult a qualified clinician for diagnosis."
)
