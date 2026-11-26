"""The source model both studies train: the served dengue model's family and settings."""

from __future__ import annotations

import json
from functools import cache
from typing import Any

from ml.clinical_data import FeatureSpec
from ml.clinical_model import fixed
from ml.config import MODELS_DIR


@cache
def dengue_params() -> dict[str, Any]:
    """The settings the served dengue model chose, so the studies test that model."""
    summary = json.loads((MODELS_DIR / "dengue_summary.json").read_text(encoding="utf-8"))
    return summary["metrics"]["chosen_params"]


def dengue_model(features: list[FeatureSpec], missing_indicators: bool = True) -> Any:
    """An unfitted copy. Missing-value columns are on by default, as in the served model."""
    return fixed("boosting", features, missing_indicators, dengue_params())
