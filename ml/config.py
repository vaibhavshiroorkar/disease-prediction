"""Paths and the registry of public datasets the project trains on."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT / "models"

RANDOM_SEED = 42

# Bundles are pickled with joblib. Bump this when the dict layout changes so a
# stale artifact fails loudly at load instead of misbehaving at predict time.
ARTIFACT_VERSION = 2


@dataclass(frozen=True)
class Source:
    """A single downloadable CSV."""

    key: str
    url: str
    filename: str
    description: str

    @property
    def path(self) -> Path:
        return RAW_DIR / self.filename


SOURCES: dict[str, Source] = {
    s.key: s
    for s in [
        Source(
            key="symptom_train",
            url="https://raw.githubusercontent.com/anujdutt9/Disease-Prediction-from-Symptoms/master/dataset/training_data.csv",
            filename="symptom_training.csv",
            description="4920 rows, 132 binary symptoms, 41 diseases (Kaggle disease-symptom dataset).",
        ),
        Source(
            key="symptom_test",
            url="https://raw.githubusercontent.com/anujdutt9/Disease-Prediction-from-Symptoms/master/dataset/test_data.csv",
            filename="symptom_testing.csv",
            description="42-row held-out split published alongside the training file.",
        ),
    ]
}
