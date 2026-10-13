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

# The four hospitals in the UCI heart disease collection. Order matters only for
# reports: Cleveland is the site everyone else's papers use.
HEART_SITES = ("cleveland", "hungarian", "switzerland", "va")

# The dengue model trains on the first year and is checked on the second. Two
# separate epidemics are a harder test than a random split of one.
DENGUE_YEARS = (2023, 2025)


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
        *[
            Source(
                key=f"heart_{site}",
                url=(
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
                    f"processed.{site}.data"
                ),
                filename=f"heart_{site}.data",
                description=f"UCI Heart Disease, {site} site. 14 processed attributes per patient.",
            )
            for site in HEART_SITES
        ],
        Source(
            key="diabetes",
            url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
            filename="pima_diabetes.csv",
            description=(
                "Pima Indians Diabetes, 768 women. Mirror of the file UCI has since withdrawn; "
                "the original came from the National Institute of Diabetes and Digestive and "
                "Kidney Diseases."
            ),
        ),
        Source(
            key="liver",
            url=(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/"
                "Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"
            ),
            filename="ilpd_liver.csv",
            description="UCI Indian Liver Patient Dataset, 583 patients from Andhra Pradesh.",
        ),
        *[
            Source(
                key=f"dengue_{year}",
                url=(
                    "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SINAN/Dengue/csv/"
                    f"DENGBR{year % 100}.csv.zip"
                ),
                filename=f"dengue_sinan_{year}.csv.zip",
                description=(
                    f"Brazil's national dengue notification register (SINAN), {year}. One row per "
                    "reported case, about 40 MB zipped. Published by the Ministry of Health on OpenDataSUS."
                ),
            )
            for year in DENGUE_YEARS
        ],
        Source(
            key="malaria_cells",
            url="https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip",
            filename="malaria_cell_images.zip",
            description=(
                "NIH thin blood smear cell images: 27,558 single red blood cells from 200 patients, "
                "half parasitized by P. falciparum. 353 MB."
            ),
        ),
    ]
}
