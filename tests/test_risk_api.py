"""The clinical risk endpoints, against the real artifacts."""

from __future__ import annotations

import pytest

from app.registry import TABULAR_RISK_MODELS, registry
from app.routers.risk import risk_band
from app.schemas import RISK_REQUESTS
from tests.conftest import requires_db, requires_model
from tests.test_malaria import cell, png_bytes

DENGUE_SIGNS = [
    "fever", "myalgia", "headache", "rash", "vomiting", "nausea", "back_pain", "conjunctivitis",
    "arthritis", "arthralgia", "petechiae", "leukopenia", "tourniquet_test", "retro_orbital_pain",
    "diabetes", "blood_disorder", "liver_disease", "kidney_disease", "hypertension", "peptic_ulcer",
    "autoimmune",
]

EXAMPLES = {
    "heart": {"age": 54, "sex": 1, "cp": 4, "trestbps": 130, "chol": 246, "fbs": 0,
              "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.0},
    "diabetes": {"pregnancies": 2, "glucose": 120, "blood_pressure": 70, "skin_thickness": 29,
                 "insulin": 125, "bmi": 32, "pedigree": 0.37, "age": 33},
    "liver": {"age": 45, "sex": 1, "total_bilirubin": 1.0, "direct_bilirubin": 0.3, "alk_phosphatase": 210,
              "alt": 35, "ast": 40, "total_proteins": 6.6, "albumin": 3.1, "ag_ratio": 0.95},
    "dengue": {"age": 34, "sex": 0, "pregnant": 0, "days_since_onset": 3,
               **{sign: 0 for sign in DENGUE_SIGNS}, "fever": 1, "headache": 1, "myalgia": 1},
}
REQUIRED_ONLY = {
    "heart": {"age": 54, "sex": 1, "cp": 4},
    "diabetes": {"pregnancies": 2, "age": 33},
    "liver": {"age": 45, "sex": 1},
    "dengue": {"age": 34, "sex": 0},
}


@pytest.mark.parametrize(("probability", "band"), [(0.05, "low"), (0.25, "borderline"), (0.4, "elevated")])
def test_risk_band_is_relative_to_the_threshold(probability: float, band: str) -> None:
    assert risk_band(probability, threshold=0.4) == band


@requires_model
@pytest.mark.parametrize("name", TABULAR_RISK_MODELS)
def test_request_schema_matches_the_artifact(name: str) -> None:
    """The schemas are hand-written for readable docs. This keeps them honest."""
    model = registry.get(name)
    fields = RISK_REQUESTS[name].model_fields
    assert list(fields) == model.feature_names
    for spec in model.feature_specs:
        field = fields[spec["name"]]
        assert field.is_required() == spec["required"], spec["name"]
        bounds = {type(m).__name__: m for m in field.metadata}
        low = spec.get("minimum", min(int(k) for k in spec.get("choices", {"0": 0})))
        high = spec.get("maximum", max(int(k) for k in spec.get("choices", {"0": 0})))
        assert bounds["Ge"].ge == low, spec["name"]
        assert bounds["Le"].le == high, spec["name"]


@requires_model
@pytest.mark.parametrize("name", TABULAR_RISK_MODELS)
def test_full_request_returns_a_probability_and_nothing_imputed(client, name: str) -> None:
    response = client.post(f"/predict/{name}", json=EXAMPLES[name])
    assert response.status_code == 200, response.text
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["flagged"] == (body["probability"] >= body["threshold"])
    assert body["imputed_fields"] == []
    assert body["operating_point"]["sensitivity"] is not None
    assert "not medical advice" in body["disclaimer"]


@requires_model
@pytest.mark.parametrize("name", TABULAR_RISK_MODELS)
def test_left_out_measurements_are_imputed_and_reported(client, name: str) -> None:
    body = client.post(f"/predict/{name}", json=REQUIRED_ONLY[name]).json()
    assert set(body["imputed_fields"]) == set(EXAMPLES[name]) - set(REQUIRED_ONLY[name])
    assert {f["feature"] for f in body["factors"]} <= set(REQUIRED_ONLY[name])


@requires_model
@pytest.mark.parametrize("name", TABULAR_RISK_MODELS)
def test_out_of_range_and_unknown_fields_are_rejected(client, name: str) -> None:
    assert client.post(f"/predict/{name}", json={**EXAMPLES[name], "age": 400}).status_code == 422
    assert client.post(f"/predict/{name}", json={**EXAMPLES[name], "cholesterol": 200}).status_code == 422


@requires_model
def test_missing_required_field_is_rejected(client) -> None:
    assert client.post("/predict/heart", json={"age": 54, "sex": 1}).status_code == 422


@requires_model
def test_heart_risk_moves_the_way_a_cardiologist_would_expect(client) -> None:
    """A textbook high-risk and low-risk patient should land on opposite sides of the line."""
    high = client.post(
        "/predict/heart",
        json={"age": 66, "sex": 1, "cp": 4, "thalach": 105, "exang": 1, "oldpeak": 3.0},
    ).json()
    low = client.post(
        "/predict/heart",
        json={"age": 35, "sex": 0, "cp": 3, "thalach": 180, "exang": 0, "oldpeak": 0.0},
    ).json()
    assert high["flagged"] is True
    assert low["flagged"] is False
    assert high["probability"] > low["probability"] + 0.5
    assert all(f["effect"] > 0 for f in high["factors"])


@requires_model
def test_diabetes_risk_rises_with_glucose(client) -> None:
    base = {"pregnancies": 1, "bmi": 30, "age": 40}
    low = client.post("/predict/diabetes", json={**base, "glucose": 85}).json()["probability"]
    high = client.post("/predict/diabetes", json={**base, "glucose": 190}).json()["probability"]
    assert high > low


@requires_model
def test_models_catalogue_lists_every_model(client) -> None:
    body = client.get("/models").json()
    assert body["count"] == 6
    by_name = {m["name"]: m for m in body["models"]}
    assert by_name["heart"]["endpoint"] == "/predict/heart"
    assert by_name["symptom"]["endpoint"] == "/predict/symptoms"
    assert 0.5 < by_name["heart"]["test_roc_auc"] < 1.0


@requires_model
def test_model_description_carries_inputs_and_scores(client) -> None:
    body = client.get("/models/heart").json()
    assert [i["name"] for i in body["inputs"]][:3] == ["age", "sex", "cp"]
    assert body["metrics"]["leave_one_site_out"]
    assert client.get("/models/kidney").status_code == 404


def test_every_tabular_model_has_a_request_schema() -> None:
    assert set(RISK_REQUESTS) == set(TABULAR_RISK_MODELS)


@requires_model
def test_dengue_risk_rises_with_bleeding_signs(client) -> None:
    """Petechiae, a positive tourniquet test and leukopenia are the textbook early signs of trouble."""
    base = EXAMPLES["dengue"]
    calm = client.post("/predict/dengue", json=base).json()
    bleeding = client.post(
        "/predict/dengue", json={**base, "petechiae": 1, "tourniquet_test": 1, "leukopenia": 1}
    ).json()
    assert bleeding["probability"] > calm["probability"]
    assert {f["feature"] for f in bleeding["factors"]} & {"petechiae", "tourniquet_test", "leukopenia"}


@requires_model
def test_malaria_scores_a_cell_image(client) -> None:
    response = client.post(
        "/predict/malaria", content=png_bytes(cell(False)), headers={"Content-Type": "image/png"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["flagged"] == (body["probability"] >= body["threshold"])
    assert body["image"]["width"] == 120 and len(body["image"]["sha256"]) == 64
    assert "not medical advice" in body["disclaimer"]


@requires_model
def test_malaria_stained_dot_raises_the_probability(client) -> None:
    def score(image) -> float:
        response = client.post("/predict/malaria", content=png_bytes(image),
                               headers={"Content-Type": "image/png"})
        return response.json()["probability"]

    assert score(cell(True)) > score(cell(False))


@requires_model
def test_malaria_rejects_what_is_not_an_image(client) -> None:
    def post(content: bytes, kind: str) -> int:
        return client.post("/predict/malaria", content=content, headers={"Content-Type": kind}).status_code

    assert post(png_bytes(cell(False)), "application/json") == 415
    assert post(b"", "image/png") == 422
    assert post(b"not an image at all", "image/png") == 422


@requires_model
@requires_db
def test_risk_prediction_is_logged_and_reviewed(client, clean_table) -> None:
    body = client.post("/predict/diabetes", json=EXAMPLES["diabetes"]).json()
    prediction_id = body["prediction_id"]
    assert isinstance(prediction_id, int)

    review = client.post(f"/risk-predictions/{prediction_id}/review", json={"had_condition": True})
    assert review.status_code == 200

    logged = client.get("/risk-predictions", params={"model": "diabetes"}).json()
    assert logged[0]["id"] == prediction_id
    assert logged[0]["had_condition"] is True

    stats = client.get("/risk-predictions/stats").json()
    assert stats["models"]["diabetes"]["reviewed"] == 1
    assert stats["models"]["diabetes"]["observed_rate"] == 1.0


@requires_db
def test_review_of_an_unknown_risk_prediction_is_404(client, clean_table) -> None:
    assert client.post("/risk-predictions/999999/review", json={"had_condition": False}).status_code == 404
