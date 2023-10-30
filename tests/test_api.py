"""End-to-end tests against the running app and the real artifact."""


from tests.conftest import requires_model


def test_root_lists_endpoints(client) -> None:
    body = client.get("/").json()
    assert body["service"] == "disease-prediction-api"
    assert any("/predict/symptoms" in e for e in body["endpoints"])


@requires_model
def test_health_reports_the_model_loaded(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["models"]["symptom"]["loaded"] is True
    assert body["models"]["symptom"]["classes"] == 41
    assert "logging_enabled" in body["database"]


@requires_model
def test_symptom_catalogue_matches_training_features(client) -> None:
    body = client.get("/symptoms").json()
    assert body["count"] == 131
    assert {"key": "itching", "label": "Itching"} in body["symptoms"]


@requires_model
def test_disease_catalogue(client) -> None:
    body = client.get("/diseases").json()
    assert body["count"] == 41
    assert "Fungal infection" in body["diseases"]


@requires_model
def test_disease_symptoms_lookup_is_case_insensitive(client) -> None:
    body = client.get("/diseases/fungal infection/symptoms").json()
    assert body["disease"] == "Fungal infection"
    assert "itching" in [s["key"] for s in body["symptoms"]]


@requires_model
def test_unknown_disease_returns_404(client) -> None:
    assert client.get("/diseases/not-a-disease/symptoms").status_code == 404


@requires_model
def test_symptom_prediction_ranks_the_right_condition_first(client) -> None:
    """The textbook fungal infection pattern should come back on top."""
    response = client.post(
        "/predict/symptoms",
        json={"symptoms": ["itching", "skin_rash", "nodal_skin_eruptions", "dischromic_patches"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predictions"][0]["disease"] == "Fungal infection"
    assert body["predictions"][0]["probability"] > 0.5
    assert body["unknown_symptoms"] == []
    assert "not medical advice" in body["disclaimer"]


@requires_model
def test_predictions_are_ranked_and_respect_top_k(client) -> None:
    body = client.post(
        "/predict/symptoms",
        json={"symptoms": ["vomiting", "fatigue", "high_fever"], "top_k": 3},
    ).json()
    probabilities = [p["probability"] for p in body["predictions"]]
    assert len(probabilities) == 3
    assert probabilities == sorted(probabilities, reverse=True)


@requires_model
def test_matched_symptoms_explain_the_top_prediction(client) -> None:
    body = client.post(
        "/predict/symptoms",
        json={"symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"]},
    ).json()
    top = body["predictions"][0]
    assert set(top["matched_symptoms"]) == {"itching", "skin_rash", "nodal_skin_eruptions"}
    assert top["unmatched_symptoms"] == []


@requires_model
def test_symptom_names_are_normalized(client) -> None:
    body = client.post("/predict/symptoms", json={"symptoms": ["Skin Rash", "ITCHING"]}).json()
    assert set(body["recognized_symptoms"]) == {"skin_rash", "itching"}


@requires_model
def test_duplicate_symptoms_are_collapsed(client) -> None:
    body = client.post(
        "/predict/symptoms", json={"symptoms": ["itching", "itching", "Itching"]}
    ).json()
    assert body["recognized_symptoms"] == ["itching"]


@requires_model
def test_unknown_symptom_is_reported_but_does_not_fail_the_request(client) -> None:
    body = client.post(
        "/predict/symptoms",
        json={"symptoms": ["itching", "skin_rash", "itchingg"]},
    ).json()
    assert body["unknown_symptoms"] == ["itchingg"]
    assert "itching" in body["suggestions"]["itchingg"]
    assert len(body["predictions"]) > 0


@requires_model
def test_all_unknown_symptoms_returns_422_with_suggestions(client) -> None:
    response = client.post("/predict/symptoms", json={"symptoms": ["headake"]})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "headache" in detail["suggestions"]["headake"]


def test_empty_symptom_list_is_rejected(client) -> None:
    assert client.post("/predict/symptoms", json={"symptoms": []}).status_code == 422


def test_too_many_symptoms_is_rejected(client) -> None:
    body = {"symptoms": [f"symptom_{i}" for i in range(51)]}
    assert client.post("/predict/symptoms", json=body).status_code == 422


def test_top_k_out_of_range_is_rejected(client) -> None:
    response = client.post("/predict/symptoms", json={"symptoms": ["itching"], "top_k": 99})
    assert response.status_code == 422


def test_missing_symptoms_field_is_rejected(client) -> None:
    assert client.post("/predict/symptoms", json={"top_k": 3}).status_code == 422


@requires_model
def test_timing_header_is_present(client) -> None:
    response = client.get("/health")
    assert float(response.headers["X-Response-Time-ms"]) >= 0
