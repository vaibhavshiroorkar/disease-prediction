"""Artifact loading behaviour."""

from __future__ import annotations

import joblib
import pytest

from app.registry import MODEL_NAMES, ModelNotAvailable, Registry, rank_probabilities
from tests.conftest import requires_model


def test_missing_artifact_raises_with_the_fix_in_the_message(tmp_path) -> None:
    registry = Registry(models_dir=tmp_path)
    with pytest.raises(ModelNotAvailable, match="python -m ml.train"):
        registry.get("symptom")


def test_wrong_artifact_version_is_rejected(tmp_path) -> None:
    """An artifact from an older layout must fail loudly, not load and mispredict."""
    joblib.dump(
        {
            "artifact_version": 1,
            "name": "symptom",
            "kind": "multiclass",
            "model": object(),
            "feature_names": ["a"],
            "classes": ["x"],
        },
        tmp_path / "symptom.joblib",
    )
    registry = Registry(models_dir=tmp_path)
    with pytest.raises(ModelNotAvailable, match="artifact_version 1"):
        registry.get("symptom")


def test_corrupt_artifact_becomes_model_not_available(tmp_path) -> None:
    (tmp_path / "symptom.joblib").write_bytes(b"this is not a pickle")
    registry = Registry(models_dir=tmp_path)
    with pytest.raises(ModelNotAvailable, match="could not read"):
        registry.get("symptom")


def test_preload_records_failures_instead_of_raising(tmp_path) -> None:
    registry = Registry(models_dir=tmp_path)
    registry.preload()
    status = registry.status()
    assert set(status) == set(MODEL_NAMES)
    assert all(entry["loaded"] is False for entry in status.values())


def test_rank_probabilities_sorts_descending() -> None:
    ranked = rank_probabilities([0.1, 0.7, 0.2], ["a", "b", "c"], top_k=2)
    assert ranked == [("b", 0.7), ("c", 0.2)]


def test_rank_probabilities_respects_top_k() -> None:
    ranked = rank_probabilities([0.1, 0.7, 0.2], ["a", "b", "c"], top_k=1)
    assert ranked == [("b", 0.7)]


@requires_model
def test_loaded_model_frame_uses_training_column_order() -> None:
    from app.registry import registry

    model = registry.get("symptom")
    frame = model.frame([0.0] * len(model.feature_names))
    assert list(frame.columns) == model.feature_names
    assert len(frame) == 1


@requires_model
def test_feature_index_covers_every_feature() -> None:
    from app.registry import registry

    model = registry.get("symptom")
    assert len(model.feature_index) == len(model.feature_names)
    assert model.feature_index[model.feature_names[7]] == 7


@requires_model
def test_importance_ranking_is_sorted_and_bounded() -> None:
    from app.registry import registry

    ranking = registry.get("symptom").importance_ranking(top=5)
    assert len(ranking) == 5
    scores = [score for _, score in ranking]
    assert scores == sorted(scores, reverse=True)


@requires_model
def test_artifact_records_the_scores_it_was_trained_with() -> None:
    from app.registry import registry

    metrics = registry.get("symptom").metrics
    assert "macro_recall" in metrics["cross_validation"]
    assert "per_class_recall" in metrics["holdout"]
    assert metrics["search"]["candidates"] > 1
