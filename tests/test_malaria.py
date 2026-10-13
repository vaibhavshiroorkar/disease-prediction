"""Tests for the blood cell feature extractor and the malaria training helpers."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.cells import FEATURE_NAMES, MAX_IMAGE_BYTES, cell_features, read_image
from ml.malaria import patient_of, per_patient


def cell(parasite: bool, size: int = 120) -> Image.Image:
    """A pale pink disc on black, like the NIH crops, with an optional stained dot."""
    image = Image.new("RGB", (size, size), "black")
    draw = ImageDraw.Draw(image)
    draw.ellipse([8, 8, size - 8, size - 8], fill=(214, 160, 170))
    if parasite:
        # Scaled with the cell, so the dot covers the same share at any size.
        draw.ellipse([0.42 * size, 0.38 * size, 0.58 * size, 0.54 * size], fill=(110, 40, 130))
    return image


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_every_feature_has_a_name() -> None:
    assert cell_features(cell(False)).shape == (len(FEATURE_NAMES),)
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_a_stained_dot_shows_up_in_the_darkness_features() -> None:
    clean = dict(zip(FEATURE_NAMES, cell_features(cell(False)), strict=True))
    stained = dict(zip(FEATURE_NAMES, cell_features(cell(True)), strict=True))
    # The resized disc's blurred rim reads as a little dark even on a clean cell,
    # so the test is the gap, not zero against non-zero.
    assert stained["dark_35"] > clean["dark_35"] + 0.02


def test_features_ignore_image_size_and_mode() -> None:
    small = cell_features(cell(True, size=90))
    large = cell_features(cell(True, size=150).convert("RGBA"))
    assert np.allclose(small[:6], large[:6], atol=0.03)


@pytest.mark.parametrize(
    ("path", "patient"),
    [
        ("cell_images/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_162.png", "C100"),
        ("cell_images/Parasitized/C37BP2_thinF_IMG_20150620_131423a_cell_92.png", "C37"),
        ("cell_images/Uninfected/C12NThinF_IMG_20150614_124212_cell_164.png", "C12"),
        ("C42P5thin_original_IMG_20150623_134426_cell_185.png", "C42"),
    ],
)
def test_patient_id_is_the_slide_number(path: str, patient: str) -> None:
    assert patient_of(path) == patient


def test_read_image_refuses_garbage_and_huge_uploads() -> None:
    assert read_image(png_bytes(cell(False))).size == (120, 120)
    with pytest.raises(ValueError, match="not a readable image"):
        read_image(b"definitely not a png")
    with pytest.raises(ValueError, match="limit"):
        read_image(b"0" * (MAX_IMAGE_BYTES + 1))


def test_per_patient_summary() -> None:
    #                  healthy patient A   infected patient B
    y = np.array([0, 0, 0, 0, 1, 1, 1, 0])
    p = np.array([0.9, 0.1, 0.1, 0.1, 0.8, 0.8, 0.2, 0.1])
    groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    summary = per_patient(y, p, groups, threshold=0.5)
    assert summary["healthy_patients_cells_flagged"] == {"patients": 1, "median": 0.25, "worst": 0.25}
    caught = summary["infected_patients_parasitized_cells_caught"]
    assert caught["median"] == pytest.approx(2 / 3, abs=1e-4)
