"""Turn a picture of one red blood cell into the numbers the malaria model reads.

This lives in the API package because serving needs it on every request and
the API never imports ml/. Training imports it from here, so both sides run
the exact same code.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

# Bump when cell_features changes. The cache and the artifact both carry it, so
# a stale cache is rebuilt and a stale artifact refuses to score new images.
FEATURE_VERSION = 1
IMAGE_SIZE = 64
# Pixels darker than this (sum of RGB in 0-1 units) are the black padding
# around the cell, not the cell.
BACKGROUND = 0.15
DARK_STEPS = (0.15, 0.25, 0.35, 0.5)
GRID = 4
ORIENTATIONS = 8
MAX_IMAGE_BYTES = 2_000_000
MAX_IMAGE_PIXELS = 4_000_000


def _feature_names() -> list[str]:
    names: list[str] = []
    for stat in ["mean", "std"]:
        names += [f"{stat}_{c}" for c in "rgb"]
    for q in [1, 5, 25, 50]:
        names += [f"p{q}_{c}" for c in "rgb"]
    names += [f"range_{c}" for c in "rgb"]
    names += [f"dark_{int(step * 100):02d}" for step in DARK_STEPS]
    names += [f"hue_{i:02d}" for i in range(16)]
    names += [f"saturation_{i}" for i in range(8)]
    names += ["cell_fraction"]
    names += [f"hog_{i}{j}_{o}" for i in range(GRID) for j in range(GRID) for o in range(ORIENTATIONS)]
    names += ["gradient_mean", "gradient_std", "gradient_p99"]
    return names


FEATURE_NAMES = _feature_names()


def read_image(data: bytes) -> Image.Image:
    """Decode an uploaded image, refusing anything that is not a small picture."""
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"image is {len(data)} bytes; the limit is {MAX_IMAGE_BYTES}")
    try:
        image = Image.open(io.BytesIO(data))
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError(f"image dimensions exceed {MAX_IMAGE_PIXELS} pixels")
        image.load()
    except Exception as exc:  # noqa: BLE001 - Pillow raises many types for bad input
        raise ValueError(f"not a readable image: {exc}") from exc
    return image


def cell_features(image: Image.Image) -> np.ndarray:
    """The 181 numbers the model sees for one cell crop.

    The crop is resized to 64x64. Statistics are taken over cell pixels only,
    so the black padding around the round cell does not dilute them.
    """
    small = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    rgb = np.asarray(small, dtype=np.float32) / 255.0
    mask = rgb.sum(axis=2) > BACKGROUND
    if mask.sum() < 50:
        # Almost all background: score the whole square rather than nothing.
        mask = np.ones(mask.shape, dtype=bool)
    pixels = rgb[mask]
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32)[mask] / 255.0
    median = np.median(pixels, axis=0)
    darkness = median.sum() - pixels.sum(axis=1)

    gray = rgb.mean(axis=2)
    gy, gx = np.gradient(gray)
    magnitude = np.hypot(gx, gy) * mask
    angle = np.arctan2(gy, gx) % np.pi
    orientation = np.minimum((angle / np.pi * ORIENTATIONS).astype(int), ORIENTATIONS - 1)
    step = IMAGE_SIZE // GRID
    hog = []
    for i in range(GRID):
        for j in range(GRID):
            block = (slice(i * step, (i + 1) * step), slice(j * step, (j + 1) * step))
            counts = np.bincount(
                orientation[block].ravel(), weights=magnitude[block].ravel(), minlength=ORIENTATIONS
            )
            hog.append(counts / (np.linalg.norm(counts) + 1e-6))
    inside = magnitude[mask]

    parts = [
        pixels.mean(axis=0),
        pixels.std(axis=0),
        np.percentile(pixels, [1, 5, 25, 50], axis=0).ravel(),
        median - pixels.min(axis=0),
        [(darkness > step).mean() for step in DARK_STEPS],
        np.histogram(hsv[:, 0], bins=16, range=(0, 1))[0] / len(hsv),
        np.histogram(hsv[:, 1], bins=8, range=(0, 1))[0] / len(hsv),
        [mask.mean()],
        np.concatenate(hog),
        [inside.mean(), inside.std(), np.percentile(inside, 99)],
    ]
    return np.concatenate([np.ravel(p) for p in parts]).astype(np.float32)
