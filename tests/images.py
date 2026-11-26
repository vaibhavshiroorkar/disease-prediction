"""Made-up blood cell images, shared by the model and the API tests."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw


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
