"""Integration checks for the inference and Grad-CAM contracts.

These tests need a Keras model file. By default they use the real weights
(downloaded from Hugging Face on first run). To run offline, point
MALARIA_MODEL_PATH at any local .h5 with the same architecture.

    python -m pytest tests          # or: python tests/test_pipeline.py
"""

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui.imaging import (  # noqa: E402
    InvalidImageError,
    decode_upload,
    heatmap_display_size,
    map_heatmap_to_original,
)
from utils.gradcam import LAST_CONV_LAYER, generate_gradcam  # noqa: E402
from utils.model_loader import load_malaria_model  # noqa: E402
from utils.predictor import IMAGE_SIZE, predict_image, prepare_image  # noqa: E402


def _cell(width, height):
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


def test_preprocessing_shape_is_independent_of_input_shape():
    for size in [(130, 130), (120, 190), (260, 140), (1600, 1200)]:
        batch = prepare_image(_cell(*size))
        assert batch.shape == (1, IMAGE_SIZE[1], IMAGE_SIZE[0], 3)
        assert batch.dtype == np.float32


def test_display_mapping_preserves_aspect_ratio():
    for size in [(130, 130), (120, 190), (260, 140), (1600, 1200), (3000, 400)]:
        w, h = heatmap_display_size(size)
        assert abs((w / h) / (size[0] / size[1]) - 1) < 0.01


def test_invalid_bytes_are_rejected():
    try:
        decode_upload("broken.png", b"definitely not an image")
    except InvalidImageError:
        return
    raise AssertionError("corrupt upload was accepted")


def test_model_prediction_and_gradcam_contract():
    model = load_malaria_model()
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 1)
    model.get_layer(LAST_CONV_LAYER)

    for size in [(130, 130), (120, 190), (260, 140)]:
        image = _cell(*size)
        result = predict_image(model, image)
        assert result["label"] in {"Parasitized", "Uninfected"}
        p = result["prob_uninfected"]
        assert 0.0 <= p <= 1.0
        expected_label = "Uninfected" if p >= 0.5 else "Parasitized"
        assert result["label"] == expected_label
        assert np.isclose(result["confidence"], p if p >= 0.5 else 1 - p)

        overlay = generate_gradcam(model, image)
        assert overlay.size == (224, 224)
        mapped = map_heatmap_to_original(overlay, image.size)
        assert abs(mapped.size[0] / mapped.size[1] - size[0] / size[1]) < 0.02


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    if not os.environ.get("MALARIA_MODEL_PATH"):
        print("(used the Hugging Face weights)")
