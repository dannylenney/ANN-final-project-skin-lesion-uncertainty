"""Tests for MONET image embedding extraction."""

import numpy as np
import pytest
import torch
from PIL import Image

from src.monet_encoder import extract_embeddings


class FakeProcessor:
    """Small stand-in for the MONET image processor."""

    def __init__(self):
        self.batch_sizes = []

    def __call__(self, images, return_tensors):
        self.batch_sizes.append(len(images))

        values = [
            image.getpixel((0, 0))[0] / 255.0
            for image in images
        ]

        pixel_values = torch.tensor(values).view(-1, 1, 1, 1)
        return {"pixel_values": pixel_values}


class FakeModel:
    """Small stand-in that returns 768-dimensional embeddings."""

    def get_image_features(self, pixel_values):
        values = pixel_values.view(-1, 1)
        return values.repeat(1, 768)


def create_test_images(tmp_path, colors):
    """Create temporary RGB images with the given red-channel values."""
    paths = []

    for index, color in enumerate(colors):
        path = tmp_path / f"image_{index}.png"
        Image.new("RGB", (10, 10), color=(color, 0, 0)).save(path)
        paths.append(path)

    return paths


def test_extract_embeddings_shape_and_order(tmp_path):
    """Return one 768-dimensional embedding per image in input order."""
    colors = [50, 100, 200]
    paths = create_test_images(tmp_path, colors)

    processor = FakeProcessor()
    model = FakeModel()

    embeddings = extract_embeddings(
        paths,
        processor,
        model,
        device="cpu",
        batch_size=2,
    )

    assert embeddings.shape == (3, 768)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()

    expected = np.array(colors, dtype=np.float32) / 255.0
    np.testing.assert_allclose(embeddings[:, 0], expected)


def test_extract_embeddings_processes_batches(tmp_path):
    """Process images in batches, including the final smaller batch."""
    paths = create_test_images(tmp_path, [20, 40, 60, 80, 100])
    processor = FakeProcessor()

    extract_embeddings(
        paths,
        processor,
        FakeModel(),
        device="cpu",
        batch_size=2,
    )

    assert processor.batch_sizes == [2, 2, 1]


def test_extract_embeddings_rejects_empty_image_list():
    """Raise an error when no image paths are provided."""
    with pytest.raises(ValueError, match="cannot be empty"):
        extract_embeddings(
            [],
            FakeProcessor(),
            FakeModel(),
            device="cpu",
        )


def test_extract_embeddings_rejects_nonpositive_batch_size(tmp_path):
    """Raise an error when the batch size is zero or negative."""
    paths = create_test_images(tmp_path, [50])

    with pytest.raises(ValueError, match="must be positive"):
        extract_embeddings(
            paths,
            FakeProcessor(),
            FakeModel(),
            device="cpu",
            batch_size=0,
        )
