"""Tests for MONET image and text embedding extraction."""

import numpy as np
import pytest
import torch
from PIL import Image

from src.monet_encoder import extract_embeddings, extract_text_embeddings


class FakeProcessor:
    def __init__(self):
        self.batch_sizes = []

    def __call__(self, images=None, text=None, return_tensors=None, padding=False):
        if images is not None:
            self.batch_sizes.append(len(images))
            values = [image.getpixel((0, 0))[0] / 255.0 for image in images]
            return {"pixel_values": torch.tensor(values).view(-1, 1, 1, 1)}
        return {
            "input_ids": torch.tensor([[len(phrase)] for phrase in text]),
            "attention_mask": torch.ones((len(text), 1), dtype=torch.long),
        }


class FakeModel:
    def get_image_features(self, pixel_values):
        values = pixel_values.view(-1, 1)
        return values.repeat(1, 768)

    def get_text_features(self, input_ids, attention_mask=None):
        return input_ids.float().repeat(1, 768)


def create_test_images(tmp_path, colors):
    paths = []
    for index, color in enumerate(colors):
        path = tmp_path / f"image_{index}.png"
        Image.new("RGB", (10, 10), color=(color, 0, 0)).save(path)
        paths.append(path)
    return paths


def test_extract_image_embeddings_shape_order_and_dtype(tmp_path):
    colors = [50, 100, 200]
    paths = create_test_images(tmp_path, colors)
    processor = FakeProcessor()
    embeddings = extract_embeddings(paths, processor, FakeModel(), "cpu", batch_size=2)

    assert embeddings.shape == (3, 768)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
    np.testing.assert_allclose(embeddings[:, 0], np.array(colors) / 255.0)


def test_extract_image_embeddings_processes_batches(tmp_path):
    paths = create_test_images(tmp_path, [20, 40, 60, 80, 100])
    processor = FakeProcessor()
    extract_embeddings(paths, processor, FakeModel(), "cpu", batch_size=2)
    assert processor.batch_sizes == [2, 2, 1]


def test_extract_image_embeddings_rejects_empty_paths():
    with pytest.raises(ValueError, match="cannot be empty"):
        extract_embeddings([], FakeProcessor(), FakeModel(), "cpu")


def test_extract_image_embeddings_rejects_nonpositive_batch_size(tmp_path):
    paths = create_test_images(tmp_path, [50])
    with pytest.raises(ValueError, match="must be positive"):
        extract_embeddings(paths, FakeProcessor(), FakeModel(), "cpu", batch_size=0)


def test_extract_text_embeddings_shape_order_and_dtype():
    phrases = ["pigment network", "hair", "ruler"]
    embeddings = extract_text_embeddings(phrases, FakeProcessor(), FakeModel(), "cpu", batch_size=2)

    assert embeddings.shape == (3, 768)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
    np.testing.assert_allclose(embeddings[:, 0], [len(phrase) for phrase in phrases])


def test_extract_text_embeddings_rejects_empty_texts():
    with pytest.raises(ValueError, match="cannot be empty"):
        extract_text_embeddings([], FakeProcessor(), FakeModel(), "cpu")


def test_extract_text_embeddings_rejects_blank_phrase():
    with pytest.raises(ValueError, match="non-empty strings"):
        extract_text_embeddings(["hair", " "], FakeProcessor(), FakeModel(), "cpu")
