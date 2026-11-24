"""Unit tests for BiomedCLIP embedding extraction (no model download required)."""

import numpy as np
import pytest
import torch
from PIL import Image

from src.biomedclip_encoder import extract_embeddings, extract_text_embeddings


class FakePreprocess:
    def __call__(self, image):
        value = image.getpixel((0, 0))[0] / 255.0
        return torch.tensor([value], dtype=torch.float32)


class FakeTokenizer:
    def __call__(self, texts):
        return torch.tensor([[float(len(text))] for text in texts])


class FakeModel:
    def encode_image(self, pixel_values):
        return pixel_values.repeat(1, 4)

    def encode_text(self, tokens):
        return tokens.repeat(1, 4)


def create_images(tmp_path, values):
    paths = []
    for index, value in enumerate(values):
        path = tmp_path / f"image_{index}.png"
        Image.new("RGB", (8, 8), (value, 0, 0)).save(path)
        paths.append(path)
    return paths


def test_extract_image_embeddings_shape_order_and_dtype(tmp_path):
    values = [32, 128, 224]
    paths = create_images(tmp_path, values)
    result = extract_embeddings(paths, FakePreprocess(), FakeModel(), "cpu", batch_size=2)

    assert result.shape == (3, 4)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[:, 0], np.array(values) / 255.0)


def test_image_extraction_batches_final_partial_batch(tmp_path):
    paths = create_images(tmp_path, [10, 20, 30, 40, 50])
    # The output shape confirms that the final one-item batch is included.
    result = extract_embeddings(paths, FakePreprocess(), FakeModel(), "cpu", batch_size=2)
    assert result.shape == (5, 4)


def test_image_extraction_rejects_empty_paths():
    with pytest.raises(ValueError, match="cannot be empty"):
        extract_embeddings([], FakePreprocess(), FakeModel(), "cpu")


def test_image_extraction_rejects_nonpositive_batch_size(tmp_path):
    paths = create_images(tmp_path, [20])
    with pytest.raises(ValueError, match="must be positive"):
        extract_embeddings(paths, FakePreprocess(), FakeModel(), "cpu", batch_size=0)


def test_extract_text_embeddings_shape_and_order():
    phrases = ["pigment network", "hair", "ruler"]
    result = extract_text_embeddings(phrases, FakeTokenizer(), FakeModel(), "cpu", batch_size=2)

    assert result.shape == (3, 4)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[:, 0], [len(text) for text in phrases])


def test_text_extraction_rejects_empty_texts():
    with pytest.raises(ValueError, match="cannot be empty"):
        extract_text_embeddings([], FakeTokenizer(), FakeModel(), "cpu")


def test_text_extraction_rejects_blank_phrase():
    with pytest.raises(ValueError, match="non-empty strings"):
        extract_text_embeddings(["hair", "  "], FakeTokenizer(), FakeModel(), "cpu")
