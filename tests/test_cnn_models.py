"""Tests for src/cnn_models.py (small input size, no training)."""

import numpy as np
import tensorflow as tf

from src.cnn_models import SCRATCH_FILTERS, build_scratch_cnn


def test_output_shape_and_rows_sum_to_one():
    model = build_scratch_cnn(input_size=64)
    probs = model(np.random.default_rng(0).random((3, 64, 64, 3), dtype=np.float32)).numpy()
    assert probs.shape == (3, 7)
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)


def test_no_normalization_layers():
    model = build_scratch_cnn(input_size=64)
    assert not any("Normalization" in type(layer).__name__ for layer in model.layers)


def test_conv_blocks_and_head():
    model = build_scratch_cnn(input_size=224)
    convs = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    assert [conv.filters for conv in convs] == SCRATCH_FILTERS == [32, 64, 128, 256, 256]
    assert all(conv.kernel_size == (3, 3) for conv in convs)
    pooled = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D)]
    assert pooled[0].input.shape[1:3] == (7, 7)
    assert [layer.rate for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)] == [0.3]
