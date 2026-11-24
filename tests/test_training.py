"""Tests for src/training.py (fake data, one tiny training epoch)."""

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from src.cnn_models import build_scratch_cnn
from src.training import (
    balanced_accuracy_from_file,
    class_weights,
    save_predictions,
    set_seed,
    train_model,
)

# 10 nv, 5 mel, 1 of every other class: 20 labels.
LABELS = ["nv"] * 10 + ["mel"] * 5 + ["akiec", "bcc", "bkl", "df", "vasc"]


def test_balanced_class_weights():
    weights = class_weights(LABELS)
    # n / (7 * n_c): rare classes 20/7, mel 20/35, nv 20/70; keys in CLASSES order.
    expected = [20 / 7, 20 / 7, 20 / 7, 20 / 7, 20 / 35, 20 / 70, 20 / 7]
    assert list(weights) == list(range(7))
    assert np.allclose(list(weights.values()), expected)


def test_sqrt_class_weights():
    assert np.allclose(list(class_weights(LABELS, sqrt=True).values()),
                       np.sqrt(list(class_weights(LABELS).values())))


def test_class_weights_raise_on_missing_class():
    with pytest.raises(ValueError):
        class_weights(["nv", "mel"])


def test_predictions_file_columns_and_balanced_accuracy(tmp_path):
    probs = np.eye(7)[[4, 5, 5]]  # predicts mel, nv, nv
    path = save_predictions("scratch_cnn", 0, "val", ["A", "B", "C"], ["mel", "nv", "mel"], probs,
                            out_dir=tmp_path)
    assert path.name == "scratch_cnn_seed0_val.csv"
    table = pd.read_csv(path)
    assert table.columns.tolist() == ["image_id", "label", "p_akiec", "p_bcc", "p_bkl", "p_df",
                                      "p_mel", "p_nv", "p_vasc"]
    assert table["image_id"].tolist() == ["A", "B", "C"]
    assert balanced_accuracy_from_file(path) == pytest.approx(0.75)  # mel 1/2, nv 1/1


def test_save_predictions_rejects_wrong_shape(tmp_path):
    with pytest.raises(ValueError):
        save_predictions("scratch_cnn", 0, "val", ["A", "B"], ["mel", "nv"], np.ones((2, 6)),
                         out_dir=tmp_path)


def test_train_model_one_epoch():
    set_seed(0)
    images = np.random.default_rng(0).random((8, 32, 32, 3), dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((images, np.arange(8) % 7)).batch(4)
    history = train_model(build_scratch_cnn(input_size=32), dataset, dataset,
                          class_weight={i: 1.0 for i in range(7)}, seed=0, epochs=1)
    assert {"loss", "val_loss"} <= set(history.history)
