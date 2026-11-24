"""Tests for src/images.py on a few fake 600x450 JPGs (no real HAM10000 files needed)."""

import numpy as np
import pytest
import tensorflow as tf

from src.data import LESION_GROUPINGS_CSV
from src.images import TRAIN_IMAGES_DIR, augment, load_image, make_dataset

IDS = ["ISIC_1", "ISIC_2", "ISIC_3", "ISIC_4"]
LABELS = ["mel", "nv", "akiec", "vasc"]


def write_jpg(path, pixels):
    tf.io.write_file(str(path), tf.io.encode_jpeg(pixels, quality=100))


@pytest.fixture
def data_dir(tmp_path):
    """A fake data folder with 4 random-noise 600x450 JPGs and an (empty) groupings CSV."""
    rng = np.random.default_rng(0)
    (tmp_path / TRAIN_IMAGES_DIR).mkdir()
    (tmp_path / LESION_GROUPINGS_CSV).touch()
    for image_id in IDS:
        write_jpg(tmp_path / TRAIN_IMAGES_DIR / f"{image_id}.jpg",
                  rng.integers(0, 256, size=(450, 600, 3), dtype=np.uint8))
    return tmp_path


def one_batch(dataset):
    images, labels = next(iter(dataset))
    return images.numpy(), labels.numpy()


def test_center_crop_removes_side_bands(tmp_path):
    pixels = np.full((450, 600, 3), 128, dtype=np.uint8)
    pixels[:, :75] = [255, 0, 0]   # the 75 px on each side should be cropped away
    pixels[:, -75:] = [0, 0, 255]
    write_jpg(tmp_path / "bands.jpg", pixels)
    image = load_image(str(tmp_path / "bands.jpg"), size=32).numpy()
    assert image.shape == (32, 32, 3) and image.dtype == np.uint8
    assert np.abs(image.astype(int) - 128).max() < 20


def test_shape_range_and_order_without_training(data_dir):
    dataset = make_dataset(IDS, LABELS, training=False, size=32, batch_size=4, data_dir=data_dir)
    images, labels = one_batch(dataset)
    assert images.shape == (4, 32, 32, 3) and images.dtype == np.float32
    assert images.min() >= 0 and images.max() <= 1
    assert labels.tolist() == [4, 5, 0, 6]  # mel, nv, akiec, vasc in CLASSES order
    assert np.array_equal(images, one_batch(dataset)[0])  # no randomness


def test_augmentation_only_when_training(data_dir):
    plain, _ = one_batch(make_dataset(IDS, LABELS, training=False, size=32, batch_size=4,
                                      data_dir=data_dir))
    dataset = make_dataset(IDS, LABELS, training=True, size=32, batch_size=4, data_dir=data_dir)
    batches = [one_batch(dataset)[0] for _ in range(5)]
    assert any(not np.array_equal(batch, plain) for batch in batches)
    for batch in batches:  # flips and rotations only move pixels: same values, no color change
        assert np.array_equal(np.sort(batch, axis=None), np.sort(plain, axis=None))


def test_augment_keeps_shape_with_zoom():
    image = tf.constant(np.random.default_rng(1).integers(0, 256, (32, 32, 3), dtype=np.uint8))
    zoomed = augment(image, zoom=0.2)
    assert zoomed.shape == (32, 32, 3) and zoomed.dtype == tf.uint8


def test_unknown_label_raises(data_dir):
    with pytest.raises(ValueError):
        make_dataset(IDS, ["mel", "nv", "akiec", "melanoma"], training=False, data_dir=data_dir)
