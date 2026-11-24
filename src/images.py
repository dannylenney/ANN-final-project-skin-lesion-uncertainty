"""tf.data pipeline for the HAM10000 training images (used by the CNNs, #13 and #14).

Each JPG is center-cropped to a square (600x450 -> 450x450) and resized, so lesions
keep their shape. Resized images are cached as uint8 and turned into 0-1 floats per
batch. Augmentation (only when training=True) changes orientation, never color.
"""

import tensorflow as tf

from src.data import CLASSES, resolve_data_dir

TRAIN_IMAGES_DIR = "ISIC2018_Task3_Training_Input"
IMAGE_SIZE = 224


def load_image(path, size=IMAGE_SIZE):
    """Read a JPG, center-crop it to a square on its shorter side, resize to size x size (uint8)."""
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    side = tf.minimum(height, width)
    image = tf.image.crop_to_bounding_box(image, (height - side) // 2, (width - side) // 2, side, side)
    image = tf.image.resize(image, [size, size], antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)


def augment(image, zoom=0.0):
    """Random orientation (and optional zoom-in) for one uint8 image. No color changes.

    A random horizontal flip plus a random 90 degree rotation already gives all 8
    orientations of the image, so a vertical flip would add nothing.
    If zoom > 0, a random square covering (1 - zoom) to 100% of the side is cropped
    and resized back to the original size.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    if zoom > 0:
        size = tf.shape(image)[0]
        scale = tf.random.uniform([], 1.0 - zoom, 1.0)
        crop = tf.cast(tf.round(scale * tf.cast(size, tf.float32)), tf.int32)
        image = tf.image.random_crop(image, tf.stack([crop, crop, 3]))
        image = tf.image.resize(image, tf.stack([size, size]), antialias=True)
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    return image


def labels_to_indices(labels):
    """Turn class names into indices in CLASSES order."""
    unknown = set(labels) - set(CLASSES)
    if unknown:
        raise ValueError(f"Unknown labels: {sorted(unknown)}")
    return [CLASSES.index(label) for label in labels]


def make_dataset(image_ids, labels, training, size=IMAGE_SIZE, batch_size=32, zoom=0.0, seed=0,
                 data_dir=None):
    """Return a tf.data.Dataset of (images, label indices) batches for training images.

    Images are float32 in [0, 1] with shape (batch, size, size, 3). With training=True
    the images are shuffled (with seed) and augmented; with training=False they keep
    the order of image_ids and are never augmented (use this for validation and predictions).
    """
    folder = resolve_data_dir(data_dir) / TRAIN_IMAGES_DIR
    paths = [str(folder / f"{image_id}.jpg") for image_id in image_ids]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels_to_indices(list(labels))))
    dataset = dataset.map(lambda path, label: (load_image(path, size), label),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    if training:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
        dataset = dataset.map(lambda image, label: (augment(image, zoom), label),
                              num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda images, label: (tf.cast(images, tf.float32) / 255.0, label))
    return dataset.prefetch(tf.data.AUTOTUNE)
