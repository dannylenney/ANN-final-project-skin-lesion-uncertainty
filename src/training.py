"""Shared training helpers: seeds, class weights, training, prediction files (#12-#15)."""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score

from src.data import CLASSES, REPO_ROOT

PREDICTIONS_DIR = REPO_ROOT / "results" / "predictions"
PROB_COLUMNS = [f"p_{name}" for name in CLASSES]


def set_seed(seed):
    """Seed Python, numpy and TensorFlow. Call before building the datasets and the model,
    so the seed also controls the initial weights."""
    tf.keras.utils.set_random_seed(seed)


def class_weights(labels, sqrt=False):
    """Return {class index: weight} in CLASSES order from a list of class names.

    Balanced weights are n / (n_classes * n_c), as in scikit-learn. With sqrt=True,
    the square root of the balanced weights (milder for very rare classes).
    """
    counts = pd.Series(list(labels)).value_counts().reindex(CLASSES, fill_value=0)
    if (counts == 0).any():
        raise ValueError(f"No examples of: {list(counts[counts == 0].index)}")
    weights = len(labels) / (len(CLASSES) * counts)
    if sqrt:
        weights = np.sqrt(weights)
    return {index: float(weights[name]) for index, name in enumerate(CLASSES)}


def train_model(model, train_ds, val_ds, class_weight, seed, lr=0.001, epochs=50, patience=5):
    """Train with Adam and early stopping on validation loss; return the Keras History.

    The best weights (lowest validation loss) are restored at the end. The seed is set
    again here, but call set_seed(seed) before building the model too (see set_seed).
    """
    set_seed(seed)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                                      restore_best_weights=True)
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight,
                     callbacks=[early_stopping], shuffle=False, verbose=2)  # datasets shuffle themselves


def save_predictions(model_name, seed, split_name, image_ids, labels, probs, out_dir=None):
    """Write <model>_seed<k>_<split>.csv (image_id, label, p_akiec ... p_vasc); return its path.

    probs must have one row per image and one column per class in CLASSES order.
    out_dir defaults to results/predictions in the repo root.
    """
    probs = np.asarray(probs)
    if probs.shape != (len(image_ids), len(CLASSES)) or len(labels) != len(image_ids):
        raise ValueError(f"Expected {len(image_ids)} labels and probs of shape "
                         f"({len(image_ids)}, {len(CLASSES)}), got {len(labels)} and {probs.shape}")
    table = pd.DataFrame(probs, columns=PROB_COLUMNS)
    table.insert(0, "label", list(labels))
    table.insert(0, "image_id", list(image_ids))
    folder = Path(out_dir) if out_dir is not None else PREDICTIONS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{model_name}_seed{seed}_{split_name}.csv"
    table.to_csv(path, index=False)
    return path


def balanced_accuracy_from_file(path):
    """Balanced accuracy of the most likely class in a predictions file."""
    table = pd.read_csv(path)
    predicted = np.array(CLASSES)[table[PROB_COLUMNS].to_numpy().argmax(axis=1)]
    return balanced_accuracy_score(table["label"], predicted)
