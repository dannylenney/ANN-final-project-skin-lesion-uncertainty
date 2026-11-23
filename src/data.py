"""Load HAM10000 labels and lesion ids, and make the lesion-level train/validation split.

The data folder is the folder that download_data.py fills (it holds the unzipped
ISIC 2018 Task 3 files). See resolve_data_dir for how it is found.
"""

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Class order used everywhere in the project (docs/plan.md, section 6).
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# ISIC column name -> our class name. Explicit, so CSV column order never matters.
ISIC_COLUMN_TO_CLASS = {
    "AKIEC": "akiec",
    "BCC": "bcc",
    "BKL": "bkl",
    "DF": "df",
    "MEL": "mel",
    "NV": "nv",
    "VASC": "vasc",
}

TRAIN_LABELS_CSV = "ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv"
TEST_LABELS_CSV = "ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv"
LESION_GROUPINGS_CSV = "ISIC2018_Task3_Training_LesionGroupings.csv"

DEFAULT_DATA_DIR = "data/raw"

# The committed split, relative to the repo root (not the working directory).
REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_CSV = REPO_ROOT / "results" / "splits" / "train_val_split.csv"
EXPECTED_SPLIT_ROWS = 10015


def resolve_data_dir(data_dir=None):
    """Return the data folder that holds the HAM10000 files.

    If data_dir is given, it must contain the lesion groupings CSV, or
    FileNotFoundError is raised (no fallback). If data_dir is None, the first of
    $DATA_DIR and data/raw that contains it is used; FileNotFoundError if neither does.
    """
    if data_dir is not None:
        candidates = [data_dir]
    else:
        candidates = [os.environ.get("DATA_DIR"), DEFAULT_DATA_DIR]
    tried = [Path(c) for c in candidates if c]
    for folder in tried:
        if (folder / LESION_GROUPINGS_CSV).exists():
            return folder
    raise FileNotFoundError(
        f"No HAM10000 files found in: {', '.join(str(f) for f in tried)}. "
        "Pass data_dir, set DATA_DIR, or run python -m src.download_data --unzip."
    )


def _labels_from_onehot(table):
    """Turn the ISIC one-hot columns into a Series of class names (our CLASSES)."""
    missing = set(ISIC_COLUMN_TO_CLASS) - set(table.columns)
    if missing:
        raise ValueError(f"Missing label columns: {sorted(missing)}")
    onehot = table[list(ISIC_COLUMN_TO_CLASS)]
    if not (onehot.sum(axis=1) == 1).all():
        raise ValueError("Every image must have exactly one label set to 1")
    return onehot.idxmax(axis=1).map(ISIC_COLUMN_TO_CLASS)


def load_metadata(data_dir=None):
    """Return one row per training image: image_id, lesion_id, label, diagnosis_confirm_type."""
    folder = resolve_data_dir(data_dir)
    labels = pd.read_csv(folder / TRAIN_LABELS_CSV)
    groupings = pd.read_csv(folder / LESION_GROUPINGS_CSV)

    table = pd.DataFrame({"image_id": labels["image"], "label": _labels_from_onehot(labels)})
    groupings = groupings.rename(columns={"image": "image_id"})
    merged = table.merge(groupings, on="image_id", how="outer", validate="one_to_one", indicator=True)
    if not (merged["_merge"] == "both").all():
        raise ValueError("Ground truth and lesion groupings do not list the same images")
    return merged[["image_id", "lesion_id", "label", "diagnosis_confirm_type"]]


def load_test(data_dir=None):
    """Return the official test set labels: image_id, label (same class names as training)."""
    folder = resolve_data_dir(data_dir)
    labels = pd.read_csv(folder / TEST_LABELS_CSV)
    return pd.DataFrame({"image_id": labels["image"], "label": _labels_from_onehot(labels)})


def make_split(df, val_frac=0.2, seed=0):
    """Split images into train and val, grouped by lesion_id and stratified by label.

    Uses StratifiedGroupKFold with round(1 / val_frac) folds and takes the first fold
    as validation, so val_frac is rounded to 1/k (0.2 -> 5 folds). Returns a table
    with image_id and split ("train" or "val"), in the same row order as df.

    Warning: the result depends on the scikit-learn and pandas versions, even with the
    same seed. For training and evaluation always use load_split(), which reads the
    committed split file. Only call make_split to deliberately create a new split.
    """
    n_folds = round(1 / val_frac)
    folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    _, val_rows = next(folds.split(df, df["label"], groups=df["lesion_id"]))
    split = pd.Series("train", index=df.index)
    split.iloc[val_rows] = "val"
    return pd.DataFrame({"image_id": df["image_id"], "split": split})


def load_split(path=None, expected_rows=EXPECTED_SPLIT_ROWS):
    """Return the committed train/val split (image_id, split). Use this, not make_split.

    path defaults to results/splits/train_val_split.csv in the repo root. Raises
    ValueError if the file doesn't have expected_rows rows, the columns image_id and
    split, unique image ids, and only "train"/"val" values.
    """
    split = pd.read_csv(SPLIT_CSV if path is None else path)
    if list(split.columns) != ["image_id", "split"]:
        raise ValueError(f"Expected columns ['image_id', 'split'], got {list(split.columns)}")
    if len(split) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(split)}")
    if split["image_id"].duplicated().any():
        raise ValueError("Some image_id values appear more than once")
    unexpected = set(split["split"]) - {"train", "val"}
    if unexpected:
        raise ValueError(f"Unexpected split values: {sorted(unexpected)}")
    return split


def split_report(df, split):
    """Return (per-class counts and shares for train and val, largest share difference in points)."""
    merged = df[["image_id", "label"]].merge(split, on="image_id", validate="one_to_one")
    counts = pd.crosstab(merged["label"], merged["split"]).reindex(CLASSES, fill_value=0)
    shares = counts / counts.sum() * 100
    report = counts.join(shares, rsuffix="_pct")
    largest_diff = (shares["train"] - shares["val"]).abs().max()
    return report, largest_diff
