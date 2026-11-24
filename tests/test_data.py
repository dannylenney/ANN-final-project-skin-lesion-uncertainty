"""Tests for src/data.py on small fake data (no real HAM10000 files needed)."""

import numpy as np
import pandas as pd
import pytest

from src.data import (
    CLASSES,
    LESION_GROUPINGS_CSV,
    TEST_LABELS_CSV,
    TRAIN_LABELS_CSV,
    load_metadata,
    load_split,
    load_test,
    make_split,
    resolve_data_dir,
    split_report,
)

# Roughly HAM10000's imbalance, scaled down: lesions per class.
FAKE_LESIONS_PER_CLASS = {"akiec": 60, "bcc": 90, "bkl": 180, "df": 40, "mel": 180, "nv": 900, "vasc": 40}


def make_fake_metadata(seed=0):
    """Fake metadata with 1-4 images per lesion and one label per lesion."""
    rng = np.random.default_rng(seed)
    rows = []
    for label, n_lesions in FAKE_LESIONS_PER_CLASS.items():
        for lesion in range(n_lesions):
            for _ in range(rng.integers(1, 5)):
                rows.append({"image_id": f"IMG_{len(rows):05d}", "lesion_id": f"{label}_{lesion}", "label": label})
    return pd.DataFrame(rows)


def write_fake_files(folder, columns):
    """Write tiny ground-truth CSVs (with the given column order) and a groupings CSV."""
    labels = {"ISIC_1": "MEL", "ISIC_2": "NV", "ISIC_3": "AKIEC"}
    onehot = pd.DataFrame([{col: float(col == isic) for col in columns} for isic in labels.values()])
    onehot.insert(0, "image", list(labels))
    for relative in (TRAIN_LABELS_CSV, TEST_LABELS_CSV):
        (folder / relative).parent.mkdir(parents=True, exist_ok=True)
        onehot.to_csv(folder / relative, index=False)
    pd.DataFrame({
        "image": list(labels),
        "lesion_id": ["HAM_1", "HAM_2", "HAM_1"],
        "diagnosis_confirm_type": ["histopathology"] * 3,
    }).to_csv(folder / LESION_GROUPINGS_CSV, index=False)


def test_no_lesion_in_both_splits():
    df = make_fake_metadata()
    split = make_split(df, val_frac=0.2, seed=0)
    lesions = df.assign(split=split["split"]).groupby("lesion_id")["split"].nunique()
    assert (lesions == 1).all()


def test_all_classes_in_val_and_shares_close():
    df = make_fake_metadata()
    split = make_split(df, val_frac=0.2, seed=0)
    report, largest_diff = split_report(df, split)
    assert (report["val"] > 0).all()
    assert list(report.index) == CLASSES
    assert largest_diff < 2


def test_every_image_in_exactly_one_split_and_seed_repeats():
    df = make_fake_metadata()
    split = make_split(df, seed=3)
    assert split["image_id"].tolist() == df["image_id"].tolist()
    assert set(split["split"]) == {"train", "val"}
    assert split.equals(make_split(df, seed=3))


@pytest.mark.parametrize("columns", [
    ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"],  # order in the real ISIC files
    ["VASC", "AKIEC", "NV", "DF", "MEL", "BKL", "BCC"],  # shuffled
])
def test_loaders_map_columns_by_name(tmp_path, columns):
    write_fake_files(tmp_path, columns)
    meta = load_metadata(tmp_path)
    assert meta["label"].tolist() == ["mel", "nv", "akiec"]
    assert meta.columns.tolist() == ["image_id", "lesion_id", "label", "diagnosis_confirm_type"]
    assert load_test(tmp_path)["label"].tolist() == ["mel", "nv", "akiec"]


def test_resolve_data_dir_order(tmp_path, monkeypatch):
    given, from_env = tmp_path / "given", tmp_path / "env"
    for folder in (given, from_env):
        folder.mkdir()
        (folder / LESION_GROUPINGS_CSV).touch()
    monkeypatch.setenv("DATA_DIR", str(from_env))
    assert resolve_data_dir(given) == given
    assert resolve_data_dir() == from_env


def test_resolve_data_dir_falls_back_to_data_raw(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "empty"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "raw" / LESION_GROUPINGS_CSV).touch()
    assert str(resolve_data_dir()) == "data/raw"


def test_resolve_data_dir_raises_on_wrong_explicit_dir(tmp_path, monkeypatch):
    good = tmp_path / "env"
    good.mkdir()
    (good / LESION_GROUPINGS_CSV).touch()
    monkeypatch.setenv("DATA_DIR", str(good))  # a valid fallback exists, but must not be used
    with pytest.raises(FileNotFoundError):
        resolve_data_dir(tmp_path / "empty")


def test_resolve_data_dir_raises_when_nothing_found(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)  # so data/raw does not exist either
    with pytest.raises(FileNotFoundError):
        resolve_data_dir()


def test_load_split_reads_valid_file(tmp_path):
    path = tmp_path / "split.csv"
    pd.DataFrame({"image_id": ["A", "B", "C"], "split": ["train", "val", "train"]}).to_csv(path, index=False)
    split = load_split(path, expected_rows=3)
    assert split["split"].tolist() == ["train", "val", "train"]


@pytest.mark.parametrize("table, rows", [
    ({"image_id": ["A", "B"], "split": ["train", "val"]}, 3),        # wrong row count
    ({"image_id": ["A", "B", "C"], "fold": ["train"] * 3}, 3),       # wrong column name
    ({"image_id": ["A", "B", "C"], "split": ["train", "val", "test"]}, 3),  # unexpected value
    ({"image_id": ["A", "A", "C"], "split": ["train", "val", "val"]}, 3),   # duplicate image
])
def test_load_split_rejects_bad_files(tmp_path, table, rows):
    path = tmp_path / "split.csv"
    pd.DataFrame(table).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_split(path, expected_rows=rows)


def test_committed_split_file_is_valid(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # the default path must not depend on the working directory
    split = load_split()
    assert split["split"].value_counts().to_dict() == {"train": 8011, "val": 2004}
