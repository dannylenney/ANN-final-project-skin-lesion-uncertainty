"""Tests for src/download_data.py on small fake files (no network)."""

import zipfile

import pandas as pd

from src.download_data import (
    LESION_GROUPINGS_CSV,
    TEST_IMAGES_ZIP,
    TRAIN_IMAGES_ZIP,
    check_lesion_csv,
    count_jpgs_in_zip,
    resolve_dest,
    verify,
)


def make_zip(path, n_jpgs, extra_names=()):
    """Write a zip with n_jpgs fake .jpg files plus any extra entries."""
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(n_jpgs):
            archive.writestr(f"images/ISIC_{i:07d}.jpg", b"fake")
        for name in extra_names:
            archive.writestr(name, b"fake")


def make_csv(path, n_rows, columns=("image", "lesion_id")):
    """Write a CSV with n_rows rows and the given columns."""
    pd.DataFrame({col: [f"{col}_{i}" for i in range(n_rows)] for col in columns}).to_csv(path, index=False)


def test_count_ignores_macosx_and_non_jpg(tmp_path):
    zip_path = tmp_path / "fake.zip"
    make_zip(zip_path, 3, extra_names=["__MACOSX/images/._ISIC_0000000.jpg", "images/LICENSE.txt"])
    assert count_jpgs_in_zip(zip_path) == 3


def test_verify_passes_with_right_counts(tmp_path):
    make_zip(tmp_path / TRAIN_IMAGES_ZIP, 5)
    make_zip(tmp_path / TEST_IMAGES_ZIP, 2)
    make_csv(tmp_path / LESION_GROUPINGS_CSV, 5)
    results = verify(tmp_path, train_images=5, test_images=2, lesion_rows=5)
    assert [passed for passed, _ in results] == [True, True, True]


def test_verify_fails_with_wrong_image_count(tmp_path):
    make_zip(tmp_path / TRAIN_IMAGES_ZIP, 4)
    make_zip(tmp_path / TEST_IMAGES_ZIP, 2)
    make_csv(tmp_path / LESION_GROUPINGS_CSV, 5)
    results = verify(tmp_path, train_images=5, test_images=2, lesion_rows=5)
    assert [passed for passed, _ in results] == [False, True, True]


def test_verify_fails_when_files_missing(tmp_path):
    results = verify(tmp_path, train_images=5, test_images=2, lesion_rows=5)
    assert not any(passed for passed, _ in results)
    assert all("missing" in message for _, message in results)


def test_csv_fails_without_lesion_id_column(tmp_path):
    csv_path = tmp_path / LESION_GROUPINGS_CSV
    make_csv(csv_path, 5, columns=("image", "diagnosis"))
    passed, message = check_lesion_csv(csv_path, expected_rows=5)
    assert not passed
    assert "NO lesion_id" in message


def test_csv_fails_with_wrong_row_count(tmp_path):
    csv_path = tmp_path / LESION_GROUPINGS_CSV
    make_csv(csv_path, 4)
    passed, _ = check_lesion_csv(csv_path, expected_rows=5)
    assert not passed


def test_resolve_dest_order(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/from/env")
    assert str(resolve_dest("given")) == "given"
    assert str(resolve_dest()) == "/from/env"
    monkeypatch.delenv("DATA_DIR")
    assert str(resolve_dest()) == "data/raw"
