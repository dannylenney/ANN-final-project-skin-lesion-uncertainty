"""Download the HAM10000 files (ISIC 2018 Challenge, Task 3) and check they are complete.

Usage:
    python -m src.download_data --dest data/raw --unzip

The destination is --dest if given, otherwise the DATA_DIR environment variable,
otherwise data/raw. Files that already exist with the right size are skipped.
"""

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

BASE_URL = "https://isic-archive.s3.amazonaws.com/challenges/2018/"

TRAIN_IMAGES_ZIP = "ISIC2018_Task3_Training_Input.zip"
TRAIN_LABELS_ZIP = "ISIC2018_Task3_Training_GroundTruth.zip"
LESION_GROUPINGS_CSV = "ISIC2018_Task3_Training_LesionGroupings.csv"
TEST_IMAGES_ZIP = "ISIC2018_Task3_Test_Input.zip"
TEST_LABELS_ZIP = "ISIC2018_Task3_Test_GroundTruth.zip"

FILES = [
    TRAIN_IMAGES_ZIP,
    TRAIN_LABELS_ZIP,
    LESION_GROUPINGS_CSV,
    TEST_IMAGES_ZIP,
    TEST_LABELS_ZIP,
]

EXPECTED_TRAIN_IMAGES = 10015
EXPECTED_TEST_IMAGES = 1512
EXPECTED_LESION_ROWS = 10015

CHUNK_BYTES = 1024 * 1024


def resolve_dest(dest=None):
    """Return the destination folder: dest, else $DATA_DIR, else data/raw."""
    if dest:
        return Path(dest)
    if os.environ.get("DATA_DIR"):
        return Path(os.environ["DATA_DIR"])
    return Path("data/raw")


def remote_size(url):
    """Return the file size in bytes reported by the server (HEAD request)."""
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def download_file(url, path):
    """Download url to path, skipping it if a file of the right size already exists.

    The download goes to a .part file first and is renamed when complete, so an
    interrupted download is never mistaken for a finished one.
    """
    expected_bytes = remote_size(url)
    if path.exists() and path.stat().st_size == expected_bytes:
        print(f"skip  {path.name} (already downloaded, {expected_bytes / 1e6:.1f} MB)")
        return

    part_path = path.with_name(path.name + ".part")
    done_bytes = 0
    last_percent = -1
    with urllib.request.urlopen(url, timeout=60) as response, open(part_path, "wb") as out:
        while True:
            chunk = response.read(CHUNK_BYTES)
            if not chunk:
                break
            out.write(chunk)
            done_bytes += len(chunk)
            percent = int(100 * done_bytes / expected_bytes)
            if percent != last_percent:
                print(f"\r      {path.name}: {done_bytes / 1e6:,.1f} / {expected_bytes / 1e6:,.1f} MB ({percent}%)",
                      end="", flush=True)
                last_percent = percent
    print()

    if done_bytes != expected_bytes:
        raise RuntimeError(f"{path.name}: got {done_bytes} bytes, expected {expected_bytes}")
    part_path.replace(path)
    print(f"done  {path.name}")


def count_jpgs_in_zip(zip_path):
    """Count .jpg files in a zip, ignoring macOS metadata entries (__MACOSX/)."""
    with zipfile.ZipFile(zip_path) as archive:
        return sum(
            1
            for name in archive.namelist()
            if name.lower().endswith(".jpg") and not name.startswith("__MACOSX/")
        )


def check_zip_images(zip_path, expected):
    """Return (passed, message) for the number of .jpg files in a zip."""
    if not zip_path.exists():
        return False, f"{zip_path.name}: file missing"
    found = count_jpgs_in_zip(zip_path)
    return found == expected, f"{zip_path.name}: {found:,} .jpg files (expected {expected:,})"


def check_lesion_csv(csv_path, expected_rows):
    """Return (passed, message) for the row count and lesion_id column of the groupings CSV."""
    if not csv_path.exists():
        return False, f"{csv_path.name}: file missing"
    table = pd.read_csv(csv_path)
    has_column = "lesion_id" in table.columns
    passed = has_column and len(table) == expected_rows
    column_note = "has lesion_id column" if has_column else "NO lesion_id column"
    return passed, f"{csv_path.name}: {len(table):,} rows (expected {expected_rows:,}), {column_note}"


def verify(dest, train_images=EXPECTED_TRAIN_IMAGES, test_images=EXPECTED_TEST_IMAGES,
           lesion_rows=EXPECTED_LESION_ROWS):
    """Run all checks on the files in dest. Returns a list of (passed, message)."""
    dest = Path(dest)
    return [
        check_zip_images(dest / TRAIN_IMAGES_ZIP, train_images),
        check_zip_images(dest / TEST_IMAGES_ZIP, test_images),
        check_lesion_csv(dest / LESION_GROUPINGS_CSV, lesion_rows),
    ]


def unzip_all(dest):
    """Extract every downloaded zip into dest, skipping macOS metadata entries."""
    for name in FILES:
        if not name.endswith(".zip"):
            continue
        print(f"unzip {name}")
        with zipfile.ZipFile(dest / name) as archive:
            members = [m for m in archive.namelist() if not m.startswith("__MACOSX/")]
            archive.extractall(dest, members=members)


def main(argv=None):
    """Download, optionally unzip, and verify. Returns 0 if every check passes, else 1."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", help="destination folder (default: $DATA_DIR, then data/raw)")
    parser.add_argument("--unzip", action="store_true", help="extract the zip files after downloading")
    args = parser.parse_args(argv)

    dest = resolve_dest(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Destination: {dest.resolve()}")

    for name in FILES:
        download_file(BASE_URL + name, dest / name)

    if args.unzip:
        unzip_all(dest)

    print("\nVerification:")
    results = verify(dest)
    for passed, message in results:
        print(f"{'PASS' if passed else 'FAIL'}  {message}")
    return 0 if all(passed for passed, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
