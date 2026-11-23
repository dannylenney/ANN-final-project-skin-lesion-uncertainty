# Classifying Skin Lesions with Uncertainty

Final project for Artificial Neural Networks and Deep Learning B (DIS Copenhagen, Fall 2026).
Danny Lenney, Jachin Choi, Ofri Geva.

Can a medical vision-language model find the skin lesion cases it is unsure about, and explain them in words?

## Data

We use HAM10000 (10,015 dermatoscope images, 7 diagnoses) from the ISIC 2018 Challenge, Task 3.
The data is not stored in this repo. Download it from https://challenge.isic-archive.com/data/ and place it in `data/raw/`.
License: CC-BY-NC. Citation: Tschandl, P., Rosendahl, C., & Kittler, H. (2018). Scientific Data. doi:10.1038/sdata.2018.161

## Data split

Everyone trains on the same split: `results/splits/train_val_split.csv` (columns `image_id`, `split` = `train` or `val`).
It covers the 10,015 training images (8,011 train, 2,004 val). The official test set is not in it.

How it was made (`src/data.py`): `StratifiedGroupKFold` with 5 folds, grouped by `lesion_id` and stratified by label, seed 0; the first fold is validation.
No lesion is in both splits, all 7 classes are in validation, and class shares in train and val differ by at most 0.05 percentage points.

**Always use `load_split()`** to get the split for training and evaluation. Don't call `make_split()` again: its result depends on the library versions, even with the same seed (in Colab with scikit-learn 1.6.1 and pandas 2.2.3, 3,236 images land in a different split). The committed file is the single source of truth.

```python
from src.data import load_metadata, load_split
df = load_metadata().merge(load_split(), on="image_id")  # adds a "split" column
```

The file was made with the snippet below. It only reproduces the same file with scikit-learn 1.9.1 and pandas 3.0.5:

```python
from src.data import load_metadata, make_split
make_split(load_metadata(), val_frac=0.2, seed=0).sort_values("image_id").to_csv("results/splits/train_val_split.csv", index=False)
```

`src/data.py` finds the data folder from its `data_dir` argument, then the `DATA_DIR` environment variable, then `data/raw`.
In Colab, copy the files from the shared Drive folder to the Colab disk, unzip them there (much faster to read than Drive) and point `DATA_DIR` at that folder:

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/data/raw
!cp /content/drive/MyDrive/ANN-project/data/raw/*.zip /content/drive/MyDrive/ANN-project/data/raw/*.csv /content/data/raw/
!cd /content/data/raw && for f in *.zip; do unzip -q -o "$f"; done
import os
os.environ["DATA_DIR"] = "/content/data/raw"

from src.data import load_metadata, load_split
df = load_metadata().merge(load_split(), on="image_id")
df["split"].value_counts()  # train 8011, val 2004
```

Run this from the cloned repo folder (`%cd ANN-final-project-skin-lesion-uncertainty`) so `src` can be imported.

## Setup

```bash
pip install -r requirements.txt
pytest
```

## How we work

- Every change starts as a GitHub Issue (feature, experiment or chore template).
- Every issue is closed by one PR. Nobody pushes to `main` directly.
- Every PR needs one approval from someone who did not write it.
- AI use is documented in issues, PRs and commits. See `AGENTS.md` for the rules our coding agents follow.
