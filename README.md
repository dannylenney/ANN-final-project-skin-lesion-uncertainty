# Classifying Skin Lesions with Uncertainty

Final project for Artificial Neural Networks and Deep Learning B (DIS Copenhagen, Fall 2026).
Danny Lenney, Jachin Choi, Ofri Geva.

Can a medical vision-language model find the skin lesion cases it is unsure about, and explain them in words?

## Data

We use HAM10000 (10,015 dermatoscope images, 7 diagnoses) from the ISIC 2018 Challenge, Task 3.
The data is not stored in this repo. Download it from https://challenge.isic-archive.com/data/ and place it in `data/raw/`.
License: CC-BY-NC. Citation: Tschandl, P., Rosendahl, C., & Kittler, H. (2018). Scientific Data. doi:10.1038/sdata.2018.161

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
