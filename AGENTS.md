# AGENTS.md

Rules for any coding agent working in this repo. Read this before every task.
Changes to this file go through a PR like any other change. When an agent repeats a mistake, add a rule.

## What this project does

We classify dermatoscope images of skin lesions (HAM10000, 7 diagnoses) and measure how uncertain the model is about each image.
Three frozen pretrained encoders (ImageNet, BiomedCLIP, MONET) turn each image into an embedding.
We train the same small Keras classifier on each encoder's embeddings, measure uncertainty with Monte Carlo dropout, test whether that uncertainty points at real mistakes, and explain uncertain cases with heatmaps and text concepts.

## Folder layout

```
data/              raw and processed data (NOT in git, see README for download)
embeddings/        saved encoder outputs as .npy (NOT in git)
src/               reusable Python code (data loading, splits, models, uncertainty, evaluation)
notebooks/         Colab or VSCode notebooks that call code in src/
tests/             pytest tests for src/
results/figures/   plots used in the blog post
docs/              blog post drafts and notes
```

## Install, run, test

Local setup (VSCode), once per laptop:

```bash
conda create -n ann python=3.11
conda activate ann
pip install -r requirements.txt
pytest                      # run all tests
```

## Where code runs

- **VSCode on our laptops** for writing code, running tests and light experiments on saved embeddings.
- **Google Colab with a GPU** for heavy steps: running the encoders over all images and any long training runs.
- Colab notebooks start by cloning this repo and importing from `src/`:
  ```python
  !git clone https://github.com/dannylenney/ANN-final-project-skin-lesion-uncertainty.git
  %cd ANN-final-project-skin-lesion-uncertainty
  ```
- Data location: locally in `data/raw/`, in Colab in Google Drive under `MyDrive/ANN-project/data/raw/`.

## Workflow

- Every change starts as a GitHub Issue made from a template (feature, experiment or chore).
- One issue = one branch = one PR. Keep PRs under about 300 changed lines; if bigger, split the issue.
- **Branch names:** `<type>/<issue-number>-<short-description>`, lowercase with hyphens.
  Types: `feature`, `experiment`, `fix`, `chore`, `docs`. Example: `feature/3-lesion-split`.
- **Commit messages:** Conventional Commits, `<type>: <short summary>` in the imperative, under 72 characters, with the issue number.
  Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, plus `exp` for experiments. Example: `feat: add lesion-level train/val split (#3)`.
- **AI tools:** we mainly use Claude (Claude Code and claude.ai), with Opus or Sonnet depending on who has which plan. Tools may change during the project, so always record the actual tool and model used in the issue and PR, not a default.
- **Reviews:** when a PR is ready, post the link in the group WhatsApp. Whoever is free first reviews it. Nobody approves their own PR.

## Coding style and libraries

- Python 3. Classifiers, training and MC dropout use TensorFlow/Keras only.
- PyTorch (and open_clip) is allowed ONLY in the encoder step that turns images into embeddings. Everything after that reads saved `.npy` files.
- Use numpy, pandas, scikit-learn (metrics, splits) and matplotlib. Ask before adding any other library.
- Put reusable logic in `src/`, not in notebooks. Notebooks call functions; they do not define the core logic.
- Small functions with docstrings. Clear names (`train_loader`, not `tl`).
- Set random seeds (numpy, tensorflow) at the start of every training or evaluation run.

## Rules

- Never edit, move or delete anything in `data/` or `embeddings/`. Never commit them.
- Never hard-code file paths. Read the data folder from one setting (`DATA_DIR`) so the same code works locally and in Colab.
- Clear all notebook outputs before committing.
- Never commit API keys, tokens or passwords.
- Only change the files the issue asks for. If something else looks wrong, say so in the PR instead of fixing it.
- Always split train/validation by `lesion_id`. Never let the same lesion appear in two splits.
- Never train or tune on the official ISIC 2018 test set. It is only used by the final evaluation script.
- Do not use BatchNormalization in the classifier head (it changes behaviour when MC dropout calls the model with `training=True`).
- Plan first: before writing code, list the steps and the files you will change, and wait for approval.
- End every commit you help write with a `Co-authored-by:` line naming the model, e.g. `Co-authored-by: Claude <noreply@anthropic.com>`.
- Do not invent functions, library features or numbers. If unsure whether something exists, say so.
