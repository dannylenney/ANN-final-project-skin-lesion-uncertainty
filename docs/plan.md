# Project plan and roadmap

Last updated: Nov 24, 2026.
**Due: blog post + GitHub repo, Tue Dec 8, 23:59.** Group presentation after that (date on Canvas).

## How to use this file

- Every task below becomes **one GitHub issue** (template in brackets), then one branch, then one PR.
- Copy the task's **Done when** list into the issue's "Definition of done". The PR that finishes the task ticks its box here.
- If the plan changes, change this file in a PR. Don't just say it in the group chat.
- Owners are a suggestion. Confirm in the group chat and assign yourself on the issue.

---

## 1. What we're building

**Research question:** Which models know when they are unsure about a skin lesion, does that doubt point at real mistakes (especially for the rare classes), and what do CNN heatmaps show for unsure cases?

**Data:** HAM10000 (ISIC 2018 Task 3), 10,015 dermatoscope images, 7 diagnoses. Train and validation split by `lesion_id`. The official ISIC 2018 test set (1,512 images) is used once, at the end.

**Main metric:** balanced accuracy. Always guessing "mole" already scores high on plain accuracy, so plain accuracy hides how the rare classes do. We also report recall per class and melanoma sensitivity.

**Models**

| Model | Encoder | Training | Why |
|---|---|---|---|
| Scratch CNN | Small CNN, no pretraining | End to end on HAM10000 | Shows what pretraining adds |
| ImageNet CNN | EfficientNetB0, ImageNet weights | Fine-tuned | Strong general baseline |
| BiomedCLIP | ViT, medical figure-caption pairs | Frozen + Keras head | Medical vision-language model |

**Doubt measures**

- **Seed ensemble (main):** train each model several times with different seeds. Doubt = entropy of the averaged probabilities, plus the share of runs that disagree with the majority.
- **Cross-model disagreement (main):** the same idea, across the three models.
- **Softmax confidence (baseline):** doubt = 1 - (top predicted probability) from a single run.
- **MC dropout (stretch):** only on the two CNNs.

**Is doubt useful?** For every measure, overall and **per class**:
- **Error-detection AUROC:** does higher doubt line up with wrong predictions?
- **Referral curve:** if the most doubtful X% go to a doctor, how much does balanced accuracy on the rest go up, compared with sending a random X%?

**Explanations**
- **Heatmaps:** Grad-CAM on the two CNNs.

**Hypotheses**
- **H1:** the fine-tuned ImageNet CNN beats the scratch CNN on balanced accuracy. Open: where BiomedCLIP lands.
- **H2:** within each class, seed-ensemble and cross-model doubt flag mistakes better than softmax confidence.

**What changed after Lorenzo's feedback (Nov 10)**

| His point | Our change |
|---|---|
| Frozen non-medical encoder will underperform | Fine-tune ImageNet CNN, add scratch CNN |
| Follow up on MONET right away | Image side works (PR #4). T1.2 recorded the concept check and scope decision |
| Model disagreement is doubt | Cross-model disagreement is a main measure |
| MC dropout on a frozen encoder only measures the head | Seed ensembles are the main measure. MC dropout is a stretch goal |
| Rare classes are most confusing | Every doubt result is broken down by class |

---

## 2. Must-have vs stretch

**The project is complete with the must-haves.** Only start stretch goals once Phase 3's must-haves are merged.

| Must-have | Stretch (cut in this order if late) |
|---|---|
| Data split + loaders | MC dropout on the CNNs |
| 3 models trained on the same split | Heatmaps for BiomedCLIP |
| 3 seeds per model |  |
| Doubt measures + AUROC + referral curves, per class |  |
| Grad-CAM on the CNNs | Error analysis on metadata (age, body site) |
| Unsure-cases gallery | |
| One final test-set run | |
| Blog post, README, presentation | |

---

## 3. Timeline

| Week | Dates | Goal | Milestone |
|---|---|---|---|
| 0 | Nov 16-18 | Unblock | PR #4 and plan PR merged, data on Drive |
| 1 | Nov 17-22 | Data + encoders | Split merged, embeddings saved, concept check concluded |
| 2 | Nov 23-29 | Models | All 3 models trained with 3 seeds, validation predictions saved |
| 3 | Nov 30-Dec 4 | Doubt + explanations | H2 answered on validation, gallery figure done |
| 4 | Dec 5-8 | Final + write-up | Test-set run (by Dec 5), blog and README merged by Dec 8 |

Rule of thumb: if a task is more than 2 days late, raise it in the group chat and cut something from the stretch column.

---

## 4. Suggested owners

| Track | Tasks | Suggested owner |
|---|---|---|
| Data + training | T0.3, T1.1, T2.1-T2.4 | Danny |
| Encoders + explanations | T1.2-T1.4, T3.3 | Ofri |
| Doubt + evaluation | T3.1, T3.2, T3.5, T4.1 | Jachin |
| Write-up | T4.2-T4.5 | Everyone, split by section |

Whoever doesn't own a task reviews it. That way everyone reads every part of the code, which helps with the oral check.

---

## 5. Tasks

### Phase 0: Unblock (by Wed Nov 18)

**T0.1 Merge MONET setup (PR #4)** (no new issue)
- Ofri marks PR #4 "Ready for review" and merges it.
- Done when:
  - [ ] PR #4 merged, issue #3 closed automatically
  - [ ] Issue #3 has the `feature` label

**T0.2 Merge this plan (issue #6)**
- Done when:
  - [ ] `docs/plan.md` and the updated "What this project does" in AGENTS.md are on main
  - [ ] Repo "About" description no longer says MC dropout is the main doubt measure
  - [ ] Jachin and Ofri have read the plan and confirmed owners in the group chat

**T0.3 Download HAM10000 (issue #5)** [Chore]
- Download from https://challenge.isic-archive.com/data/ (Task 3): training images, training ground truth, test images, test ground truth, and the metadata file with `lesion_id`.
- Put them in a shared Drive folder `MyDrive/ANN-project/data/raw/`. Everyone adds a shortcut to their own My Drive so the Colab paths match.
- Done when:
  - [ ] 10,015 training images and 1,512 test images in the Drive folder (counted in Colab, not by eye)
  - [ ] Ground-truth CSVs and metadata CSV open with pandas in Colab for all three of us
  - [ ] Issue #5 closed with a comment listing the exact file names

**T0.4 Ask Lorenzo one question** (no issue, just email or class)
- "For the frozen encoders, seed ensembles only vary the head. Is cross-model disagreement enough to cover them?"
- Done when: his answer is written into Open questions below.

### Phase 1: Data and encoders (Nov 17-22)

**T1.1 Data loader and lesion-level split** [Feature] (needs T0.3)
- `src/data.py`:
  - `load_metadata(data_dir)`: one row per image with `image_id`, `lesion_id`, `label`
  - `make_split(df, val_frac, seed)`: grouped by `lesion_id`, stratified by label as far as grouping allows
  - `load_test(data_dir)`
- Save the split as `results/splits/train_val_split.csv` (`image_id`, `split`) and commit it, so everyone trains on exactly the same split.
- Done when:
  - [ ] `pytest tests/test_data.py` passes
  - [ ] No `lesion_id` in both train and val (tested)
  - [ ] All 7 classes in val; class shares in train vs val differ by under 2 percentage points (tested)
  - [ ] Split CSV committed and documented in README

**T1.2 MONET vs BiomedCLIP concept check (#9)**
- On 100 training images, both models retrieved visible hair, neither top five showed a recognizable ruler, and the pigment-network rankings were inconclusive.
- Decision: use the three classifiers in section 1 and CNN heatmaps. The check assessed concept retrieval only.

**T1.3 BiomedCLIP encoder** [Feature]
- `src/biomedclip_encoder.py` with the same interface as `src/monet_encoder.py`: `load_...()` and `extract_embeddings(...)`.
- Done when:
  - [ ] Unit tests with fake models pass
  - [ ] Smoke test on one real image prints the embedding shape (recorded in the PR)
  - [ ] Any new library added to a requirements file and agreed in the group chat

**T1.4 Extract embeddings for all images** [Feature] (needs T1.1, T1.3; Colab GPU)
- `notebooks/01_extract_embeddings.ipynb` calls `src/`. Write the files to Drive `MyDrive/ANN-project/embeddings/` (never to git).
- Done when:
  - [ ] BiomedCLIP has a train/val file with one row per training image and a test file with one row per test image, each with a matching `image_id` file
  - [ ] No NaNs (checked in the notebook)
  - [ ] Extraction runtime written in the PR (helps plan the seed runs)
  - [ ] Notebook outputs cleared before commit

### Phase 2: Models (Nov 21-29)

**T2.1 Keras head on embeddings** [Feature] (needs T1.4)
- `src/heads.py`:
  - `build_head(input_dim)`: Dense 256 relu, Dropout 0.3, Dense 128 relu, Dropout 0.3, Dense 7 softmax. No BatchNorm.
  - `train_head(X_train, y_train, X_val, y_val, seed)`: Adam, class weights, early stopping on validation loss.
- Logistic regression on the same embeddings as a sanity baseline.
- Done when:
  - [ ] Head trains on BiomedCLIP embeddings
  - [ ] Validation predictions saved in the shared format (section 6)
  - [ ] Validation balanced accuracy for head vs logistic regression in the PR

**T2.2 Scratch CNN** [Feature] (needs T1.1)
- `src/cnn_models.py`: `build_scratch_cnn()`, a few conv blocks and a small dense head.
- `src/images.py`: tf.data pipeline (resize, normalize, flips and small rotations on train only).
- Done when:
  - [ ] Trains in Colab with class weights and early stopping
  - [ ] Validation predictions saved; balanced accuracy and training curve in the PR

**T2.3 Fine-tuned EfficientNetB0** [Feature] (needs T2.2's pipeline)
- Stage 1: freeze the base and train a new head. Stage 2: unfreeze the top layers and train with a low learning rate. Keep the base's BatchNorm layers in inference mode (Keras fine-tuning guide).
- Done when:
  - [ ] Validation predictions saved; balanced accuracy for stage 1 vs stage 2 in the PR
  - [ ] Time per run recorded

**T2.4 Seed runs** [Experiment] (needs T2.1-T2.3)
- Train the three models with seeds `[0, 1, 2]` for nine runs.
- Done when:
  - [ ] Validation predictions for every model x seed saved
  - [ ] Table: validation balanced accuracy per model as mean and spread over seeds, plus recall per class
  - [ ] H1 answered on validation in the issue's Results

### Phase 3: Doubt and explanations (Nov 28-Dec 4)

**T3.1 Doubt scores** [Feature] (can start any time, uses fake data)
- `src/doubt.py`: `softmax_doubt`, `ensemble_entropy`, `seed_disagreement`, `cross_model_disagreement`, plus `mc_dropout_probs` (stretch).
- Done when:
  - [x] Unit tests on small hand-made probability arrays with known answers pass

**T3.2 Does doubt find mistakes?** [Experiment] (H2; needs T2.4, T3.1)
- `src/evaluate.py`: `error_detection_auroc(doubt, is_wrong)` overall and per true class. If a class has no errors (or only errors), report that instead of a number. Also `referral_curve(...)` vs random referral, with the random curve averaged over many shuffles.
- Done when:
  - [ ] `results/figures/referral_curves.png` and `results/figures/auroc_per_class.png`
  - [ ] Table: AUROC per doubt measure x class, with the number of images and errors per class
  - [ ] H2 marked supported / not supported / mixed in Results

**T3.3 Grad-CAM for the CNNs** [Feature] (needs T2.2, T2.3)
- `src/explain.py`: `grad_cam(model, image, layer_name)` that returns a heatmap to overlay on the image.
- Done when:
  - [ ] Unit test checks the heatmap's shape and value range
  - [ ] A sample figure for 4 validation images per CNN

**T3.4 Concept explanations (#19)** — skipped after T1.2.

**T3.5 Unsure-cases gallery** [Feature] (needs T3.2 and T3.3)
- For each class, the 3 most doubtful validation images: image, true vs predicted label, doubt score, and Grad-CAM for the CNNs.
- Done when:
  - [ ] `results/figures/unsure_gallery.png`, readable when shrunk to blog width
  - [ ] Each of us has written one sentence about what the unsure cases have in common

### Phase 4: Final results and write-up (Dec 4-8)

**T4.1 Final test-set run** [Experiment] (by Fri Dec 5)
- Settings frozen first: the chosen models, number of seeds and doubt measures, written in the issue before running. Run once on the 1,512 test images.
- Done when:
  - [ ] Final table: balanced accuracy, melanoma sensitivity, AUROC per doubt measure
  - [ ] Commit hash and notebook that produced it are linked in Results
  - [ ] Nothing was re-tuned after seeing test numbers

**T4.2 Blog post** [Feature] (draft by Dec 6, final Dec 8)
- `docs/blog.md`, or wherever Canvas says. Non-technical: a reader who has never trained a model should follow it.
- Outline: the problem → the data → three models in one picture → do models know when they're unsure? (referral curve) → the unsure cases (gallery) → can they explain themselves? (CNN heatmaps) → limitations → how we used AI → sources.
- Done when:
  - [ ] All three of us have reviewed it in the PR
  - [ ] Every number in it matches a table in an experiment issue
  - [ ] Limitations cover: not a medical device; HAM10000 is mostly fair skin from two clinics; only half the labels are confirmed by histopathology
  - [ ] Citations: HAM10000, BiomedCLIP, EfficientNet, Grad-CAM, plus any code we took from elsewhere

**T4.3 Final README** [Feature]
- Done when:
  - [ ] A new person can go from clone to figures following the README alone (one of us tests this on a clean Colab)
  - [ ] One-paragraph results summary with the main figure
  - [ ] AI-use statement and citations

**T4.4 Repo cleanup** [Chore]
- Done when:
  - [ ] Every issue is closed with a PR or a comment saying why
  - [ ] No merged branches left; notebook outputs cleared
  - [ ] AGENTS.md has a rule for every repeated agent mistake we hit

**T4.5 Presentation + oral check prep** [Chore]
- Slides reuse the blog figures. The oral check will ask about our code, so we hold one session where each of us walks the others through their `src/` files.
- Done when:
  - [ ] Slides done and practiced once with a timer
  - [ ] Each of us can explain every file in `src/` without notes

---

## 6. Shared conventions

- **Class order everywhere:** `akiec, bcc, bkl, df, mel, nv, vasc` (alphabetical). Map the ISIC column names to this order explicitly; don't rely on CSV column order.
- **Seeds:** `[0, 1, 2]`.
- **Paths:** read the data folder from `DATA_DIR`; never hard-code paths.
- **Prediction files:** `results/predictions/<model>_seed<k>_<split>.csv`, with columns `image_id, label, p_akiec, p_bcc, p_bkl, p_df, p_mel, p_nv, p_vasc`. CSV because `.npy` is git-ignored. Model names: `scratch_cnn, imagenet_cnn, biomedclip`.
- **Figures:** `results/figures/`, PNG, with a title and axis labels a non-technical reader understands.

## 7. Workflow habits (part of the grade)

- One issue, one branch, one PR. PRs under ~300 changed lines.
- The issue's planning model and chat filled in. Write down what we rejected or changed from the AI's suggestion, and why.
- Reviews actually run the "How to test" steps and say what was checked (see PR #4's review).
- When an agent makes the same mistake twice, add a rule to AGENTS.md in a small PR.
- Experiments get Results filled in before closing, including "it didn't work".

## 8. Open questions

- For frozen encoders, seed ensembles only vary the head. Is cross-model disagreement enough? (T0.4)
- Can we get Derm7pt in time? Stretch only.
- Presentation date and blog format: check Canvas.

## 9. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Colab GPU limits slow down CNN seed runs | 3 seeds, smaller image size, save checkpoints to Drive |
| A class has too few validation errors for AUROC | Report counts and show examples instead of a number |
| Running late | Cut from the stretch column in order; never cut the final test run or the write-up |
