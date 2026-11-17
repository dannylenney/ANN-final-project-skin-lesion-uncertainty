# MONET Encoder Setup

## Overview

This project uses MONET (`chanwkim/monet`) as one of three pretrained image encoders for the HAM10000 skin lesion dataset.

MONET converts dermatoscope images into 768-dimensional image embeddings. These embeddings can then be used by the downstream TensorFlow/Keras classifier.

PyTorch is used only for image embedding extraction. Classifier training and uncertainty evaluation remain in TensorFlow/Keras.

## Environment Setup

We recommend a separate Conda environment with Python 3.11.

```bash
conda create -n monet python=3.11
conda activate monet
python -m pip install -r requirements-monet.txt
python -m pip install pytest
```

Check the installation:

```bash
python --version
python -m pip check
```

## Loading the Model

The reusable implementation is in `src/monet_encoder.py`.

```python
from src.monet_encoder import load_monet_model

processor, model, device = load_monet_model()

print("Device:", device)
```

The model is downloaded from Hugging Face the first time it is loaded. Subsequent runs can use the cached model files.

The loader automatically selects CUDA when available and otherwise uses the CPU.

## Extracting Image Embeddings

```python
from pathlib import Path
import numpy as np

from src.monet_encoder import load_monet_model, extract_embeddings

processor, model, device = load_monet_model()

image_paths = sorted(Path("data/raw/images").glob("*.jpg"))

embeddings = extract_embeddings(
    image_paths=image_paths,
    processor=processor,
    model=model,
    device=device,
    batch_size=32,
)

print("Embedding shape:", embeddings.shape)

np.save("embeddings/monet_embeddings.npy", embeddings)
```

The example image directory is illustrative; confirm the actual dataset layout before running extraction.

Each image produces one 768-dimensional embedding. The output array has shape `(number_of_images, 768)`.

The `data/` and `embeddings/` directories are not committed to Git.

## Hardware Notes

The current Hugging Face checkpoint is approximately 1.7 GB. A GPU is recommended for extracting embeddings across the full HAM10000 dataset. The project recommends Google Colab with a GPU for this step. CPU execution is useful for small tests.

## Running Tests

Run the MONET encoder unit tests with:

```bash
python -m pytest -v tests/test_monet_encoder.py
```

These tests use mock processor and model objects. They do not download MONET or evaluate medical classification performance.

A separate smoke test verified that the real MONET model loads and produces finite embeddings with shape `(1, 768)` from a test image.

## Important

- Do not commit raw data or generated embeddings.
- Do not use PyTorch for downstream classifier training.
- Use consistent preprocessing and embedding dimensions when generating dataset embeddings.