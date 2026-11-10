"""Extract image embeddings using the pretrained MONET model."""

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


MODEL_NAME = "chanwkim/monet"


def load_monet_model(device: str | None = None):
    """Load the pretrained MONET model and image processor."""
    import torch
    from transformers import (
        AutoModelForZeroShotImageClassification,
        AutoProcessor,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForZeroShotImageClassification.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    return processor, model, device


def extract_embeddings(
    image_paths: Sequence[str | Path],
    processor,
    model,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract MONET image embeddings from a sequence of image paths."""
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if not image_paths:
        raise ValueError("image_paths cannot be empty.")

    all_embeddings = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]

        images = []
        for path in batch_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))

        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            embeddings = model.get_image_features(
                pixel_values=pixel_values
            )

        all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)
