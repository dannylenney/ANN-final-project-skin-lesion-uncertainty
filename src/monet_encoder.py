"""Extract image and text embeddings using the pretrained MONET model."""

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

MODEL_NAME = "chanwkim/monet"


def load_monet_model(device: str | None = None):
    """Load the pretrained MONET model and image/text processor."""
    import torch
    from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForZeroShotImageClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    return processor, model, device


def extract_embeddings(
    image_paths: Sequence[str | Path],
    processor,
    model,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract MONET image embeddings from image paths in input order."""
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if len(image_paths) == 0:
        raise ValueError("image_paths cannot be empty.")

    all_embeddings = []
    for start in range(0, len(image_paths), batch_size):
        images = []
        for path in image_paths[start : start + batch_size]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))

        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            features = model.get_image_features(pixel_values=pixel_values)
        all_embeddings.append(features.detach().float().cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(embeddings).all():
        raise ValueError("MONET produced non-finite image embeddings.")
    return embeddings


def extract_text_embeddings(
    texts: Sequence[str],
    processor,
    model,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract MONET text embeddings for concept phrases in input order."""
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if len(texts) == 0:
        raise ValueError("texts cannot be empty.")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("texts must contain non-empty strings.")

    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        inputs = processor(
            text=list(texts[start : start + batch_size]),
            return_tensors="pt",
            padding=True,
        )
        # Keep only arguments accepted by the CLIP text-feature method.
        text_inputs = {
            key: inputs[key].to(device)
            for key in ("input_ids", "attention_mask")
            if key in inputs
        }
        with torch.inference_mode():
            features = model.get_text_features(**text_inputs)
        all_embeddings.append(features.detach().float().cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(embeddings).all():
        raise ValueError("MONET produced non-finite text embeddings.")
    return embeddings
