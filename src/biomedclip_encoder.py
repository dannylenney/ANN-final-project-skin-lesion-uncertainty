"""Extract image and text embeddings using the pretrained BiomedCLIP model."""

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

MODEL_NAME = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def load_biomedclip_model(device: str | None = None):
    """Load BiomedCLIP, its image preprocessing transform, and tokenizer.

    Returns:
        preprocess, tokenizer, model, device
    """
    import torch
    from open_clip import create_model_from_pretrained, get_tokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_model_name = f"hf-hub:{MODEL_NAME}"
    model, preprocess = create_model_from_pretrained(hf_model_name)
    tokenizer = get_tokenizer(hf_model_name)

    model.to(device)
    model.eval()
    model.requires_grad_(False)

    return preprocess, tokenizer, model, device


def extract_embeddings(
    image_paths: Sequence[str | Path],
    preprocess,
    model,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Return one BiomedCLIP image embedding per image, preserving input order."""
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if len(image_paths) == 0:
        raise ValueError("image_paths cannot be empty.")

    batches = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        image_tensors = []

        for path in batch_paths:
            with Image.open(path) as image:
                image_tensors.append(preprocess(image.convert("RGB")))

        pixel_values = torch.stack(image_tensors).to(device)
        with torch.inference_mode():
            features = model.encode_image(pixel_values)

        batches.append(features.detach().float().cpu().numpy())

    embeddings = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(embeddings).all():
        raise ValueError("BiomedCLIP produced non-finite image embeddings.")
    return embeddings


def extract_text_embeddings(
    texts: Sequence[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Return one BiomedCLIP text embedding per phrase, preserving input order."""
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if len(texts) == 0:
        raise ValueError("texts cannot be empty.")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("texts must contain non-empty strings.")

    batches = []
    for start in range(0, len(texts), batch_size):
        token_batch = tokenizer(list(texts[start : start + batch_size])).to(device)
        with torch.inference_mode():
            features = model.encode_text(token_batch)
        batches.append(features.detach().float().cpu().numpy())

    embeddings = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(embeddings).all():
        raise ValueError("BiomedCLIP produced non-finite text embeddings.")
    return embeddings
