
# Issue #10: BiomedCLIP real-model smoke test

Run this after installing `requirements-biomedclip.txt` in an environment with access
to the pretrained Hugging Face model. It requires one local image file from HAM10000.

```python
from pathlib import Path

from src.biomedclip_encoder import (
    load_biomedclip_model,
    extract_embeddings,
    extract_text_embeddings,
)

image_path = Path("data/raw/ISIC2018_Task3_Training_Input/ISIC_0000000.jpg")
preprocess, tokenizer, model, device = load_biomedclip_model()

image_features = extract_embeddings([image_path], preprocess, model, device)
concepts = ["pigment network", "hair", "ruler"]
text_features = extract_text_embeddings(concepts, tokenizer, model, device)

print("Device:", device)
print("Image embedding shape:", image_features.shape)
print("Text embedding shape:", text_features.shape)
```

Replace the example image filename with an image that exists in your downloaded
HAM10000 folder. Record the printed shapes and device in the pull request.

This is only a one-image smoke test. It does not extract the full dataset, train a
classifier, or use the official ISIC test set.
