"""CLIP text-image alignment scoring."""

import torch
import numpy as np
from pathlib import Path
from PIL import Image


def compute_clip_scores(
    clip_model, preprocess, tokenizer,
    image_paths: list[Path], prompts: list[str],
) -> list[dict]:
    """Compute CLIP cosine similarity between each (image, prompt) pair."""
    results = []

    for img_path, prompt in zip(image_paths, prompts):
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to("cuda")
        text = tokenizer([prompt]).to("cuda")

        with torch.no_grad():
            img_features = clip_model.encode_image(image)
            txt_features = clip_model.encode_text(text)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
            score = (img_features @ txt_features.T).item()

        results.append({
            "image": str(img_path),
            "prompt": prompt[:100],
            "clip_score": score,
        })

    return results
