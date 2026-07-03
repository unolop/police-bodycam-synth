"""Face identity management for consistent character generation.

Uses IP-Adapter FaceID with SDXL to maintain consistent face identity
across frames within a scene sequence.

Pipeline:
1. Build an identity pool from reference face images (or generate synthetic ones)
2. Extract face embeddings from reference images using InsightFace
3. During generation, condition SDXL via IP-Adapter with the face embedding
   so the same person appears consistently across scene frames.
"""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


def load_identity_pool(ref_dir: Path, face_app) -> list[dict]:
    """Load face reference images and extract embeddings for identity pool.

    Args:
        ref_dir: Directory containing reference face images (one person per image).
        face_app: InsightFace FaceAnalysis instance.

    Returns:
        List of identity dicts with 'image', 'embedding', 'source_file'.
    """
    identities = []
    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    ref_images = sorted(
        p for p in ref_dir.iterdir()
        if p.suffix.lower() in image_exts
    )

    if not ref_images:
        raise FileNotFoundError(f"No reference face images found in {ref_dir}")

    for img_path in ref_images:
        image = Image.open(img_path).convert("RGB")
        img_array = np.array(image)
        img_bgr = img_array[:, :, ::-1]

        faces = face_app.get(img_bgr)
        if not faces:
            print(f"  Warning: No face detected in {img_path.name}, skipping")
            continue

        # Use the largest/most prominent face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        identities.append({
            "image": image,
            "embedding": face.embedding,
            "source_file": str(img_path),
            "bbox": face.bbox.tolist(),
        })

    print(f"Loaded {len(identities)} face identities from {ref_dir}")
    return identities


def assign_identities_to_scenes(
    num_scenes: int,
    identities: list[dict],
    seed: int = 42,
) -> dict[int, int]:
    """Assign a face identity to each scene.

    Returns mapping: scene_index -> identity_index.
    Cycles through identities so each person appears in multiple scenes.
    """
    rng = random.Random(seed)
    identity_indices = list(range(len(identities)))

    assignments = {}
    for scene_idx in range(num_scenes):
        assignments[scene_idx] = identity_indices[scene_idx % len(identity_indices)]

    # Shuffle to avoid predictable ordering
    scene_list = list(assignments.items())
    rng.shuffle(scene_list)
    shuffled = {}
    for i, (_, identity_idx) in enumerate(scene_list):
        shuffled[i] = identity_idx
    return shuffled


def setup_ip_adapter_faceid(pipe, config):
    """Load IP-Adapter FaceID weights into the SDXL pipeline.

    This modifies the pipeline in-place to accept face embeddings
    as conditioning during generation.
    """
    pipe.load_ip_adapter(
        config.ip_adapter.model_repo,
        subfolder=config.ip_adapter.subfolder,
        weight_name=config.ip_adapter.weight_name,
    )
    pipe.set_ip_adapter_scale(config.ip_adapter.scale)
    return pipe


def save_identity_pool_meta(identities: list[dict], output_path: Path):
    """Save identity pool metadata (without embeddings) for reproducibility."""
    meta = []
    for i, ident in enumerate(identities):
        meta.append({
            "identity_index": i,
            "source_file": ident["source_file"],
            "bbox": ident["bbox"],
            "embedding_dim": len(ident["embedding"]),
        })
    output_path.write_text(json.dumps(meta, indent=2))
