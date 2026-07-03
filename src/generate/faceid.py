"""Mode C: Face-ID conditioned generation using IP-Adapter FaceID + face dataset references.

Uses face images from the Korean Face ID dataset as identity references.
For each prompt, a face reference is selected and injected via IP-Adapter
so the generated bodycam image contains a recognizable, consistent face.
"""

import json
import os
import random
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


def discover_face_dataset(dataset_dir: str | Path) -> dict[str, list[Path]]:
    """Discover face ID dataset structure and return {subject_id: [image_paths]}.

    Handles the Korean folder names by walking the directory tree.
    Filters to 'normal' (no hat/mask) + 'distance1' (close) images as the
    best reference faces for identity conditioning.
    """
    dataset_dir = Path(dataset_dir)
    src_base = None

    # Walk to find the source image folder (01.원천데이터/01.Recognition)
    for root, dirs, files in os.walk(dataset_dir):
        bn = os.path.basename(root)
        if bn == '01.Recognition':
            parent = os.path.basename(os.path.dirname(root))
            if '01.' in parent or '원천' in parent:
                src_base = Path(root)
                break

    if src_base is None:
        raise FileNotFoundError(
            f"Could not find source image folder (01.원천데이터/01.Recognition) in {dataset_dir}"
        )

    subjects = {}
    for subj_dir in sorted(src_base.iterdir()):
        if not subj_dir.is_dir():
            continue
        subj_id = subj_dir.name
        # Prefer: normal accessory, close distance, direct lighting
        # Filename pattern: {id}_{location}_{lighting?}_{accessory}_{distance}_{frame}.png
        best = sorted(subj_dir.glob("*_normal_distance1_*.png"))
        if not best:
            best = sorted(subj_dir.glob("*_normal_*.png"))
        if not best:
            best = sorted(subj_dir.glob("*.png"))
        if best:
            subjects[subj_id] = best

    return subjects


def load_face_labels(dataset_dir: str | Path) -> dict[str, dict]:
    """Load demographic info (gender, birth year) per subject from label JSONs."""
    dataset_dir = Path(dataset_dir)
    label_base = None

    for root, dirs, files in os.walk(dataset_dir):
        bn = os.path.basename(root)
        if bn == '01.Recognition':
            parent = os.path.basename(os.path.dirname(root))
            if '02.' in parent or '라벨링' in parent:
                label_base = Path(root)
                break

    if label_base is None:
        return {}

    info = {}
    for subj_dir in sorted(label_base.iterdir()):
        if not subj_dir.is_dir():
            continue
        subj_id = subj_dir.name
        # Read first JSON to get demographics
        jsons = sorted(subj_dir.glob("*.json"))
        if jsons:
            with open(jsons[0]) as f:
                data = json.load(f)
            info[subj_id] = {
                "gender": data.get("gender"),
                "birth": data.get("birth"),
                "device": data.get("device"),
            }
    return info


def assign_identities(
    prompts: list[dict],
    subjects: dict[str, list[Path]],
    labels: dict[str, dict],
    seed: int = 42,
) -> list[dict]:
    """Assign a face identity to each prompt.

    For scenario 2 (missing person), tries to match target type
    (child/elderly) to appropriate age groups. Otherwise assigns randomly.
    """
    rng = random.Random(seed)
    subject_ids = sorted(subjects.keys())

    # Build age-based pools
    elderly_ids = []
    young_ids = []
    all_ids = list(subject_ids)

    for sid in subject_ids:
        lbl = labels.get(sid, {})
        birth = lbl.get("birth")
        if birth:
            age = 2022 - int(birth)
            if age >= 60:
                elderly_ids.append(sid)
            elif age <= 25:
                young_ids.append(sid)

    for prompt in prompts:
        scenario = prompt.get("scenario", 1)
        slots = prompt.get("slots", {})

        # Try to match demographics for missing person scenario
        if scenario == 2:
            target_type = slots.get("target_type", "")
            if "elderly" in target_type or "dementia" in target_type:
                pool = elderly_ids if elderly_ids else all_ids
            elif "child" in target_type or "teenage" in target_type:
                pool = young_ids if young_ids else all_ids
            else:
                pool = all_ids
        else:
            pool = all_ids

        chosen_id = rng.choice(pool)
        # Pick a random reference image for this identity
        ref_images = subjects[chosen_id]
        chosen_ref = rng.choice(ref_images)

        prompt["identity_id"] = chosen_id
        prompt["identity_ref_path"] = str(chosen_ref)
        prompt["identity_meta"] = labels.get(chosen_id, {})

    return prompts


def _init_face_app(config):
    """Initialize InsightFace for embedding extraction."""
    from insightface.app import FaceAnalysis
    import numpy as np

    face_app = FaceAnalysis(
        name=config.faceid.insightface_model,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    return face_app


def extract_face_embedding(face_app, image_path: Path) -> "torch.Tensor | None":
    """Extract 512-dim face embedding from an image using InsightFace."""
    import numpy as np
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return None

    # InsightFace expects BGR, which cv2 already provides
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    faces = face_app.get(img)
    if not faces:
        return None

    # Use the largest face
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return torch.tensor(face.embedding, dtype=torch.float32)


def generate_faceid_batch(
    pipe_and_proj: tuple,
    prompts: list[dict],
    config,
    run_dir: Path,
) -> list[Path]:
    """Generate images with face identity using IP-Adapter-FaceID.

    Each prompt must have 'identity_ref_path' set by assign_identities().
    Uses InsightFace embeddings (not pixel features) to avoid grayscale bleed.

    Args:
        pipe_and_proj: Tuple of (pipeline, image_proj) from load_faceid_pipeline()
    """
    from compel import Compel, ReturnedEmbeddingsType

    pipe, image_proj = pipe_and_proj
    dtype = torch.float16 if config.generation.fp16 else torch.float32

    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = run_dir / "image_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Load progress
    progress_file = run_dir / "generate_progress.json"
    completed_ids = set()
    if progress_file.exists():
        data = json.loads(progress_file.read_text())
        completed_ids = set(data.get("completed_ids", []))

    pending = [p for p in prompts if p["prompt_id"] not in completed_ids]
    if not pending:
        print(f"All {len(prompts)} images already generated.")
        return []

    print(f"Generating {len(pending)} face-ID images ({len(completed_ids)} already done)...")

    # Initialize InsightFace for embedding extraction
    face_app = _init_face_app(config)
    print(f"  InsightFace initialized ({config.faceid.insightface_model})")

    # Pre-extract embeddings for all unique identity refs (avoids redundant extraction)
    unique_refs = set()
    for p in pending:
        ref = p.get("identity_ref_path")
        if ref:
            unique_refs.add(ref)

    embedding_cache = {}
    for ref_path in unique_refs:
        emb = extract_face_embedding(face_app, Path(ref_path))
        if emb is not None:
            embedding_cache[ref_path] = emb

    print(f"  Extracted {len(embedding_cache)}/{len(unique_refs)} face embeddings")

    # Free InsightFace from GPU
    del face_app
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Build compel for long prompts
    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
        truncate_long_prompts=False,
    )

    generated_paths = []
    generator = torch.Generator(device="cuda").manual_seed(config.generation.seed)

    for entry in tqdm(pending, desc="Generating (face-ID)"):
        prompt_id = entry["prompt_id"]
        image_path = image_dir / f"{prompt_id}.png"

        # Get pre-extracted face embedding
        ref_path = entry.get("identity_ref_path")
        face_emb = embedding_cache.get(ref_path) if ref_path else None

        # Pre-project embedding through FaceID projection
        if face_emb is not None:
            emb_input = face_emb.unsqueeze(0).to(dtype=dtype, device="cuda")
            pos_embeds = image_proj(emb_input)  # (1, 4, 2048)
            neg_embeds_ip = image_proj(torch.zeros_like(emb_input))  # (1, 4, 2048)
            ip_embeds = torch.cat([neg_embeds_ip, pos_embeds], dim=0)  # (2, 4, 2048)
        else:
            print(f"  WARNING: No face embedding for {prompt_id}, using zero conditioning")
            zero = torch.zeros(1, config.faceid.num_tokens, 2048, dtype=dtype, device="cuda")
            ip_embeds = torch.cat([zero, zero], dim=0)

        # Encode prompts (no truncation)
        prompt_embeds, pooled_prompt_embeds = compel(entry["prompt"])
        neg_embeds, neg_pooled = compel(entry["negative_prompt"])

        # Pad to same sequence length
        max_len = max(prompt_embeds.shape[1], neg_embeds.shape[1])
        if prompt_embeds.shape[1] < max_len:
            pad = torch.zeros(1, max_len - prompt_embeds.shape[1], prompt_embeds.shape[2],
                              device=prompt_embeds.device, dtype=prompt_embeds.dtype)
            prompt_embeds = torch.cat([prompt_embeds, pad], dim=1)
        if neg_embeds.shape[1] < max_len:
            pad = torch.zeros(1, max_len - neg_embeds.shape[1], neg_embeds.shape[2],
                              device=neg_embeds.device, dtype=neg_embeds.dtype)
            neg_embeds = torch.cat([neg_embeds, pad], dim=1)

        image = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=neg_embeds,
            negative_pooled_prompt_embeds=neg_pooled,
            ip_adapter_image_embeds=[ip_embeds],
            width=config.generation.width,
            height=config.generation.height,
            num_inference_steps=config.generation.steps,
            guidance_scale=config.generation.guidance_scale,
            generator=generator,
        ).images[0]

        image.save(image_path)
        generated_paths.append(image_path)

        # Save metadata
        meta = {
            "prompt_id": prompt_id,
            "prompt": entry["prompt"],
            "negative_prompt": entry["negative_prompt"],
            "seed": config.generation.seed,
            "steps": config.generation.steps,
            "guidance_scale": config.generation.guidance_scale,
            "model": config.generation.model,
            "mode": "faceid",
            "identity_id": entry.get("identity_id"),
            "identity_ref": entry.get("identity_ref_path"),
            "identity_meta": entry.get("identity_meta", {}),
            "faceid_scale": config.faceid.scale,
            "slots": entry.get("slots", {}),
        }
        (meta_dir / f"{prompt_id}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2)
        )

        completed_ids.add(prompt_id)
        progress_file.write_text(json.dumps({
            "stage": "generate_faceid",
            "total": len(prompts),
            "completed": len(completed_ids),
            "completed_ids": sorted(completed_ids),
        }, indent=2))

    return generated_paths
