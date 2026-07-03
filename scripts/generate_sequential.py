#!/home/david/miniconda3/envs/police/bin/python3
"""Sequential face→scenario generation pipeline.

Two-phase approach:
  Phase 1 — Face portraits: Generate bodycam-quality Asian face images per identity
            using the face ID dataset as reference (IP-Adapter FaceID).
  Phase 2 — Scenario actions: For each identity, generate action sets across
            scenarios (POI, missing person, weapons/threat).

Each phase runs in batches with face detection checkpoints between batches.
If detection rate drops below threshold, the batch is flagged for review.

Usage:
    # Phase 1 only (face portraits)
    python scripts/generate_sequential.py --phase 1

    # Phase 2 only (scenario actions, requires phase 1 output)
    python scripts/generate_sequential.py --phase 2

    # Both phases sequentially
    python scripts/generate_sequential.py --phase both

    # Dry run (generate prompts only, no images)
    python scripts/generate_sequential.py --phase both --dry-run
"""

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompts import templates as T
from src.generate.faceid import discover_face_dataset, load_face_labels

# ── Configuration ─────────────────────────────────────────────────
FACE_DATASET_DIR = "data/face id dataset"

# Generation defaults
MODEL = "SG161222/RealVisXL_V5.0"
STEPS = 30
GUIDANCE = 7.5
WIDTH, HEIGHT = 1024, 768  # landscape for bodycam aspect ratio
FACEID_SCALE = 0.3

# Batch sizes
FACE_BATCH_SIZE = 10        # identities per batch in phase 1
SCENARIO_BATCH_SIZE = 20    # images per batch in phase 2

# Quality gate
MIN_FACE_DETECTION_RATE = 0.80  # per-batch minimum

# Output
OUTPUT_DIR = Path("output/sequential_gen")


# ── Phase 1: Face portrait prompts ───────────────────────────────

# Bodycam-quality face portrait prompts (intentionally low quality)
FACE_PORTRAIT_PROMPTS = [
    (
        "{style}. An Asian {gender} facing the camera about 1.5 meters ahead, "
        "standing still on a {location}. The person's face and upper body are "
        "clearly visible, front view. {time}, {weather}. {lighting}"
    ),
    (
        "{style}. An Asian {gender} looking directly at the camera about 2 meters "
        "ahead, {action}. Face clearly visible, frontal view. "
        "{location} at {time}. {lighting}"
    ),
    (
        "{style}. An Asian {gender} turning toward the camera about 1 meter ahead, "
        "three-quarter face view. {location}, {time}, {weather}. {lighting}"
    ),
]

FACE_ACTIONS = [
    "standing with hands at sides",
    "looking up from a phone",
    "talking and gesturing",
    "standing with arms crossed",
    "walking toward the camera",
]

FACE_NEGATIVE = (
    "cartoon, anime, illustration, painting, drawing, sketch, "
    "watermark, deformed, disfigured, bad anatomy, extra limbs, "
    "3d render, cgi, digital art, concept art, "
    "studio lighting, professional photography, DSLR, bokeh, shallow depth of field, "
    "portrait studio, white background, clean background, "
    "high quality, sharp, 4k, 8k, ultra HD, masterpiece, "
    "cinematic, movie still, color grading, dramatic lighting, "
    "bird's-eye view, drone shot, surveillance camera, CCTV, "
    "back of head, person facing away, person from behind, rear view, "
    "caucasian, european, western"
)


# ── Phase 2: Scenario action prompts ─────────────────────────────

SCENARIO_ACTION_SETS = {
    "s1_poi_walking": {
        "scenario": 1,
        "label": "POI walking toward officer",
        "prompts": [
            (
                "{style}. An Asian {desc} facing the camera about 2 meters ahead, "
                "walking toward the camera with a neutral expression. "
                "Face and upper body clearly visible, front view. "
                "{location} at {time}, {weather}. {lighting}"
            ),
            (
                "{style}. An Asian {desc} facing the camera about 3 meters ahead, "
                "stopping and looking directly at the camera. "
                "Face clearly visible, frontal view. "
                "{location} at {time}, {weather}. {lighting}"
            ),
        ],
    },
    "s1_poi_standing": {
        "scenario": 1,
        "label": "POI standing/loitering",
        "prompts": [
            (
                "{style}. An Asian {desc} facing the camera about 1.5 meters ahead, "
                "standing near a wall with hands in pockets. "
                "Face and upper body clearly visible, front view. "
                "{location} at {time}, {weather}. {lighting}"
            ),
            (
                "{style}. An Asian {desc} facing the camera about 2 meters ahead, "
                "leaning against a railing and looking at the camera. "
                "Face clearly visible. "
                "{location} at {time}, {weather}. {lighting}"
            ),
        ],
    },
    "s2_missing_child": {
        "scenario": 2,
        "label": "Missing child scenario",
        "prompts": [
            (
                "{style}. A small Asian child about 3 meters ahead, "
                "standing alone and looking toward the camera with a lost expression. "
                "The child's face is clearly visible from the front. "
                "{location} at {time}, {weather}. No one else is nearby. {lighting}"
            ),
            (
                "{style}. A small Asian child about 4 meters ahead, "
                "sitting on the ground facing the camera and crying. "
                "The child's face is visible. "
                "{location} at {time}, {weather}. {lighting}"
            ),
        ],
    },
    "s2_missing_elderly": {
        "scenario": 2,
        "label": "Missing elderly person",
        "prompts": [
            (
                "{style}. An elderly Asian {gender} about 3 meters ahead, "
                "standing and looking confused, facing the camera. "
                "Wearing simple clothes, face clearly visible, front view. "
                "{location} at {time}, {weather}. {lighting}"
            ),
            (
                "{style}. An elderly Asian {gender} about 5 meters ahead, "
                "wandering aimlessly and looking toward the camera. "
                "Face visible, stooped posture. "
                "{location} at {time}, {weather}. {lighting}"
            ),
        ],
    },
    "s1_poi_weapon_knife": {
        "scenario": 1,
        "label": "POI holding knife",
        "prompts": [
            (
                "{style}. An Asian {desc} facing the camera about 3 meters ahead, "
                "standing in an aggressive stance. A large kitchen knife is visible "
                "in their right hand. Face and hands clearly visible, front view. "
                "{location} at {time}, {weather}. {lighting}"
            ),
            (
                "{style}. An Asian {desc} facing the camera about 2 meters ahead, "
                "holding a knife at their side with their right hand. "
                "Face clearly visible, tense expression, frontal view. "
                "{location} at {time}, {weather}. "
                "A few bystanders are backing away. {lighting}"
            ),
        ],
    },
    "s1_poi_weapon_bat": {
        "scenario": 1,
        "label": "POI holding baseball bat",
        "prompts": [
            (
                "{style}. An Asian {desc} facing the camera about 3 meters ahead, "
                "gripping a metal baseball bat with both hands. "
                "Face and upper body clearly visible, front view. "
                "{location} at {time}, {weather}. {lighting}"
            ),
            (
                "{style}. An Asian {desc} facing the camera about 2.5 meters ahead, "
                "holding a baseball bat over their shoulder. "
                "Face clearly visible, confrontational stance. "
                "{location} at {time}, {weather}. {lighting}"
            ),
        ],
    },
    "s1_poi_weapon_bottle": {
        "scenario": 1,
        "label": "POI holding broken bottle",
        "prompts": [
            (
                "{style}. An Asian {desc} facing the camera about 2 meters ahead, "
                "holding a broken glass bottle in their right hand. "
                "Face and hands clearly visible, front view, aggressive stance. "
                "{location} at {time}, {weather}. {lighting}"
            ),
        ],
    },
    "s1_poi_weapon_pipe": {
        "scenario": 1,
        "label": "POI holding metal pipe",
        "prompts": [
            (
                "{style}. An Asian {desc} facing the camera about 3 meters ahead, "
                "gripping a long metal pipe in both hands. "
                "Face clearly visible, threatening posture, front view. "
                "{location} at {time}, {weather}. {lighting}"
            ),
        ],
    },
    "s1_poi_abandoned_bag": {
        "scenario": 1,
        "label": "Suspicious abandoned bag",
        "prompts": [
            (
                "{style}. A large black duffel bag sitting alone on the ground "
                "in the center of the frame. An Asian {desc} is standing about "
                "4 meters ahead, facing the camera, looking around nervously. "
                "Face clearly visible, front view. "
                "{location} at {time}, {weather}. {lighting}"
            ),
        ],
    },
    "s1_poi_confrontation": {
        "scenario": 1,
        "label": "POI confrontation/aggressive",
        "prompts": [
            (
                "{style}. An Asian {desc} facing the camera about 1.5 meters ahead, "
                "shouting and gesturing aggressively with raised hands. "
                "Face clearly visible, angry expression, frontal view. "
                "{location} at {time}, {weather}. {lighting}"
            ),
            (
                "{style}. An Asian {desc} facing the camera about 2 meters ahead, "
                "pushing forward with an aggressive stance. "
                "Face clearly visible, confrontational expression. "
                "{location} at {time}, {weather}. "
                "Several pedestrians watching from a distance. {lighting}"
            ),
        ],
    },
}

# Subject descriptions for randomization
SUBJECT_DESCS = [
    "man in a dark hoodie and jeans",
    "man wearing a black jacket",
    "young man in a baseball cap and tracksuit",
    "man in a gray coat",
    "man wearing sunglasses and a beanie",
    "woman in a dark parka",
    "person in a black mask and dark clothing",
    "man with a shaved head wearing a leather jacket",
    "person in dark streetwear with a backpack",
    "man in work overalls",
]


def _build_face_prompts(
    subjects: dict[str, list[Path]],
    labels: dict[str, dict],
    seed: int = 42,
) -> list[dict]:
    """Build phase 1 prompts: one portrait set per identity."""
    rng = random.Random(seed)
    prompts = []

    for subj_id in sorted(subjects.keys()):
        lbl = labels.get(subj_id, {})
        gender = lbl.get("gender", "person")
        if gender == "female":
            gender_word = "woman"
        elif gender == "male":
            gender_word = "man"
        else:
            gender_word = "person"

        # Pick best reference image (normal, close distance, indoor direct)
        ref_images = subjects[subj_id]
        # Prefer outdoor direct normal for better face visibility
        outdoor_direct = [p for p in ref_images if "outdoor_direct" in p.stem]
        indoor_normal = [p for p in ref_images if "indoor_normal" in p.stem]
        ref = (outdoor_direct or indoor_normal or ref_images)[0]

        # Generate 3 portraits per identity (different locations/times)
        for var_idx in range(3):
            template = rng.choice(FACE_PORTRAIT_PROMPTS)
            prompt_text = template.format(
                style=T.STYLE_PREFIX,
                gender=gender_word,
                location=rng.choice(T.LOCATIONS),
                time=rng.choice(T.TIME_OF_DAY),
                weather=rng.choice(T.WEATHER),
                lighting=rng.choice(T.LIGHTING_DETAILS),
                action=rng.choice(FACE_ACTIONS),
            )
            prompt_text = prompt_text.replace(". .", ".").replace("  ", " ").strip()

            prompts.append({
                "prompt_id": f"face_{subj_id}_v{var_idx:02d}",
                "phase": 1,
                "prompt": prompt_text,
                "negative_prompt": FACE_NEGATIVE,
                "identity_id": subj_id,
                "identity_ref_path": str(ref),
                "identity_meta": lbl,
                "seed": rng.randint(0, 2**31),
            })

    return prompts


def _build_scenario_prompts(
    subjects: dict[str, list[Path]],
    labels: dict[str, dict],
    seed: int = 42,
) -> list[dict]:
    """Build phase 2 prompts: scenario action sets per identity."""
    rng = random.Random(seed)
    prompts = []

    subject_ids = sorted(subjects.keys())

    for action_key, action_set in SCENARIO_ACTION_SETS.items():
        scenario = action_set["scenario"]

        for subj_id in subject_ids:
            lbl = labels.get(subj_id, {})
            gender = lbl.get("gender", "person")
            gender_word = "woman" if gender == "female" else "man" if gender == "male" else "person"

            # Pick reference
            ref_images = subjects[subj_id]
            outdoor_direct = [p for p in ref_images if "outdoor_direct" in p.stem]
            indoor_normal = [p for p in ref_images if "indoor_normal" in p.stem]
            ref = (outdoor_direct or indoor_normal or ref_images)[0]

            desc = rng.choice(SUBJECT_DESCS)

            for tmpl_idx, template in enumerate(action_set["prompts"]):
                prompt_text = template.format(
                    style=T.STYLE_PREFIX,
                    gender=gender_word,
                    desc=desc,
                    location=rng.choice(T.LOCATIONS),
                    time=rng.choice(T.TIME_OF_DAY),
                    weather=rng.choice(T.WEATHER),
                    lighting=rng.choice(T.LIGHTING_DETAILS),
                )
                prompt_text = prompt_text.replace(". .", ".").replace("  ", " ").strip()

                prompts.append({
                    "prompt_id": f"{action_key}_{subj_id}_t{tmpl_idx:02d}",
                    "phase": 2,
                    "scenario": scenario,
                    "action_set": action_key,
                    "action_label": action_set["label"],
                    "prompt": prompt_text,
                    "negative_prompt": FACE_NEGATIVE,
                    "identity_id": subj_id,
                    "identity_ref_path": str(ref),
                    "identity_meta": lbl,
                    "seed": rng.randint(0, 2**31),
                })

    return prompts


def _run_face_check(image_dir: Path, batch_label: str) -> dict:
    """Quick face detection check on a batch of images."""
    try:
        from insightface.app import FaceAnalysis
        import cv2
    except ImportError:
        print(f"  [SKIP] InsightFace not available, skipping face check")
        return {"skipped": True}

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    images = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    if not images:
        return {"total": 0, "detected": 0, "rate": 0}

    detected = 0
    failed = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        faces = face_app.get(img)
        if faces:
            detected += 1
        else:
            failed.append(img_path.name)

    rate = detected / len(images) if images else 0
    status = "PASS" if rate >= MIN_FACE_DETECTION_RATE else "FAIL"

    print(f"  [{status}] {batch_label}: {detected}/{len(images)} faces ({rate*100:.1f}%)")
    if failed and len(failed) <= 5:
        print(f"         Failed: {', '.join(failed)}")
    elif failed:
        print(f"         Failed: {len(failed)} images")

    del face_app
    gc.collect()

    return {
        "batch": batch_label,
        "total": len(images),
        "detected": detected,
        "rate": rate,
        "status": status,
        "failed_images": failed,
    }


def _load_pipeline():
    """Load SDXL pipeline and compel encoder once."""
    import torch
    from diffusers import StableDiffusionXLPipeline
    from compel import Compel, ReturnedEmbeddingsType

    print("  Loading SDXL pipeline...")
    dtype = torch.float16
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL, torch_dtype=dtype, use_safetensors=True, variant="fp16",
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
        truncate_long_prompts=False,
    )
    print("  Pipeline ready")
    return pipe, compel


def _generate_images(prompts: list[dict], output_dir: Path, pipe_and_compel=None, dry_run: bool = False):
    """Generate images using SDXL. If pipe_and_compel provided, reuses it."""
    if dry_run:
        prompt_dir = output_dir / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for p in prompts:
            (prompt_dir / f"{p['prompt_id']}.json").write_text(
                json.dumps(p, ensure_ascii=False, indent=2)
            )
        print(f"  [DRY RUN] Saved {len(prompts)} prompts to {prompt_dir}")
        return

    import torch

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    existing = {f.stem for f in image_dir.glob("*.png")}
    pending = [p for p in prompts if p["prompt_id"] not in existing]

    if not pending:
        print(f"  All {len(prompts)} images already generated.")
        return

    print(f"  Generating {len(pending)} images ({len(existing)} already done)...")

    # Load pipeline if not provided
    own_pipe = False
    if pipe_and_compel is None:
        pipe_and_compel = _load_pipeline()
        own_pipe = True
    pipe, compel = pipe_and_compel

    t0 = time.time()
    for i, entry in enumerate(pending):
        pid = entry["prompt_id"]

        generator = torch.Generator(device="cuda").manual_seed(entry.get("seed", 42))

        prompt_embeds, pooled = compel(entry["prompt"])
        neg_embeds, neg_pooled = compel(entry["negative_prompt"])

        max_len = max(prompt_embeds.shape[1], neg_embeds.shape[1])
        if prompt_embeds.shape[1] < max_len:
            prompt_embeds = torch.cat([prompt_embeds,
                torch.zeros(1, max_len - prompt_embeds.shape[1], prompt_embeds.shape[2],
                            device=prompt_embeds.device, dtype=prompt_embeds.dtype)], dim=1)
        if neg_embeds.shape[1] < max_len:
            neg_embeds = torch.cat([neg_embeds,
                torch.zeros(1, max_len - neg_embeds.shape[1], neg_embeds.shape[2],
                            device=neg_embeds.device, dtype=neg_embeds.dtype)], dim=1)

        image = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=neg_embeds,
            negative_pooled_prompt_embeds=neg_pooled,
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=generator,
        ).images[0]
        image.save(image_dir / f"{pid}.png")

        meta = {
            "prompt_id": pid,
            "prompt": entry["prompt"],
            "negative_prompt": entry["negative_prompt"],
            "seed": entry.get("seed", 42),
            "identity_id": entry.get("identity_id"),
            "identity_ref": entry.get("identity_ref_path"),
            "action_set": entry.get("action_set"),
            "action_label": entry.get("action_label"),
            "scenario": entry.get("scenario"),
            "phase": entry.get("phase"),
            "model": MODEL,
            "steps": STEPS,
            "guidance_scale": GUIDANCE,
        }
        (meta_dir / f"{pid}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2)
        )

        elapsed = time.time() - t0
        eta = (elapsed / (i + 1)) * (len(pending) - i - 1)
        print(f"  [{i+1}/{len(pending)}] {pid} ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    if own_pipe:
        del pipe, compel
        gc.collect()
        torch.cuda.empty_cache()


def run_phase1(subjects, labels, dry_run=False):
    """Phase 1: Generate face portraits per identity."""
    print("\n" + "=" * 60)
    print("PHASE 1: Face Portrait Generation")
    print(f"  Identities: {len(subjects)}")
    print(f"  Portraits per identity: 3")
    print(f"  Total images: {len(subjects) * 3}")
    print("=" * 60)

    prompts = _build_face_prompts(subjects, labels)
    phase_dir = OUTPUT_DIR / "phase1_faces"

    prompt_save = phase_dir / "all_prompts.jsonl"
    prompt_save.parent.mkdir(parents=True, exist_ok=True)
    with open(prompt_save, "w") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  Saved {len(prompts)} prompts to {prompt_save}")

    # Load pipeline once for all batches
    pipe_and_compel = None if dry_run else _load_pipeline()

    subject_ids = sorted(subjects.keys())
    batch_results = []

    for batch_start in range(0, len(subject_ids), FACE_BATCH_SIZE):
        batch_end = min(batch_start + FACE_BATCH_SIZE, len(subject_ids))
        batch_ids = set(subject_ids[batch_start:batch_end])
        batch_prompts = [p for p in prompts if p["identity_id"] in batch_ids]
        batch_num = batch_start // FACE_BATCH_SIZE + 1

        print(f"\n--- Batch {batch_num} ({len(batch_ids)} identities, "
              f"{len(batch_prompts)} images) ---")

        _generate_images(batch_prompts, phase_dir, pipe_and_compel=pipe_and_compel, dry_run=dry_run)

    # Unload pipeline before face check (free VRAM)
    if pipe_and_compel is not None:
        import torch
        del pipe_and_compel
        gc.collect()
        torch.cuda.empty_cache()

    # Face detection checkpoint on all generated images
    if not dry_run:
        check = _run_face_check(phase_dir / "images", "Phase 1 (all)")
        batch_results.append(check)

    results_path = phase_dir / "batch_results.json"
    with open(results_path, "w") as f:
        json.dump(batch_results, f, indent=2)

    return batch_results


def run_phase2(subjects, labels, dry_run=False):
    """Phase 2: Generate scenario action images per identity."""
    print("\n" + "=" * 60)
    print("PHASE 2: Scenario Action Generation")
    print(f"  Identities: {len(subjects)}")
    print(f"  Action sets: {len(SCENARIO_ACTION_SETS)}")

    prompts = _build_scenario_prompts(subjects, labels)
    total_per_action = {}
    for p in prompts:
        key = p["action_set"]
        total_per_action[key] = total_per_action.get(key, 0) + 1

    print(f"  Total images: {len(prompts)}")
    for key, count in sorted(total_per_action.items()):
        label = SCENARIO_ACTION_SETS[key]["label"]
        print(f"    {key}: {count} ({label})")
    print("=" * 60)

    phase_dir = OUTPUT_DIR / "phase2_scenarios"
    prompt_save = phase_dir / "all_prompts.jsonl"
    prompt_save.parent.mkdir(parents=True, exist_ok=True)
    with open(prompt_save, "w") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Load pipeline once for all action sets
    pipe_and_compel = None if dry_run else _load_pipeline()

    batch_results = []
    for action_key in sorted(SCENARIO_ACTION_SETS.keys()):
        action_prompts = [p for p in prompts if p["action_set"] == action_key]
        action_dir = phase_dir / action_key
        label = SCENARIO_ACTION_SETS[action_key]["label"]

        print(f"\n--- {action_key}: {label} ({len(action_prompts)} images) ---")
        _generate_images(action_prompts, action_dir, pipe_and_compel=pipe_and_compel, dry_run=dry_run)

    # Unload pipeline before face checks
    if pipe_and_compel is not None:
        import torch
        del pipe_and_compel
        gc.collect()
        torch.cuda.empty_cache()

    # Face detection checkpoints per action set
    if not dry_run:
        for action_key in sorted(SCENARIO_ACTION_SETS.keys()):
            action_dir = phase_dir / action_key
            check = _run_face_check(action_dir / "images", f"Phase 2: {action_key}")
            batch_results.append(check)

    results_path = phase_dir / "batch_results.json"
    with open(results_path, "w") as f:
        json.dump(batch_results, f, indent=2)

    return batch_results


def print_summary(p1_results, p2_results):
    """Print final summary of all batches."""
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)

    if p1_results:
        print("\nPhase 1 — Face Portraits:")
        for r in p1_results:
            if isinstance(r, dict) and not r.get("skipped"):
                print(f"  [{r.get('status', '?')}] {r.get('batch', '?')}: "
                      f"{r.get('detected', 0)}/{r.get('total', 0)} "
                      f"({r.get('rate', 0)*100:.1f}%)")

    if p2_results:
        print("\nPhase 2 — Scenario Actions:")
        for r in p2_results:
            if isinstance(r, dict) and not r.get("skipped"):
                print(f"  [{r.get('status', '?')}] {r.get('batch', '?')}: "
                      f"{r.get('detected', 0)}/{r.get('total', 0)} "
                      f"({r.get('rate', 0)*100:.1f}%)")

    # Overall stats
    all_results = (p1_results or []) + (p2_results or [])
    valid = [r for r in all_results if isinstance(r, dict) and not r.get("skipped")]
    if valid:
        total_imgs = sum(r.get("total", 0) for r in valid)
        total_det = sum(r.get("detected", 0) for r in valid)
        overall_rate = total_det / total_imgs if total_imgs > 0 else 0
        failed_batches = [r for r in valid if r.get("status") == "FAIL"]
        print(f"\nOverall: {total_det}/{total_imgs} faces detected ({overall_rate*100:.1f}%)")
        if failed_batches:
            print(f"  WARNING: {len(failed_batches)} batches below {MIN_FACE_DETECTION_RATE*100:.0f}% threshold")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential face→scenario generation")
    parser.add_argument("--phase", choices=["1", "2", "both"], default="both",
                        help="Which phase to run (default: both)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate prompts only, no images")
    parser.add_argument("--face-dataset", type=str, default=FACE_DATASET_DIR,
                        help="Path to face ID dataset")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    OUTPUT_DIR = Path(args.output_dir)

    print("=" * 60)
    print("Sequential Face → Scenario Generation Pipeline")
    print(f"  Face dataset: {args.face_dataset}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Phase: {args.phase}")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 60)

    # Discover face dataset
    print("\nLoading face ID dataset...")
    subjects = discover_face_dataset(args.face_dataset)
    labels = load_face_labels(args.face_dataset)
    print(f"  Found {len(subjects)} subjects, {len(labels)} with labels")

    t_start = time.time()
    p1_results = None
    p2_results = None

    if args.phase in ("1", "both"):
        p1_results = run_phase1(subjects, labels, dry_run=args.dry_run)

    if args.phase in ("2", "both"):
        p2_results = run_phase2(subjects, labels, dry_run=args.dry_run)

    print_summary(p1_results, p2_results)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
