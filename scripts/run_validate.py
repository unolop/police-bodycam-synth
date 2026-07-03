#!/usr/bin/env python3
"""CLI: Run quality validation on generated + annotated dataset."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.prompts.prompt_store import load_prompts


def main():
    parser = argparse.ArgumentParser(description="Validate dataset quality")
    parser.add_argument("--config", default="config/default.yaml", nargs="+")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    config = load_config(*args.config)
    run_dir = Path(args.run_dir)

    image_dir = run_dir / "images"
    ann_dir = run_dir / "annotations"
    face_emb_dir = ann_dir / "face_embeddings"
    coco_path = ann_dir / "coco.json"

    image_paths = sorted(image_dir.glob("*.png"))
    prompts = load_prompts(run_dir / "prompts.jsonl")
    prompt_map = {p["prompt_id"]: p["prompt"] for p in prompts}

    # Match images to prompts
    matched_paths = []
    matched_prompts = []
    for p in image_paths:
        pid = p.stem
        if pid in prompt_map:
            matched_paths.append(p)
            matched_prompts.append(prompt_map[pid])

    print(f"Validating {len(matched_paths)} images...")

    # CLIP scores
    from src.generate.vram import model_scope
    from src.validate.clip_score import compute_clip_scores
    from src.validate.face_quality import compute_face_metrics
    from src.validate.fid_kid import compute_fid_kid
    from src.validate.report import generate_report

    with model_scope("validate", config) as models:
        clip_results = compute_clip_scores(
            models["clip_model"], models["clip_preprocess"], models["clip_tokenizer"],
            matched_paths, matched_prompts,
        )

    # Face metrics
    face_metrics = compute_face_metrics(face_emb_dir)

    # FID/KID
    real_path = config.validation.fid_real_path
    fid_kid = compute_fid_kid(image_dir, real_path)

    # Report
    report = generate_report(
        clip_results, face_metrics, fid_kid, coco_path,
        run_dir / "validation" / "report.json",
        min_clip_score=config.validation.min_clip_score,
    )


if __name__ == "__main__":
    main()
