#!/usr/bin/env python3
"""CLI: Annotate generated images with detections, masks, and face embeddings."""

import argparse
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Annotate generated images")
    parser.add_argument("--config", default="config/default.yaml", nargs="+")
    parser.add_argument("--run-dir", required=True, help="Path to run directory")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3], default=None)
    args = parser.parse_args()

    overrides = {}
    if args.scenario:
        overrides["scenario"] = args.scenario

    config = load_config(*args.config, overrides=overrides)
    run_dir = Path(args.run_dir)
    image_dir = run_dir / "images"
    ann_dir = run_dir / "annotations"
    face_emb_dir = ann_dir / "face_embeddings"
    ann_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        print(f"No images found in {image_dir}")
        sys.exit(1)

    print(f"Annotating {len(image_paths)} images...")

    # Load image metadata for scene info
    import json
    meta_dir = run_dir / "image_meta"
    image_meta = {}
    if meta_dir.exists():
        for meta_file in meta_dir.glob("*.json"):
            meta = json.loads(meta_file.read_text())
            image_meta[meta.get("prompt_id", meta_file.stem)] = meta

    from src.generate.vram import model_scope
    from src.annotate.detector import detect_objects
    from src.annotate.segmentor import segment_from_boxes
    from src.annotate.yolo_detector import detect_yolo
    from src.annotate.face_embedder import extract_face_embeddings
    from src.annotate.merger import merge_detections
    from src.annotate.coco_formatter import build_coco_dataset

    with model_scope("annotate", config) as models:
        all_image_annotations = []

        for img_idx, img_path in enumerate(tqdm(image_paths, desc="Annotating")):
            image = Image.open(img_path).convert("RGB")
            prompt_id = img_path.stem
            w, h = image.size

            # Grounding DINO detection
            dino_dets = detect_objects(
                models["dino"], models["dino_processor"], image, config.scenario,
                box_threshold=config.annotation.dino_box_threshold,
                text_threshold=config.annotation.dino_text_threshold,
            )

            # SAM2 segmentation from DINO boxes
            seg_results = segment_from_boxes(models["sam2"], image, dino_dets)

            # YOLO secondary detection
            yolo_dets = detect_yolo(models["yolo"], image, conf=config.annotation.yolo_conf)

            # Merge detections
            merged_dets = merge_detections(dino_dets, yolo_dets)

            # For merged dets that came from YOLO (no mask), create seg_results without mask
            seg_map = {id(d["detection"]): d for d in seg_results}
            final_seg = []
            for det in merged_dets:
                if id(det) in seg_map:
                    final_seg.append(seg_map[id(det)])
                else:
                    # Check if we have a matching DINO detection with mask
                    matched = False
                    for sr in seg_results:
                        if sr["detection"].bbox == det.bbox:
                            final_seg.append(sr)
                            matched = True
                            break
                    if not matched:
                        final_seg.append({
                            "detection": det,
                            "mask": None,
                            "mask_score": 0,
                        })

            # Face embeddings
            face_embs = extract_face_embeddings(
                models["face"], image, prompt_id, face_emb_dir,
            )

            img_ann = {
                "image_id": img_idx + 1,
                "file_name": img_path.name,
                "width": w,
                "height": h,
                "prompt_id": prompt_id,
                "detections": final_seg,
                "face_embeddings": face_embs,
            }
            # Attach scene metadata if available
            meta = image_meta.get(prompt_id, {})
            if "scene_id" in meta:
                img_ann["scene_id"] = meta["scene_id"]
                img_ann["frame_index"] = meta.get("frame_index", 0)
            all_image_annotations.append(img_ann)

    # Build COCO JSON
    coco_path = ann_dir / "coco.json"
    build_coco_dataset(all_image_annotations, coco_path)
    print(f"Annotations saved to {coco_path}")


if __name__ == "__main__":
    main()
