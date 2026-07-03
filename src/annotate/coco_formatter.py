"""Convert merged detections + masks into COCO JSON format."""

import json
from datetime import datetime
from pathlib import Path

from .merger import CATEGORIES, map_label
from .segmentor import mask_to_rle


def build_coco_dataset(
    image_annotations: list[dict],
    output_path: Path,
    dataset_description: str = "Synthetic Police Bodycam Dataset",
):
    """Build and save COCO-format annotation JSON.

    Args:
        image_annotations: List of dicts, each with:
            - image_id (int)
            - file_name (str)
            - width, height (int)
            - prompt_id (str)
            - detections: list of dicts with 'detection', 'mask', 'mask_score'
            - face_embeddings: list of dicts with 'bbox', 'embedding_file', 'quality_score'
        output_path: Where to save coco.json.
    """
    coco = {
        "info": {
            "description": dataset_description,
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
    }

    ann_id = 1
    for img_ann in image_annotations:
        # Image entry
        img_entry = {
            "id": img_ann["image_id"],
            "file_name": img_ann["file_name"],
            "width": img_ann["width"],
            "height": img_ann["height"],
            "prompt_id": img_ann.get("prompt_id", ""),
        }
        # Scene-sequential metadata (for video recognition training)
        if img_ann.get("scene_id"):
            img_entry["scene_id"] = img_ann["scene_id"]
            img_entry["frame_index"] = img_ann.get("frame_index", 0)
        coco["images"].append(img_entry)

        # Detection annotations
        for seg_result in img_ann.get("detections", []):
            det = seg_result["detection"]
            mask = seg_result.get("mask")
            cat_id = map_label(det.label)
            if cat_id is None:
                continue

            x1, y1, x2, y2 = det.bbox
            w = x2 - x1
            h = y2 - y1

            ann = {
                "id": ann_id,
                "image_id": img_ann["image_id"],
                "category_id": cat_id,
                "bbox": [x1, y1, w, h],  # COCO format: [x, y, w, h]
                "area": w * h,
                "iscrowd": 0,
                "score": det.confidence,
            }

            if mask is not None:
                ann["segmentation"] = mask_to_rle(mask)

            # Check if this detection has a face embedding
            face_emb = _match_face_embedding(det.bbox, img_ann.get("face_embeddings", []))
            ann["face_embedding_file"] = face_emb

            coco["annotations"].append(ann)
            ann_id += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    print(f"COCO annotations saved: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    return coco


def _match_face_embedding(det_bbox: list[float], face_embeddings: list[dict]) -> str | None:
    """Match a detection bbox to a face embedding by IoU."""
    if not face_embeddings:
        return None

    best_iou = 0
    best_emb = None
    for fe in face_embeddings:
        fb = fe["bbox"]
        # Simple IoU
        x1 = max(det_bbox[0], fb[0])
        y1 = max(det_bbox[1], fb[1])
        x2 = min(det_bbox[2], fb[2])
        y2 = min(det_bbox[3], fb[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
        area_b = (fb[2] - fb[0]) * (fb[3] - fb[1])
        union = area_a + area_b - inter
        iou = inter / union if union > 0 else 0
        if iou > best_iou:
            best_iou = iou
            best_emb = fe.get("embedding_file")

    return best_emb if best_iou > 0.3 else None
