"""Merge Grounding DINO + YOLO detections into unified annotations."""

from .detector import Detection

# Unified category mapping
CATEGORIES = [
    {"id": 1, "name": "person", "supercategory": "human"},
    {"id": 2, "name": "face", "supercategory": "human"},
    {"id": 3, "name": "child", "supercategory": "human"},
    {"id": 4, "name": "elderly_person", "supercategory": "human"},
    {"id": 5, "name": "car", "supercategory": "vehicle"},
    {"id": 6, "name": "motorcycle", "supercategory": "vehicle"},
    {"id": 7, "name": "bicycle", "supercategory": "vehicle"},
    {"id": 8, "name": "bus", "supercategory": "vehicle"},
    {"id": 9, "name": "truck", "supercategory": "vehicle"},
    {"id": 10, "name": "knife", "supercategory": "dangerous_object"},
    {"id": 11, "name": "gun", "supercategory": "dangerous_object"},
    {"id": 12, "name": "suspicious_bag", "supercategory": "suspicious_object"},
    {"id": 13, "name": "backpack", "supercategory": "suspicious_object"},
    {"id": 14, "name": "walking_cane", "supercategory": "object"},
    {"id": 15, "name": "wheelchair", "supercategory": "object"},
]

# Map free-form labels to category IDs
_LABEL_MAP = {}
for cat in CATEGORIES:
    _LABEL_MAP[cat["name"]] = cat["id"]
    # Add common variations
_LABEL_MAP.update({
    "elderly person": 4, "elderly": 4, "old person": 4, "senior": 4,
    "suspicious bag": 12, "bag": 12,
    "walking cane": 14, "cane": 14,
    "motorbike": 6, "bike": 7,
})


def _compute_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def map_label(label: str) -> int | None:
    """Map a free-form label to a unified category ID."""
    label = label.strip().lower()
    if label in _LABEL_MAP:
        return _LABEL_MAP[label]
    # Fuzzy match: check if any category name is contained in the label
    for cat in CATEGORIES:
        if cat["name"] in label:
            return cat["id"]
    return None


def merge_detections(
    dino_dets: list[Detection],
    yolo_dets: list[Detection],
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Merge DINO (primary) and YOLO (secondary) detections.

    DINO detections are kept as-is.
    YOLO detections that don't overlap with any DINO detection (IoU < threshold)
    are added as new detections.
    """
    merged = list(dino_dets)

    for yolo_det in yolo_dets:
        # Check if this YOLO detection matches any DINO detection
        cat_id = map_label(yolo_det.label)
        if cat_id is None:
            continue  # Skip labels we don't care about

        matched = False
        for dino_det in dino_dets:
            if _compute_iou(yolo_det.bbox, dino_det.bbox) >= iou_threshold:
                matched = True
                break

        if not matched:
            merged.append(yolo_det)

    return merged
