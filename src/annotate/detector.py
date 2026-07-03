"""Grounding DINO open-vocabulary object detection (transformers API)."""

from dataclasses import dataclass

import torch
from PIL import Image


@dataclass
class Detection:
    bbox: list[float]  # [x1, y1, x2, y2]
    label: str
    confidence: float


# Detection text prompts per scenario
DETECT_CLASSES_COMMON = [
    "person", "face", "car", "motorcycle", "bicycle", "bus", "truck",
]

DETECT_CLASSES_POI = [
    "knife", "gun", "suspicious bag", "backpack",
]

DETECT_CLASSES_MISSING = [
    "child", "elderly person", "walking cane", "wheelchair",
]

DETECT_CLASSES_DANGER = [
    "knife", "gun", "bat", "metal pipe", "bottle", "crowbar",
    "hammer", "chain", "abandoned bag", "suspicious package",
    "masked person",
]


def get_detect_text(scenario: int) -> str:
    """Build Grounding DINO text prompt for detection."""
    classes = DETECT_CLASSES_COMMON.copy()
    if scenario == 1:
        classes.extend(DETECT_CLASSES_POI)
    elif scenario == 2:
        classes.extend(DETECT_CLASSES_MISSING)
    elif scenario == 3:
        classes.extend(DETECT_CLASSES_DANGER)
    return " . ".join(classes) + " ."


def detect_objects(
    model, processor, image: Image.Image, scenario: int,
    box_threshold: float = 0.3, text_threshold: float = 0.25,
) -> list[Detection]:
    """Run Grounding DINO detection on a single image using transformers API."""
    text_prompt = get_detect_text(scenario)

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],  # (height, width)
    )[0]

    detections = []
    for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
        x1, y1, x2, y2 = box.tolist()
        detections.append(Detection(
            bbox=[x1, y1, x2, y2],
            label=label.strip().lower(),
            confidence=float(score),
        ))

    return detections
