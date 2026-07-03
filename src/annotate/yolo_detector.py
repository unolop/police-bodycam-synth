"""YOLOv8 secondary detection for cross-validation."""

from pathlib import Path

import numpy as np
from PIL import Image

from .detector import Detection


def detect_yolo(model, image: Image.Image, conf: float = 0.25) -> list[Detection]:
    """Run YOLOv8 detection on a single image."""
    results = model(np.array(image), conf=conf, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            score = float(box.conf[0])
            detections.append(Detection(
                bbox=[x1, y1, x2, y2],
                label=label.lower(),
                confidence=score,
            ))

    return detections
