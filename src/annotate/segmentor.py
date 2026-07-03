"""SAM2 segmentation from bounding box prompts."""

import numpy as np
from PIL import Image

from .detector import Detection


def segment_from_boxes(
    predictor, image: Image.Image, detections: list[Detection],
) -> list[dict]:
    """Generate segmentation masks for each detection using SAM2.

    Returns list of dicts with 'detection' and 'mask' (binary numpy array).
    """
    img_array = np.array(image.convert("RGB"))
    predictor.set_image(img_array)

    results = []
    for det in detections:
        box = np.array(det.bbox)  # [x1, y1, x2, y2]
        masks, scores, _ = predictor.predict(
            box=box[None, :],  # (1, 4)
            multimask_output=False,
        )
        # masks shape: (1, H, W), take best mask
        mask = masks[0].astype(bool)
        results.append({
            "detection": det,
            "mask": mask,
            "mask_score": float(scores[0]),
        })

    return results


def mask_to_rle(mask: np.ndarray) -> dict:
    """Convert binary mask to COCO RLE format."""
    # Flatten in column-major (Fortran) order as COCO expects
    flat = mask.flatten(order="F")
    runs = []
    prev = 0
    count = 0
    for val in flat:
        if val != prev:
            runs.append(count)
            count = 1
            prev = val
        else:
            count += 1
    runs.append(count)

    # If mask starts with 1, prepend 0-length run
    if flat[0] == 1:
        runs.insert(0, 0)

    return {
        "counts": runs,
        "size": list(mask.shape),  # [H, W]
    }
