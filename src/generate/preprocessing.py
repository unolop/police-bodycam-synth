"""Preprocessing for ControlNet conditioning: canny edges and openpose."""

import cv2
import numpy as np
from PIL import Image


def extract_canny(image: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    """Extract Canny edges from a PIL image."""
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    edges = cv2.Canny(gray, low, high)
    # Convert back to 3-channel for ControlNet
    edges_rgb = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges_rgb)


def extract_openpose(image: Image.Image) -> Image.Image:
    """Extract OpenPose skeleton from a PIL image using controlnet_aux."""
    from controlnet_aux import OpenposeDetector
    detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    pose = detector(image)
    return pose
