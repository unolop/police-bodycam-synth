"""ArcFace face embedding extraction."""

import numpy as np
from pathlib import Path
from PIL import Image


def extract_face_embeddings(
    face_app, image: Image.Image, image_id: str, output_dir: Path,
) -> list[dict]:
    """Detect faces and extract embeddings.

    Returns list of dicts with bbox, embedding path, quality score.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    img_array = np.array(image.convert("RGB"))
    # insightface expects BGR
    img_bgr = img_array[:, :, ::-1]

    faces = face_app.get(img_bgr)

    results = []
    for i, face in enumerate(faces):
        emb = face.embedding  # 512-dim
        bbox = face.bbox.tolist()  # [x1, y1, x2, y2]
        quality = float(face.det_score)

        emb_path = output_dir / f"{image_id}_face{i}.npy"
        np.save(emb_path, emb)

        results.append({
            "bbox": bbox,
            "embedding_file": str(emb_path),
            "quality_score": quality,
            "embedding_dim": len(emb),
        })

    return results
