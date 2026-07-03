"""Face detection rate and embedding quality metrics.

For the HE face matching pipeline, what matters is:
1. Face detection rate — can we reliably extract faces from synthetic bodycam images?
2. Embedding quality — are the extracted 512-dim vectors discriminative enough
   for the encrypted similarity search to produce meaningful matches?
3. Embedding diversity — do synthetic faces cover a broad identity space,
   or do they collapse to similar-looking faces (mode collapse)?
"""

import numpy as np
from pathlib import Path


def compute_face_metrics(face_embedding_dir: Path) -> dict:
    """Compute face quality metrics from saved embeddings."""
    emb_files = list(face_embedding_dir.glob("*.npy"))

    if not emb_files:
        return {
            "total_faces": 0,
            "mean_embedding_norm": 0,
            "std_embedding_norm": 0,
        }

    norms = []
    for f in emb_files:
        emb = np.load(f)
        norms.append(np.linalg.norm(emb))

    return {
        "total_faces": len(emb_files),
        "mean_embedding_norm": float(np.mean(norms)),
        "std_embedding_norm": float(np.std(norms)),
        "min_embedding_norm": float(np.min(norms)),
        "max_embedding_norm": float(np.max(norms)),
    }


def compute_face_detection_and_embeddings(
    image_dir: Path,
    output_dir: Path | None = None,
    det_size: int = 640,
) -> dict:
    """Run face detection + embedding extraction on a directory of images.

    Returns detection rate, embedding statistics, and pairwise diversity metrics.
    Optionally saves embeddings to output_dir for downstream use.
    """
    from insightface.app import FaceAnalysis
    import cv2

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(det_size, det_size))

    image_paths = sorted(
        list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    )
    if not image_paths:
        return {"error": "No images found", "total_images": 0}

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = []
    face_areas = []
    detection_results = []
    faces_detected = 0

    for img_path in image_paths:
        img_cv = cv2.imread(str(img_path))
        faces = face_app.get(img_cv)

        if faces:
            # Take the largest face
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            x1, y1, x2, y2 = [int(c) for c in face.bbox]
            area = (x2 - x1) * (y2 - y1)
            emb = face.normed_embedding  # 512-dim, L2-normalized

            embeddings.append(emb)
            face_areas.append(area)
            faces_detected += 1

            if output_dir:
                np.save(output_dir / f"{img_path.stem}.npy", emb)

            detection_results.append({
                "image": img_path.name,
                "face_found": True,
                "face_area": area,
                "num_faces": len(faces),
            })
        else:
            detection_results.append({
                "image": img_path.name,
                "face_found": False,
            })

    n = len(image_paths)
    detection_rate = faces_detected / n if n > 0 else 0

    result = {
        "total_images": n,
        "faces_detected": faces_detected,
        "detection_rate": detection_rate,
        "detection_rate_pct": f"{detection_rate * 100:.1f}%",
    }

    if face_areas:
        result["face_area_mean"] = float(np.mean(face_areas))
        result["face_area_std"] = float(np.std(face_areas))
        result["face_area_median"] = float(np.median(face_areas))

    # Embedding quality metrics
    if len(embeddings) >= 2:
        emb_matrix = np.stack(embeddings)  # (N, 512)
        norms = np.linalg.norm(emb_matrix, axis=1)

        # Pairwise cosine similarity (embeddings are already L2-normalized)
        cos_sim = emb_matrix @ emb_matrix.T
        # Extract upper triangle (exclude diagonal)
        triu_idx = np.triu_indices(len(embeddings), k=1)
        pairwise_sims = cos_sim[triu_idx]

        result["embedding_stats"] = {
            "num_embeddings": len(embeddings),
            "embedding_dim": emb_matrix.shape[1],
            "norm_mean": float(np.mean(norms)),
            "norm_std": float(np.std(norms)),
            # Pairwise cosine similarity distribution
            "cosine_sim_mean": float(np.mean(pairwise_sims)),
            "cosine_sim_std": float(np.std(pairwise_sims)),
            "cosine_sim_min": float(np.min(pairwise_sims)),
            "cosine_sim_max": float(np.max(pairwise_sims)),
            "cosine_sim_median": float(np.median(pairwise_sims)),
            # High similarity pairs (potential mode collapse)
            "pairs_above_0.7": int(np.sum(pairwise_sims > 0.7)),
            "pairs_above_0.5": int(np.sum(pairwise_sims > 0.5)),
            "total_pairs": len(pairwise_sims),
        }

    result["per_image"] = detection_results

    del face_app
    return result
