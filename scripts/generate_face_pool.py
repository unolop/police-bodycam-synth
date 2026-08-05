#!/usr/bin/env python3
"""Generate synthetic face identity pool using DeceiveD (StyleGAN2+APA).

Steps:
1. Run DeceiveD unconditional model → generate N face images
2. Extract ArcFace 512-dim embeddings from each
3. Measure quality: FID/KID against Celeb-K sample, embedding diversity
4. Select top-K most diverse identities for the bodycam pipeline

Usage:
    python scripts/generate_face_pool.py --num-faces 50 --num-select 20
    python scripts/generate_face_pool.py --num-faces 50 --num-select 20 --measure-only  # skip generation
"""

import argparse
import json
import sys
import glob
import os
from pathlib import Path

import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DECEIVED_ROOT = None  # resolved dynamically due to Korean path names
CELEB_K_SAMPLE = None

def find_deceived_paths():
    global DECEIVED_ROOT, CELEB_K_SAMPLE
    base = PROJECT_ROOT / "data" / "celeb-k" / "model_unzipped"
    matches = glob.glob(str(base / "**/DeceiveD"), recursive=True)
    if not matches:
        raise FileNotFoundError("DeceiveD source not found in model_unzipped/")
    DECEIVED_ROOT = Path(matches[0])

    sample_base = PROJECT_ROOT / "data" / "celeb-k" / "sample_unzipped"
    sample_matches = glob.glob(str(sample_base / "**/*.jpg"), recursive=True) + \
                     glob.glob(str(sample_base / "**/*.png"), recursive=True)
    if sample_matches:
        CELEB_K_SAMPLE = Path(sample_matches[0]).parent
    return DECEIVED_ROOT


# ---------- Step 1: Generate faces with DeceiveD ----------

def generate_faces(num_faces: int, output_dir: Path, truncation: float = 0.7, device: str = "cuda:1"):
    """Generate face images using DeceiveD unconditional checkpoint."""
    import torch
    # Add DeceiveD to path for dnnlib/legacy imports
    sys.path.insert(0, str(DECEIVED_ROOT))
    import dnnlib
    import legacy

    ckpt = DECEIVED_ROOT / "Checkpoint" / "1_Uncond" / "network-snapshot-000008.pkl"
    print(f"Loading DeceiveD from {ckpt.name}...")

    with dnnlib.util.open_url(str(ckpt)) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    label = torch.zeros([1, G.c_dim], device=device)

    print(f"Generating {num_faces} faces (trunc={truncation})...")
    for seed in range(num_faces):
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=truncation, noise_mode='const')
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        from PIL import Image
        Image.fromarray(img[0].cpu().numpy(), 'RGB').save(output_dir / f"face_{seed:04d}.png")

        if (seed + 1) % 10 == 0:
            print(f"  {seed + 1}/{num_faces} generated")

    print(f"Done: {num_faces} faces saved to {output_dir}")
    del G
    torch.cuda.empty_cache()


# ---------- Step 2: Extract ArcFace embeddings ----------

def extract_embeddings(image_dir: Path, output_dir: Path, device_id: int = 1) -> np.ndarray:
    """Extract ArcFace 512-dim embeddings from face images."""
    from insightface.app import FaceAnalysis
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider"],
    )
    face_app.prepare(ctx_id=device_id, det_size=(640, 640))

    image_paths = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    embeddings = []
    names = []

    print(f"Extracting embeddings from {len(image_paths)} images...")
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        faces = face_app.get(img)

        if faces:
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb = face.normed_embedding
            np.save(output_dir / f"{img_path.stem}.npy", emb)
            embeddings.append(emb)
            names.append(img_path.stem)
        else:
            print(f"  WARNING: no face detected in {img_path.name}")

    print(f"Extracted {len(embeddings)}/{len(image_paths)} embeddings")
    del face_app

    return np.stack(embeddings), names


# ---------- Step 3: Quality metrics ----------

def measure_quality(
    generated_dir: Path,
    embeddings: np.ndarray,
    names: list,
    real_dir: Path | None = None,
) -> dict:
    """Compute FID/KID and embedding diversity metrics."""
    metrics = {}

    # FID/KID (only meaningful with large diverse reference set like FFHQ)
    if real_dir and real_dir.exists():
        num_real = len(list(Path(real_dir).glob("*.png")) + list(Path(real_dir).glob("*.jpg")))
        if num_real >= 1000:
            try:
                from cleanfid import fid
                print(f"Computing FID/KID against {num_real} reference images...")
                metrics["fid"] = float(fid.compute_fid(str(generated_dir), str(real_dir)))
                metrics["kid"] = float(fid.compute_kid(str(generated_dir), str(real_dir)))
                print(f"  FID: {metrics['fid']:.2f}")
                print(f"  KID: {metrics['kid']:.4f}")
            except Exception as e:
                print(f"  FID/KID computation failed: {e}")
                metrics["fid"] = None
                metrics["kid"] = None
        else:
            print(f"Only {num_real} reference images — too few for reliable FID/KID (need 1000+), skipping")
            metrics["fid"] = None
            metrics["kid"] = None
    else:
        metrics["fid"] = None
        metrics["kid"] = None

    # Reference-free quality metrics
    # 1. Face detection rate — if DeceiveD produces non-face artifacts, detection will fail
    num_generated = len(list(generated_dir.glob("*.png")))
    face_det_rate = len(embeddings) / num_generated if num_generated > 0 else 0
    metrics["face_detection_rate"] = face_det_rate
    print(f"\nFace detection rate: {face_det_rate*100:.1f}% ({len(embeddings)}/{num_generated})")
    if face_det_rate < 0.9:
        print("  WARNING: low detection rate — model may be producing artifacts")

    # 2. Embedding norm consistency — real faces have ~uniform norms after L2 normalization
    norms = np.linalg.norm(embeddings, axis=1)
    metrics["embedding_norm_std"] = float(np.std(norms))

    # 3. Truncation comparison hint
    print(f"\nTip: run with --truncation 1.0 for more diversity (but lower quality)")
    print(f"     run with --truncation 0.5 for higher quality (but less diversity)")
    print(f"     compare face_pool_report.json between runs to find the sweet spot")

    # Embedding diversity
    if len(embeddings) >= 2:
        cos_sim = embeddings @ embeddings.T
        triu = np.triu_indices(len(embeddings), k=1)
        pairwise = cos_sim[triu]

        metrics["embedding_diversity"] = {
            "num_identities": len(embeddings),
            "cosine_sim_mean": float(np.mean(pairwise)),
            "cosine_sim_std": float(np.std(pairwise)),
            "cosine_sim_min": float(np.min(pairwise)),
            "cosine_sim_max": float(np.max(pairwise)),
            "cosine_sim_median": float(np.median(pairwise)),
            # Tail retention indicators
            "pairs_above_0.7": int(np.sum(pairwise > 0.7)),  # too similar (mode collapse)
            "pairs_below_0.1": int(np.sum(pairwise < 0.1)),  # very different (good diversity)
            "total_pairs": len(pairwise),
            # Spread — higher is better (config threshold: 0.3)
            "spread_std": float(np.std(pairwise)),
        }

        print(f"\nEmbedding diversity:")
        print(f"  Mean cosine sim: {np.mean(pairwise):.3f} (lower = more diverse)")
        print(f"  Std:  {np.std(pairwise):.3f} (higher = more spread, threshold: 0.3)")
        print(f"  Min:  {np.min(pairwise):.3f}  Max: {np.max(pairwise):.3f}")
        print(f"  Pairs > 0.7 (near-duplicates): {np.sum(pairwise > 0.7)}/{len(pairwise)}")

    return metrics


# ---------- Step 4: Select most diverse subset ----------

def select_diverse_identities(embeddings: np.ndarray, names: list, k: int) -> list:
    """Greedy farthest-point selection: pick K most spread-out identities."""
    if len(embeddings) <= k:
        print(f"Only {len(embeddings)} embeddings, returning all")
        return list(range(len(embeddings)))

    cos_sim = embeddings @ embeddings.T

    # Start with the identity most different from all others (lowest mean similarity)
    mean_sims = cos_sim.mean(axis=1)
    selected = [int(np.argmin(mean_sims))]

    for _ in range(k - 1):
        # For each candidate, find max similarity to any already-selected
        max_sim_to_selected = cos_sim[:, selected].max(axis=1)
        # Mask already selected
        max_sim_to_selected[selected] = 2.0
        # Pick the one with lowest max-similarity (most different from selected set)
        next_idx = int(np.argmin(max_sim_to_selected))
        selected.append(next_idx)

    selected_names = [names[i] for i in selected]
    print(f"\nSelected {k} most diverse identities:")
    for i, idx in enumerate(selected):
        print(f"  {i+1}. {names[idx]} (mean sim to others: {cos_sim[idx, selected].mean():.3f})")

    return selected


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Generate face identity pool")
    parser.add_argument("--num-faces", type=int, default=200, help="Faces to generate")
    parser.add_argument("--num-select", type=int, default=20, help="Identities to select")
    parser.add_argument("--truncation", type=float, default=0.7, help="StyleGAN truncation psi")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--measure-only", action="store_true", help="Skip generation, measure existing")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device within CUDA_VISIBLE_DEVICES scope")
    args = parser.parse_args()

    find_deceived_paths()

    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        base_dir = PROJECT_ROOT / "data" / "celeb-k" / "face_pool"

    faces_dir = base_dir / "generated"
    emb_dir = base_dir / "embeddings"
    selected_dir = base_dir / "selected"

    # Step 1: Generate
    if not args.measure_only:
        generate_faces(args.num_faces, faces_dir, args.truncation, args.device)

    # Step 2: Extract embeddings
    device_id = 0  # always 0 within CUDA_VISIBLE_DEVICES scope
    embeddings, names = extract_embeddings(faces_dir, emb_dir, device_id)

    # Step 3: Quality metrics
    metrics = measure_quality(faces_dir, embeddings, names, real_dir=CELEB_K_SAMPLE)

    # Step 4: Select diverse subset
    selected_indices = select_diverse_identities(embeddings, names, args.num_select)
    selected_dir.mkdir(parents=True, exist_ok=True)

    # Copy selected face images and embeddings
    import shutil
    selected_info = []
    for rank, idx in enumerate(selected_indices):
        name = names[idx]
        src_img = faces_dir / f"{name}.png"
        src_emb = emb_dir / f"{name}.npy"
        if src_img.exists():
            shutil.copy2(src_img, selected_dir / f"{name}.png")
        if src_emb.exists():
            shutil.copy2(src_emb, selected_dir / f"{name}.npy")
        selected_info.append({
            "rank": rank,
            "name": name,
            "seed": int(name.split("_")[-1]),
        })

    metrics["selected_identities"] = selected_info
    metrics["config"] = {
        "num_generated": args.num_faces,
        "num_selected": args.num_select,
        "truncation": args.truncation,
        "model": "DeceiveD_1_Uncond",
    }

    # Save report
    report_path = base_dir / "face_pool_report.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nReport saved: {report_path}")
    print(f"Selected faces: {selected_dir}")
    print(f"\nTo use in pipeline: --identity-ref-dir {selected_dir}")


if __name__ == "__main__":
    main()
