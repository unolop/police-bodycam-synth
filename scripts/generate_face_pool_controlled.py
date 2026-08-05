#!/usr/bin/env python3
"""Generate controlled face identity pool using DeceiveD 4_Cond_Exp (neutral expression).

Strategy:
1. Generate 200 neutral faces (class=2/무표정) with random seeds using 4_Cond_Exp
2. Extract ArcFace embeddings + predict age/gender with InsightFace
3. Save gender/age as metadata alongside each embedding
4. Select 20 most diverse identities via farthest-point sampling

Gender is random (not forced) but recorded — can filter M/F later via metadata.

Expression class indices (Celeb-K A-G):
  0=happy  1=surprise  2=neutral(무표정)  3=disgust  4=angry  5=fear  6=sad

Usage:
    python scripts/generate_face_pool_controlled.py
    python scripts/generate_face_pool_controlled.py --num-faces 200 --num-select 20
"""

import argparse
import glob
import json
import sys
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
CELEB_K = PROJECT_ROOT / "data" / "celeb-k"
NEUTRAL_CLASS = 2  # 무표정


def find_deceived():
    matches = glob.glob(str(CELEB_K / "model_unzipped" / "**/DeceiveD"), recursive=True)
    if not matches:
        raise FileNotFoundError("DeceiveD not found")
    return Path(matches[0])


def load_generator(deceived_root: Path, device: str):
    sys.path.insert(0, str(deceived_root))
    import dnnlib, legacy
    ckpt = list((deceived_root / "Checkpoint" / "4_Cond_Exp").glob("*.pkl"))[0]
    with dnnlib.util.open_url(str(ckpt)) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    print(f"Loaded 4_Cond_Exp: c_dim={G.c_dim}, z_dim={G.z_dim}")
    return G


def generate_face(G, seed: int, class_idx: int, device: str, truncation: float = 0.7):
    import torch
    label = torch.zeros([1, G.c_dim], device=device)
    label[:, class_idx] = 1
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    img = G(z, label, truncation_psi=truncation, noise_mode='const')
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(img[0].cpu().numpy(), 'RGB')


def analyze_face(face_app, img_pil: Image.Image):
    """Returns (embedding, gender, age) or (None, None, None) if no face detected."""
    bordered = Image.new('RGB', (640, 640), (200, 200, 200))
    bordered.paste(img_pil.resize((512, 512)), (64, 64))
    img_cv = cv2.cvtColor(np.array(bordered), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_cv)
    if not faces:
        return None, None, None
    f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    return f.normed_embedding, f.sex, int(f.age)


def select_diverse(embeddings: np.ndarray, k: int) -> list:
    """Greedy farthest-point selection."""
    if len(embeddings) <= k:
        return list(range(len(embeddings)))
    cos_sim = embeddings @ embeddings.T
    selected = [int(np.argmin(cos_sim.mean(axis=1)))]
    for _ in range(k - 1):
        max_sim = cos_sim[:, selected].max(axis=1)
        max_sim[selected] = 2.0
        selected.append(int(np.argmin(max_sim)))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-faces", type=int, default=200)
    parser.add_argument("--num-select", type=int, default=20)
    parser.add_argument("--truncation", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    deceived = find_deceived()
    out_dir = CELEB_K / "face_pool_controlled"
    img_dir = out_dir / "generated"
    emb_dir = out_dir / "embeddings"
    sel_dir = out_dir / "selected"
    for d in [img_dir, emb_dir, sel_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load InsightFace
    from insightface.app import FaceAnalysis
    face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    # Load generator
    G = load_generator(deceived, args.device)

    # Step 1: Generate neutral faces
    print(f"\nGenerating {args.num_faces} neutral (무표정) faces with 4_Cond_Exp...")
    embeddings, records, names = [], [], []

    for seed in range(args.num_faces):
        img = generate_face(G, seed, NEUTRAL_CLASS, args.device, args.truncation)
        img_path = img_dir / f"face_{seed:04d}.png"
        img.save(img_path)

        emb, gender, age = analyze_face(face_app, img)

        if emb is not None:
            np.save(emb_dir / f"face_{seed:04d}.npy", emb)
            embeddings.append(emb)
            names.append(f"face_{seed:04d}")
            records.append({"seed": seed, "name": f"face_{seed:04d}", "gender": gender, "age": age})
        else:
            print(f"  WARNING: no face detected for seed {seed}")

        if (seed + 1) % 20 == 0:
            male = sum(1 for r in records if r["gender"] == "M")
            female = sum(1 for r in records if r["gender"] == "F")
            print(f"  {seed+1}/{args.num_faces} — M:{male} F:{female} detected:{len(records)}")

    embeddings = np.stack(embeddings)
    male_count = sum(1 for r in records if r["gender"] == "M")
    female_count = sum(1 for r in records if r["gender"] == "F")
    print(f"\nTotal: {len(records)} faces — Male: {male_count}, Female: {female_count}")

    # Step 2: Select most diverse
    print(f"\nSelecting {args.num_select} most diverse identities...")
    selected_indices = select_diverse(embeddings, args.num_select)
    selected_info = []

    for rank, idx in enumerate(selected_indices):
        rec = records[idx]
        name = rec["name"]
        shutil.copy2(img_dir / f"{name}.png", sel_dir / f"{name}.png")
        shutil.copy2(emb_dir / f"{name}.npy", sel_dir / f"{name}.npy")
        selected_info.append({**rec, "rank": rank})
        print(f"  {rank+1:2d}. {name}  gender={rec['gender']}  age={rec['age']}")

    # Pairwise stats
    cos_sim = embeddings @ embeddings.T
    triu = np.triu_indices(len(embeddings), k=1)
    pairwise = cos_sim[triu]

    report = {
        "config": {
            "model": "DeceiveD_4_Cond_Exp",
            "expression": "neutral (class=2, 무표정)",
            "truncation": args.truncation,
            "num_generated": args.num_faces,
            "num_selected": args.num_select,
        },
        "stats": {
            "total_detected": len(records),
            "male": male_count,
            "female": female_count,
            "detection_rate": len(records) / args.num_faces,
            "cosine_sim_mean": float(pairwise.mean()),
            "cosine_sim_std": float(pairwise.std()),
            "cosine_sim_min": float(pairwise.min()),
            "cosine_sim_max": float(pairwise.max()),
            "near_duplicates_above_0.7": int((pairwise > 0.7).sum()),
        },
        "all_records": records,
        "selected_identities": selected_info,
    }

    with open(out_dir / "face_pool_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== Done ===")
    print(f"Generated : {args.num_faces} neutral faces")
    print(f"Detected  : {len(records)} ({male_count}M + {female_count}F)")
    print(f"Selected  : {args.num_select} most diverse")
    print(f"Near-dups : {(pairwise > 0.7).sum()}")
    print(f"Output    : {out_dir}")
    print(f"\nUse: --identity-ref-dir {sel_dir}")


if __name__ == "__main__":
    main()
