"""SSIM and LPIPS metrics for synthetic data safety and utility evaluation.

Safety perspective: synthetic images that are too similar to reference/original
images pose re-identification risk. We measure:
  - SSIM (Structural Similarity): higher = more structurally similar
  - LPIPS (Learned Perceptual Image Patch Similarity): lower = more perceptually similar

Utility perspective: intra-set diversity — synthetic images should cover a
broad range of appearances rather than collapsing to a few modes.
"""

import numpy as np
from pathlib import Path
from PIL import Image


def compute_ssim_pair(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two images (grayscale or per-channel mean)."""
    from skimage.metrics import structural_similarity
    if img1.shape != img2.shape:
        # Resize img2 to match img1
        img2 = np.array(Image.fromarray(img2).resize(
            (img1.shape[1], img1.shape[0]), Image.BILINEAR
        ))
    if img1.ndim == 3:
        return structural_similarity(img1, img2, channel_axis=2, data_range=255)
    return structural_similarity(img1, img2, data_range=255)


def compute_ssim_against_reference(
    synthetic_dir: Path,
    reference_dir: Path | None,
    max_pairs: int = 500,
    seed: int = 42,
) -> dict:
    """Compute SSIM between synthetic images and reference images.

    For safety evaluation: high SSIM means synthetic images look too similar
    to originals, posing re-identification risk.

    If no reference dir, computes intra-set diversity (pairwise within synthetic).
    """
    syn_paths = sorted(Path(synthetic_dir).glob("*.png"))
    if not syn_paths:
        return {"ssim_mean": None, "note": "No synthetic images found"}

    rng = np.random.RandomState(seed)

    if reference_dir and Path(reference_dir).exists():
        ref_paths = sorted(Path(reference_dir).glob("*.png"))
        if not ref_paths:
            ref_paths = sorted(Path(reference_dir).glob("*.jpg"))
        if not ref_paths:
            return {"ssim_mean": None, "note": "No reference images found"}

        # Sample pairs: each synthetic vs random reference
        n_pairs = min(max_pairs, len(syn_paths))
        indices = rng.choice(len(syn_paths), n_pairs, replace=False)
        scores = []
        for idx in indices:
            syn_img = np.array(Image.open(syn_paths[idx]).convert("RGB"))
            ref_img = np.array(Image.open(
                ref_paths[rng.randint(len(ref_paths))]
            ).convert("RGB"))
            scores.append(compute_ssim_pair(syn_img, ref_img))

        return {
            "ssim_mean": float(np.mean(scores)),
            "ssim_std": float(np.std(scores)),
            "ssim_max": float(np.max(scores)),
            "ssim_min": float(np.min(scores)),
            "num_pairs": len(scores),
            "mode": "synthetic_vs_reference",
        }
    else:
        # Intra-set diversity: sample random pairs within synthetic set
        n_pairs = min(max_pairs, len(syn_paths) * (len(syn_paths) - 1) // 2)
        scores = []
        for _ in range(n_pairs):
            i, j = rng.choice(len(syn_paths), 2, replace=False)
            img_i = np.array(Image.open(syn_paths[i]).convert("RGB"))
            img_j = np.array(Image.open(syn_paths[j]).convert("RGB"))
            scores.append(compute_ssim_pair(img_i, img_j))

        return {
            "ssim_intra_mean": float(np.mean(scores)),
            "ssim_intra_std": float(np.std(scores)),
            "ssim_intra_max": float(np.max(scores)),
            "num_pairs": len(scores),
            "mode": "intra_set_diversity",
        }


def compute_lpips_scores(
    synthetic_dir: Path,
    reference_dir: Path | None,
    max_pairs: int = 500,
    seed: int = 42,
) -> dict:
    """Compute LPIPS perceptual distance.

    Lower LPIPS = more perceptually similar (higher re-identification risk).

    Uses AlexNet backbone (fast, well-calibrated for perceptual similarity).
    """
    import torch
    import lpips

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = lpips.LPIPS(net="alex").to(device)

    syn_paths = sorted(Path(synthetic_dir).glob("*.png"))
    if not syn_paths:
        return {"lpips_mean": None, "note": "No synthetic images found"}

    rng = np.random.RandomState(seed)

    def load_tensor(path):
        img = Image.open(path).convert("RGB").resize((256, 256), Image.BILINEAR)
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
        return t.unsqueeze(0).to(device)

    if reference_dir and Path(reference_dir).exists():
        ref_paths = sorted(Path(reference_dir).glob("*.png"))
        if not ref_paths:
            ref_paths = sorted(Path(reference_dir).glob("*.jpg"))
        if not ref_paths:
            return {"lpips_mean": None, "note": "No reference images found"}

        n_pairs = min(max_pairs, len(syn_paths))
        indices = rng.choice(len(syn_paths), n_pairs, replace=False)
        scores = []
        with torch.no_grad():
            for idx in indices:
                syn_t = load_tensor(syn_paths[idx])
                ref_t = load_tensor(ref_paths[rng.randint(len(ref_paths))])
                d = loss_fn(syn_t, ref_t).item()
                scores.append(d)

        return {
            "lpips_mean": float(np.mean(scores)),
            "lpips_std": float(np.std(scores)),
            "lpips_min": float(np.min(scores)),
            "lpips_max": float(np.max(scores)),
            "num_pairs": len(scores),
            "mode": "synthetic_vs_reference",
            "note": "Lower LPIPS = more similar (higher re-ID risk)",
        }
    else:
        # Intra-set diversity
        n_pairs = min(max_pairs, len(syn_paths) * (len(syn_paths) - 1) // 2)
        scores = []
        with torch.no_grad():
            for _ in range(n_pairs):
                i, j = rng.choice(len(syn_paths), 2, replace=False)
                syn_i = load_tensor(syn_paths[i])
                syn_j = load_tensor(syn_paths[j])
                d = loss_fn(syn_i, syn_j).item()
                scores.append(d)

        return {
            "lpips_intra_mean": float(np.mean(scores)),
            "lpips_intra_std": float(np.std(scores)),
            "lpips_intra_min": float(np.min(scores)),
            "num_pairs": len(scores),
            "mode": "intra_set_diversity",
            "note": "Higher intra-set LPIPS = more diverse (better)",
        }

    del loss_fn
    torch.cuda.empty_cache()
