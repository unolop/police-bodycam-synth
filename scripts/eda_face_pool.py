#!/usr/bin/env python3
"""EDA on generated face identity pool."""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
emb_dir = PROJECT_ROOT / "data/celeb-k/face_pool/embeddings"
img_dir = PROJECT_ROOT / "data/celeb-k/face_pool/generated"
report_path = PROJECT_ROOT / "data/celeb-k/face_pool/face_pool_report.json"

emb_files = sorted(emb_dir.glob("*.npy"))
embeddings = np.stack([np.load(f) for f in emb_files])
seeds = [int(f.stem.split("_")[-1]) for f in emb_files]
print(f"Loaded {len(embeddings)} embeddings, shape: {embeddings.shape}")

# Pairwise cosine similarity
cos_sim = embeddings @ embeddings.T
triu = np.triu_indices(len(embeddings), k=1)
pairwise = cos_sim[triu]
norms = np.linalg.norm(embeddings, axis=1)

# PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

# Selected identities
with open(report_path) as f:
    report = json.load(f)
selected_seeds = [s["seed"] for s in report["selected_identities"]]
selected_mask = np.array([s in selected_seeds for s in seeds])

fig = plt.figure(figsize=(18, 10))

# 1. Cosine similarity histogram
ax1 = fig.add_subplot(2, 3, 1)
ax1.hist(pairwise, bins=50, color="steelblue", edgecolor="white")
ax1.axvline(pairwise.mean(), color="red", linestyle="--", label=f"mean={pairwise.mean():.3f}")
ax1.axvline(0.7, color="orange", linestyle="--", label="0.7 near-dup threshold")
ax1.set_title("Pairwise Cosine Similarity Distribution")
ax1.set_xlabel("Cosine Similarity")
ax1.set_ylabel("Pair Count")
ax1.legend(fontsize=8)

# 2. PCA scatter with seed IDs on selected
ax2 = fig.add_subplot(2, 3, 2)
ax2.scatter(coords[~selected_mask, 0], coords[~selected_mask, 1],
            alpha=0.4, s=15, c="steelblue", label="All 200")
ax2.scatter(coords[selected_mask, 0], coords[selected_mask, 1],
            color="red", s=60, label="Selected 20", zorder=5)
for i, (x, y, sel) in enumerate(zip(coords[:, 0], coords[:, 1], selected_mask)):
    if sel:
        ax2.annotate(f"ID{seeds[i]}", (x, y), fontsize=5,
                     ha="center", va="bottom", color="darkred")
ax2.set_title(f"Identity Space PCA (var: {pca.explained_variance_ratio_.sum():.2f})")
ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.3f})")
ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.3f})")
ax2.legend(fontsize=8)

# 3. Embedding norm — ArcFace normalizes to 1.0 so show as bar
ax3 = fig.add_subplot(2, 3, 3)
ax3.bar(["Min", "Mean", "Max", "Std×100"],
        [norms.min(), norms.mean(), norms.max(), norms.std() * 100],
        color=["steelblue", "green", "steelblue", "orange"])
ax3.set_title(f"ArcFace Embedding Norms\n(all ≈1.0, std×100={norms.std()*100:.4f})")
ax3.set_ylabel("Value")
for i, v in enumerate([norms.min(), norms.mean(), norms.max(), norms.std() * 100]):
    ax3.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)

# 4. Embedding heatmap — first 100 dims, all 200 identities
ax4 = fig.add_subplot(2, 3, 4)
im4 = ax4.imshow(embeddings[:, :100], aspect="auto", cmap="RdBu", vmin=-0.15, vmax=0.15)
plt.colorbar(im4, ax=ax4)
ax4.set_title("Embedding Matrix (dims 0-99)")
ax4.set_xlabel("Embedding Dimension")
ax4.set_ylabel("Identity (seed order)")

# 5. Full cosine similarity matrix
ax5 = fig.add_subplot(2, 3, 5)
im5 = ax5.imshow(cos_sim, aspect="auto", cmap="hot", vmin=0, vmax=1)
plt.colorbar(im5, ax=ax5)
ax5.set_title("Full Pairwise Cosine Similarity (200×200)")
ax5.set_xlabel("Identity Index")
ax5.set_ylabel("Identity Index")

# 6. Selected 20 face thumbnails grid
ax6 = fig.add_subplot(2, 3, 6)
sel_indices = [i for i, s in enumerate(selected_mask) if s]
grid = Image.new("RGB", (5 * 64, 4 * 64), (220, 220, 220))
for rank, idx in enumerate(sel_indices[:20]):
    seed = seeds[idx]
    img_path = img_dir / f"face_{seed:04d}.png"
    if img_path.exists():
        img = Image.open(img_path).resize((64, 64))
        r, c = rank // 5, rank % 5
        grid.paste(img, (c * 64, r * 64))
ax6.imshow(np.array(grid))
ax6.set_title("Selected 20 Identities (thumbnails)")
ax6.axis("off")

plt.suptitle("Face Identity Pool EDA — 200 DeceiveD Generated Faces", fontsize=13, y=1.01)
plt.tight_layout()
out_path = PROJECT_ROOT / "data/celeb-k/face_pool/eda.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

# Stats summary
print(f"\n=== Stats ===")
print(f"Identities: {len(embeddings)}")
print(f"Cosine sim: mean={pairwise.mean():.3f}, std={pairwise.std():.3f}, min={pairwise.min():.3f}, max={pairwise.max():.3f}")
print(f"Near-duplicates (>0.7): {(pairwise > 0.7).sum()} / {len(pairwise)} pairs")
print(f"Very different (<0.1): {(pairwise < 0.1).sum()} / {len(pairwise)} pairs")
print(f"Embedding norm: mean={norms.mean():.6f}, std={norms.std():.8f}")
print(f"PCA variance explained: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
