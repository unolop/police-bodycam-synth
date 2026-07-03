#!/usr/bin/env python3
"""Evaluate synthetic bodycam images using the project evaluation framework.

Metrics computed:
  Utility:
    - FID/KID: distribution similarity vs reference dataset (if available)
    - CLIP score: text-image alignment per prompt

  Safety:
    - SSIM: structural similarity (vs reference or intra-set diversity)
    - LPIPS: perceptual similarity (vs reference or intra-set diversity)

  Face Quality:
    - Face detection rate (InsightFace buffalo_l)
    - Face area statistics
    - Embedding discriminability (pairwise cosine similarity distribution)

Usage:
    python scripts/run_evaluation.py output/overnight_test/scenario_1
    python scripts/run_evaluation.py output/overnight_test/scenario_1 --reference-dir data/real_bodycam
    python scripts/run_evaluation.py output/overnight_test --all-scenarios
"""

import argparse
import json
import sys
import gc
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def evaluate_directory(
    image_dir: Path,
    reference_dir: Path | None = None,
    prompts_path: Path | None = None,
    output_path: Path | None = None,
    skip_lpips: bool = False,
    max_pairs: int = 200,
):
    """Run full evaluation on a single directory of images."""
    import torch

    if output_path is None:
        output_path = image_dir / "evaluation_report.json"

    print(f"\nEvaluating: {image_dir}")
    print(f"  Reference: {reference_dir or 'None (intra-set diversity mode)'}")
    print(f"  Output: {output_path}")

    results = {}
    t0 = time.time()

    # ── 1. Face detection + embedding quality ──
    print("\n[1/4] Face detection & embedding analysis...")
    from src.validate.face_quality import compute_face_detection_and_embeddings

    emb_dir = image_dir / "embeddings"
    face_metrics = compute_face_detection_and_embeddings(
        image_dir, output_dir=emb_dir,
    )
    results["face_quality"] = face_metrics
    per_image = face_metrics.pop("per_image", [])

    rate = face_metrics.get("detection_rate", 0)
    print(f"  Face detection: {face_metrics.get('faces_detected', 0)}"
          f"/{face_metrics.get('total_images', 0)} ({rate*100:.1f}%)")
    emb_stats = face_metrics.get("embedding_stats", {})
    if emb_stats:
        print(f"  Cosine sim: {emb_stats.get('cosine_sim_mean', 0):.3f} "
              f"± {emb_stats.get('cosine_sim_std', 0):.3f}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 2. SSIM ──
    print("\n[2/4] SSIM computation...")
    from src.validate.ssim_lpips import compute_ssim_against_reference

    ssim_metrics = compute_ssim_against_reference(
        image_dir, reference_dir, max_pairs=max_pairs,
    )
    results["ssim"] = ssim_metrics
    key = "ssim_mean" if "ssim_mean" in ssim_metrics else "ssim_intra_mean"
    if ssim_metrics.get(key) is not None:
        print(f"  SSIM ({ssim_metrics.get('mode', '')}): {ssim_metrics[key]:.4f}")

    # ── 3. LPIPS ──
    if not skip_lpips:
        print("\n[3/4] LPIPS computation...")
        from src.validate.ssim_lpips import compute_lpips_scores

        lpips_metrics = compute_lpips_scores(
            image_dir, reference_dir, max_pairs=max_pairs,
        )
        results["lpips"] = lpips_metrics
        key = "lpips_mean" if "lpips_mean" in lpips_metrics else "lpips_intra_mean"
        if lpips_metrics.get(key) is not None:
            print(f"  LPIPS ({lpips_metrics.get('mode', '')}): {lpips_metrics[key]:.4f}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("\n[3/4] LPIPS skipped")
        results["lpips"] = {"note": "skipped"}

    # ── 4. FID/KID ──
    print("\n[4/4] FID/KID computation...")
    from src.validate.fid_kid import compute_fid_kid

    fid_kid = compute_fid_kid(image_dir, reference_dir)
    results["fid_kid"] = fid_kid
    if fid_kid.get("fid") is not None:
        print(f"  FID: {fid_kid['fid']:.2f}, KID: {fid_kid.get('kid', 'N/A')}")
    else:
        print(f"  FID/KID: {fid_kid.get('note', 'skipped')}")

    # ── Generate report ──
    from src.validate.report import generate_report

    report = generate_report(
        face_metrics=results["face_quality"],
        fid_kid_metrics=results["fid_kid"],
        ssim_metrics=results["ssim"],
        lpips_metrics=results["lpips"],
        output_path=output_path,
    )

    elapsed = time.time() - t0
    print(f"\nEvaluation completed in {elapsed:.1f}s")

    return report


def evaluate_all_scenarios(
    base_dir: Path,
    reference_dir: Path | None = None,
    skip_lpips: bool = False,
    max_pairs: int = 200,
):
    """Evaluate all scenario directories under base_dir."""
    scenario_dirs = sorted(base_dir.glob("scenario_*"))
    scenario_dirs = [d for d in scenario_dirs if d.is_dir() and "meta" not in d.name]

    if not scenario_dirs:
        print(f"No scenario_* directories found in {base_dir}")
        return

    all_reports = {}
    for sdir in scenario_dirs:
        report = evaluate_directory(
            sdir,
            reference_dir=reference_dir,
            output_path=sdir / "evaluation_report.json",
            skip_lpips=skip_lpips,
            max_pairs=max_pairs,
        )
        all_reports[sdir.name] = report

    # Combined summary
    print("\n" + "=" * 70)
    print("COMBINED EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<30} ", end="")
    for name in all_reports:
        print(f"{name:>18} ", end="")
    print()
    print("-" * 70)

    # Face detection rate
    print(f"{'Face detection rate':<30} ", end="")
    for name, r in all_reports.items():
        fq = r.get("face_quality", {})
        print(f"{fq.get('detection_rate_pct', 'N/A'):>18} ", end="")
    print()

    # Face area
    print(f"{'Avg face area (px²)':<30} ", end="")
    for name, r in all_reports.items():
        fq = r.get("face_quality", {})
        val = fq.get("face_area_mean", 0)
        print(f"{val:>18,.0f} ", end="")
    print()

    # Cosine sim
    print(f"{'Embedding cosine sim mean':<30} ", end="")
    for name, r in all_reports.items():
        fq = r.get("face_quality", {})
        emb = fq.get("embedding_stats", {})
        val = emb.get("cosine_sim_mean", None)
        print(f"{val:>18.3f} " if val is not None else f"{'N/A':>18} ", end="")
    print()

    # SSIM
    print(f"{'SSIM (intra-set)':<30} ", end="")
    for name, r in all_reports.items():
        s = r.get("safety", {}).get("ssim", {})
        key = "ssim_mean" if "ssim_mean" in s else "ssim_intra_mean"
        val = s.get(key, None)
        print(f"{val:>18.4f} " if val is not None else f"{'N/A':>18} ", end="")
    print()

    # LPIPS
    print(f"{'LPIPS (intra-set)':<30} ", end="")
    for name, r in all_reports.items():
        lp = r.get("safety", {}).get("lpips", {})
        key = "lpips_mean" if "lpips_mean" in lp else "lpips_intra_mean"
        val = lp.get(key, None)
        print(f"{val:>18.4f} " if val is not None else f"{'N/A':>18} ", end="")
    print()

    print("=" * 70)

    # Save combined
    combined_path = base_dir / "evaluation_combined.json"
    with open(combined_path, "w") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    print(f"\nCombined report saved to {combined_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate synthetic bodycam images")
    parser.add_argument("image_dir", type=str, help="Directory of generated images")
    parser.add_argument("--reference-dir", type=str, default=None,
                        help="Directory of real reference images (for FID, SSIM, LPIPS)")
    parser.add_argument("--all-scenarios", action="store_true",
                        help="Evaluate all scenario_* subdirectories")
    parser.add_argument("--skip-lpips", action="store_true",
                        help="Skip LPIPS computation (saves time/VRAM)")
    parser.add_argument("--max-pairs", type=int, default=200,
                        help="Max pairs for SSIM/LPIPS sampling (default: 200)")

    args = parser.parse_args()
    image_dir = Path(args.image_dir)

    if args.all_scenarios:
        evaluate_all_scenarios(
            image_dir,
            reference_dir=Path(args.reference_dir) if args.reference_dir else None,
            skip_lpips=args.skip_lpips,
            max_pairs=args.max_pairs,
        )
    else:
        evaluate_directory(
            image_dir,
            reference_dir=Path(args.reference_dir) if args.reference_dir else None,
            skip_lpips=args.skip_lpips,
            max_pairs=args.max_pairs,
        )
