"""Aggregate quality validation report.

Combines all evaluation metrics following the synthetic data evaluation
framework from the project PPTs:
  - Utility: FID, CLIP score, model performance
  - Safety: SSIM, LPIPS (re-identification risk)
  - Face quality: detection rate, embedding discriminability
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np


def generate_report(
    clip_results: list[dict] | None = None,
    face_metrics: dict | None = None,
    fid_kid_metrics: dict | None = None,
    ssim_metrics: dict | None = None,
    lpips_metrics: dict | None = None,
    coco_path: Path | None = None,
    output_path: Path = Path("output/evaluation_report.json"),
    min_clip_score: float = 0.2,
) -> dict:
    """Generate aggregate quality report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "framework": "Synthetic Bodycam Data Evaluation",
    }

    # ── Utility metrics ──
    utility = {}

    # CLIP text-image alignment
    if clip_results:
        scores = [r["clip_score"] for r in clip_results]
        flagged = [r for r in clip_results if r["clip_score"] < min_clip_score]
        utility["clip"] = {
            "total_images": len(clip_results),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "flagged_low_clip": len(flagged),
        }

    # FID/KID distribution similarity
    if fid_kid_metrics:
        utility["fid_kid"] = fid_kid_metrics

    report["utility"] = utility

    # ── Safety metrics ──
    safety = {}

    if ssim_metrics:
        safety["ssim"] = ssim_metrics

    if lpips_metrics:
        safety["lpips"] = lpips_metrics

    report["safety"] = safety

    # ── Face quality ──
    if face_metrics:
        report["face_quality"] = face_metrics

    # ── Category distribution ──
    if coco_path and coco_path.exists():
        coco = json.loads(coco_path.read_text())
        cat_names = {c["id"]: c["name"] for c in coco["categories"]}
        category_dist = {}
        for ann in coco["annotations"]:
            name = cat_names.get(ann["category_id"], "unknown")
            category_dist[name] = category_dist.get(name, 0) + 1
        report["category_distribution"] = category_dist

    # ── Summary verdicts ──
    verdicts = []
    if face_metrics and "detection_rate" in face_metrics:
        rate = face_metrics["detection_rate"]
        verdicts.append({
            "metric": "face_detection_rate",
            "value": rate,
            "threshold": 0.85,
            "pass": rate >= 0.85,
        })
    if ssim_metrics and "ssim_mean" in ssim_metrics and ssim_metrics["ssim_mean"] is not None:
        val = ssim_metrics["ssim_mean"]
        verdicts.append({
            "metric": "ssim_safety",
            "value": val,
            "threshold": 0.6,
            "pass": val < 0.6,
            "note": "Below threshold = safe (not too similar to reference)",
        })
    if lpips_metrics and "lpips_mean" in lpips_metrics and lpips_metrics["lpips_mean"] is not None:
        val = lpips_metrics["lpips_mean"]
        verdicts.append({
            "metric": "lpips_safety",
            "value": val,
            "threshold": 0.3,
            "pass": val > 0.3,
            "note": "Above threshold = safe (perceptually different from reference)",
        })
    if fid_kid_metrics and fid_kid_metrics.get("fid") is not None:
        fid_val = fid_kid_metrics["fid"]
        verdicts.append({
            "metric": "fid",
            "value": fid_val,
            "note": "Lower is better (closer to real distribution)",
        })

    report["verdicts"] = verdicts

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\nEvaluation report saved to {output_path}")
    _print_summary(report)

    return report


def _print_summary(report: dict):
    """Print human-readable summary."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    # Face quality
    fq = report.get("face_quality", {})
    if fq:
        print(f"\nFace Quality:")
        print(f"  Detection rate: {fq.get('detection_rate_pct', 'N/A')}")
        print(f"  Avg face area: {fq.get('face_area_mean', 0):,.0f} px²")
        emb = fq.get("embedding_stats", {})
        if emb:
            print(f"  Embedding cosine sim: {emb.get('cosine_sim_mean', 0):.3f} "
                  f"± {emb.get('cosine_sim_std', 0):.3f}")
            print(f"  High-similarity pairs (>0.7): "
                  f"{emb.get('pairs_above_0.7', 0)}/{emb.get('total_pairs', 0)}")

    # Utility
    util = report.get("utility", {})
    if "clip" in util:
        c = util["clip"]
        print(f"\nUtility — CLIP Score:")
        print(f"  Mean: {c['score_mean']:.3f} ± {c['score_std']:.3f}")
    if "fid_kid" in util:
        fk = util["fid_kid"]
        if fk.get("fid") is not None:
            print(f"\nUtility — FID/KID:")
            print(f"  FID: {fk['fid']:.2f}")
            print(f"  KID: {fk.get('kid', 'N/A')}")

    # Safety
    safety = report.get("safety", {})
    if "ssim" in safety:
        s = safety["ssim"]
        key = "ssim_mean" if "ssim_mean" in s else "ssim_intra_mean"
        if s.get(key) is not None:
            print(f"\nSafety — SSIM ({s.get('mode', '')}):")
            print(f"  Mean: {s[key]:.4f}")
    if "lpips" in safety:
        lp = safety["lpips"]
        key = "lpips_mean" if "lpips_mean" in lp else "lpips_intra_mean"
        if lp.get(key) is not None:
            print(f"\nSafety — LPIPS ({lp.get('mode', '')}):")
            print(f"  Mean: {lp[key]:.4f}")

    # Verdicts
    verdicts = report.get("verdicts", [])
    if verdicts:
        print(f"\nVerdicts:")
        for v in verdicts:
            status = "PASS" if v.get("pass", True) else "FAIL"
            val = v["value"]
            val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
            print(f"  [{status}] {v['metric']}: {val_str}")

    print("=" * 60)
