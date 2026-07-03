#!/usr/bin/env python3
"""Overnight test: generate images across all 3 scenarios with multiple seeds.

Generates batches to evaluate:
  - Scene realism and prompt adherence
  - Face visibility (forward-facing subjects)
  - Threat objects rendering (scenario 3)
  - Child/elderly rendering (scenario 2)
  - Location and lighting diversity

Output structure:
  output/overnight_test/
    scenario_1/  (POI identification)
    scenario_2/  (missing person)
    scenario_3/  (threat detection)
    prompts/     (saved prompts for each run)
    report.json  (face detection stats)
"""

import sys
import json
import time
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.config import load_config
from src.prompts.scenario_engine import generate_prompts
from src.prompts.prompt_store import save_prompts, load_prompts


# ── Configuration ──────────────────────────────────────────────
IMAGES_PER_SCENARIO = 200      # 200 images × 3 scenarios = 600 total
SEEDS = [42, 123, 456, 789, 1001, 2024, 3141, 5555, 7777, 9999]  # 10 seeds
PROMPTS_PER_SEED = 20
MODEL = "SG161222/RealVisXL_V5.0"
STEPS = 30
GUIDANCE = 7.5
WIDTH, HEIGHT = 1024, 1024

OUTPUT_DIR = Path("output/overnight_test")


def generate_all_prompts():
    """Generate prompts for all scenarios and seeds."""
    prompts_dir = OUTPUT_DIR / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    all_prompts = {}
    for scenario in [1, 2, 3]:
        scenario_prompts = []
        for seed in SEEDS:
            for p in generate_prompts(scenario, PROMPTS_PER_SEED, seed):
                # Make prompt_id unique across seeds
                p["prompt_id"] = f"s{scenario}_seed{seed}_{p['prompt_id'].split('_')[1]}"
                p["seed"] = seed
                scenario_prompts.append(p)

        # Save prompts
        path = prompts_dir / f"scenario_{scenario}.jsonl"
        with open(path, "w") as f:
            for p in scenario_prompts:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        all_prompts[scenario] = scenario_prompts
        print(f"  Scenario {scenario}: {len(scenario_prompts)} prompts generated")

    return all_prompts


def generate_images(all_prompts):
    """Generate images for all scenarios."""
    from diffusers import StableDiffusionXLPipeline
    from compel import Compel, ReturnedEmbeddingsType

    dtype = torch.float16
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL, torch_dtype=dtype, use_safetensors=True, variant="fp16",
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
        truncate_long_prompts=False,
    )

    total = sum(len(v) for v in all_prompts.values())
    done = 0
    t0 = time.time()

    for scenario, prompts in all_prompts.items():
        image_dir = OUTPUT_DIR / f"scenario_{scenario}"
        image_dir.mkdir(parents=True, exist_ok=True)
        meta_dir = OUTPUT_DIR / f"scenario_{scenario}_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        # Check what's already done
        existing = {f.stem for f in image_dir.glob("*.png")}

        for p in prompts:
            pid = p["prompt_id"]
            if pid in existing:
                done += 1
                continue

            generator = torch.Generator(device="cuda").manual_seed(p["seed"])

            prompt_embeds, pooled = compel(p["prompt"])
            neg_embeds, neg_pooled = compel(p["negative_prompt"])

            max_len = max(prompt_embeds.shape[1], neg_embeds.shape[1])
            if prompt_embeds.shape[1] < max_len:
                prompt_embeds = torch.cat([prompt_embeds,
                    torch.zeros(1, max_len - prompt_embeds.shape[1], prompt_embeds.shape[2],
                                device=prompt_embeds.device, dtype=prompt_embeds.dtype)], dim=1)
            if neg_embeds.shape[1] < max_len:
                neg_embeds = torch.cat([neg_embeds,
                    torch.zeros(1, max_len - neg_embeds.shape[1], neg_embeds.shape[2],
                                device=neg_embeds.device, dtype=neg_embeds.dtype)], dim=1)

            img = pipe(
                prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled,
                negative_prompt_embeds=neg_embeds, negative_pooled_prompt_embeds=neg_pooled,
                width=WIDTH, height=HEIGHT,
                num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                generator=generator,
            ).images[0]

            img.save(image_dir / f"{pid}.png")

            # Save metadata
            meta = {
                "prompt_id": pid,
                "scenario": scenario,
                "prompt": p["prompt"],
                "seed": p["seed"],
                "slots": p.get("slots", {}),
                "model": MODEL,
                "steps": STEPS,
                "guidance_scale": GUIDANCE,
            }
            (meta_dir / f"{pid}.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2)
            )

            done += 1
            elapsed = time.time() - t0
            eta = (elapsed / done) * (total - done) if done > 0 else 0
            print(f"  [{done}/{total}] {pid} ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    del pipe, compel
    gc.collect()
    torch.cuda.empty_cache()


def run_face_detection():
    """Run face detection on all generated images and produce a report."""
    from insightface.app import FaceAnalysis
    import cv2

    print("Running face detection on all images...")
    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    report = {}
    for scenario in [1, 2, 3]:
        image_dir = OUTPUT_DIR / f"scenario_{scenario}"
        images = sorted(image_dir.glob("*.png"))

        results = []
        faces_found = 0
        total_face_area = 0

        for img_path in images:
            img_cv = cv2.imread(str(img_path))
            faces = face_app.get(img_cv)

            if faces:
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                x1, y1, x2, y2 = [int(c) for c in face.bbox]
                w, h = x2 - x1, y2 - y1
                area = w * h
                faces_found += 1
                total_face_area += area
                results.append({
                    "image": img_path.name,
                    "face_found": True,
                    "face_size": f"{w}x{h}",
                    "face_area": area,
                    "num_faces": len(faces),
                })
            else:
                results.append({
                    "image": img_path.name,
                    "face_found": False,
                })

        n = len(images)
        avg_area = total_face_area / faces_found if faces_found > 0 else 0
        summary = {
            "total_images": n,
            "faces_detected": faces_found,
            "detection_rate": f"{faces_found/n*100:.1f}%" if n > 0 else "0%",
            "avg_face_area_px": int(avg_area),
            "images": results,
        }
        report[f"scenario_{scenario}"] = summary
        print(f"  Scenario {scenario}: {faces_found}/{n} faces detected "
              f"({faces_found/n*100:.1f}%), avg area {int(avg_area)}px²")

    del face_app
    gc.collect()

    # Save report
    report_path = OUTPUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nReport saved to {report_path}")
    return report


def run_object_detection_s3():
    """Run object detection on scenario 3 images to check for threat objects."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  YOLO not available, skipping object detection")
        return {}

    print("Running YOLO object detection on scenario 3 images...")
    model = YOLO("yolov8x.pt")

    image_dir = OUTPUT_DIR / "scenario_3"
    images = sorted(image_dir.glob("*.png"))

    # Objects we care about detecting
    threat_classes = {
        "knife", "scissors", "baseball bat", "bottle", "backpack",
        "handbag", "suitcase", "umbrella",
    }

    results = {}
    threat_found = 0

    for img_path in images:
        detections = model(str(img_path), verbose=False)[0]
        detected_objects = []
        has_threat = False

        for box in detections.boxes:
            cls_name = detections.names[int(box.cls)]
            conf = float(box.conf)
            if conf > 0.3:
                detected_objects.append({"class": cls_name, "conf": round(conf, 3)})
                if cls_name in threat_classes:
                    has_threat = True

        if has_threat:
            threat_found += 1

        results[img_path.name] = {
            "objects": detected_objects,
            "threat_object_detected": has_threat,
        }

    n = len(images)
    print(f"  Scenario 3: {threat_found}/{n} images have detectable threat objects "
          f"({threat_found/n*100:.1f}%)")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Save
    det_path = OUTPUT_DIR / "s3_object_detection.json"
    det_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return {"threat_object_rate": f"{threat_found/n*100:.1f}%", "total": n, "detected": threat_found}


def print_summary(report):
    """Print final summary."""
    print("\n" + "=" * 60)
    print("OVERNIGHT TEST SUMMARY")
    print("=" * 60)
    for scenario_key in ["scenario_1", "scenario_2", "scenario_3"]:
        s = report[scenario_key]
        scenario_num = scenario_key.split("_")[1]
        labels = {"1": "POI Identification", "2": "Missing Person", "3": "Threat Detection"}
        print(f"\nScenario {scenario_num} ({labels[scenario_num]}):")
        print(f"  Images generated: {s['total_images']}")
        print(f"  Face detection rate: {s['detection_rate']}")
        print(f"  Avg face area: {s['avg_face_area_px']}px²")

        # Count no-face images
        no_face = [img for img in s["images"] if not img["face_found"]]
        if no_face:
            print(f"  No face detected in: {', '.join(img['image'] for img in no_face[:10])}")
            if len(no_face) > 10:
                print(f"    ... and {len(no_face) - 10} more")

    print("\n" + "=" * 60)
    total_images = sum(report[k]["total_images"] for k in report)
    total_faces = sum(report[k]["faces_detected"] for k in report)
    print(f"Total: {total_images} images, {total_faces} faces detected "
          f"({total_faces/total_images*100:.1f}%)")


if __name__ == "__main__":
    print("=" * 60)
    print("OVERNIGHT TEST: Scene Quality Evaluation")
    print(f"  {IMAGES_PER_SCENARIO} images × 3 scenarios = {IMAGES_PER_SCENARIO * 3} total")
    print(f"  Model: {MODEL}")
    print(f"  Steps: {STEPS}, Guidance: {GUIDANCE}, Size: {WIDTH}×{HEIGHT}")
    print("=" * 60)

    t_start = time.time()

    print("\n[1/3] Generating prompts...")
    all_prompts = generate_all_prompts()

    print("\n[2/3] Generating images...")
    generate_images(all_prompts)

    print("\n[3/4] Running face detection...")
    report = run_face_detection()

    print("\n[4/4] Running object detection (scenario 3)...")
    s3_obj = run_object_detection_s3()
    if s3_obj:
        report["scenario_3"]["object_detection"] = s3_obj

    print_summary(report)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
