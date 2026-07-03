#!/usr/bin/env python3
"""CLI: Generate synthetic bodycam images (Mode A or B), with scene-sequential support."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic bodycam images")
    parser.add_argument("--config", default=["config/default.yaml"], nargs="*")
    parser.add_argument("--mode", choices=["text2img", "conditioned", "faceid"], default=None)
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--model", default=None, help="Diffusers model repo or local path")
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--frames-dir", default=None, help="Reference frames dir (Mode B)")
    # Scene mode args
    parser.add_argument("--no-scenes", action="store_true",
                        help="Disable scene mode, generate independent images")
    parser.add_argument("--num-scenes", type=int, default=None)
    parser.add_argument("--frames-per-scene", type=int, default=None)
    parser.add_argument("--identity-ref-dir", default=None,
                        help="Dir of reference face images for identity conditioning")
    parser.add_argument("--face-dataset", default=None,
                        help="Path to face ID dataset dir (Mode C: faceid)")
    args = parser.parse_args()

    # Build overrides from CLI args
    overrides = {}
    if args.mode:
        overrides["mode"] = args.mode
    if args.scenario:
        overrides["scenario"] = args.scenario
    if args.model:
        overrides.setdefault("generation", {})["model"] = args.model
    if args.num_images:
        overrides.setdefault("prompt", {})["num_prompts"] = args.num_images
    if args.run_id:
        overrides["run_id"] = args.run_id
    if args.no_scenes:
        overrides.setdefault("scene", {})["enabled"] = False
    if args.num_scenes:
        overrides.setdefault("scene", {})["num_scenes"] = args.num_scenes
    if args.frames_per_scene:
        overrides.setdefault("scene", {})["frames_per_scene"] = args.frames_per_scene
    if args.identity_ref_dir:
        overrides.setdefault("scene", {})["identity_ref_dir"] = args.identity_ref_dir
        overrides.setdefault("scene", {})["identity_enabled"] = True
        overrides.setdefault("ip_adapter", {})["enabled"] = True

    config = load_config(*args.config, overrides=overrides)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # Generate prompts
    prompts_path = run_dir / "prompts.jsonl"
    if not prompts_path.exists():
        if config.scene.enabled:
            # Scene-sequential mode
            from src.prompts.scene_planner import generate_scene_prompts
            from src.prompts.prompt_store import save_prompts

            num_scenes = config.scene.num_scenes
            fpsc = config.scene.frames_per_scene
            total = num_scenes * fpsc
            print(f"Generating {num_scenes} scenes × {fpsc} frames = {total} prompts "
                  f"for scenario {config.scenario}...")
            prompts_iter = generate_scene_prompts(
                config.scenario, num_scenes, fpsc, config.prompt.diversity_seed,
            )
            count = save_prompts(prompts_iter, prompts_path)
            print(f"Saved {count} scene-sequential prompts to {prompts_path}")
        else:
            # Independent mode (original behavior)
            from src.prompts.scenario_engine import generate_prompts
            from src.prompts.prompt_store import save_prompts

            print(f"Generating {config.prompt.num_prompts} prompts for scenario {config.scenario}...")
            prompts_iter = generate_prompts(
                config.scenario, config.prompt.num_prompts, config.prompt.diversity_seed,
            )
            count = save_prompts(prompts_iter, prompts_path)
            print(f"Saved {count} prompts to {prompts_path}")

    from src.prompts.prompt_store import load_prompts
    prompts = load_prompts(prompts_path)

    # Load identity pool if enabled
    identity_pool = None
    if config.scene.identity_enabled and config.scene.identity_ref_dir:
        from src.generate.identity import load_identity_pool, assign_identities_to_scenes, save_identity_pool_meta
        from insightface.app import FaceAnalysis

        print("Loading face identity pool...")
        face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        face_app.prepare(ctx_id=0, det_size=(640, 640))

        ref_dir = Path(config.scene.identity_ref_dir)
        identity_pool = load_identity_pool(ref_dir, face_app)
        save_identity_pool_meta(identity_pool, run_dir / "identity_pool.json")

        # Assign identities to scenes
        scene_ids = set()
        for p in prompts:
            if "scene_id" in p:
                scene_ids.add(p["scene_id"])

        scene_list = sorted(scene_ids)
        assignments = assign_identities_to_scenes(len(scene_list), identity_pool, config.prompt.diversity_seed)
        scene_to_identity = {scene_list[i]: assignments[i] for i in range(len(scene_list))}

        # Attach identity images to prompts
        for p in prompts:
            if "scene_id" in p and p["scene_id"] in scene_to_identity:
                idx = scene_to_identity[p["scene_id"]]
                p["identity_image"] = identity_pool[idx]["image"]

        del face_app
        import gc
        gc.collect()

    # Generate images
    if config.mode == "text2img":
        from src.generate.vram import model_scope
        from src.generate.text2img import generate_batch

        scene_info = ""
        if config.scene.enabled:
            scene_info = f" ({config.scene.num_scenes} scenes × {config.scene.frames_per_scene} frames)"
        print(f"Mode A: Text-to-image generation ({len(prompts)} images{scene_info})")

        with model_scope("text2img", config) as pipe:
            paths = generate_batch(pipe, prompts, config, run_dir)
        print(f"Generated {len(paths)} images in {run_dir / 'images'}")

    elif config.mode == "conditioned":
        if not args.frames_dir:
            print("ERROR: --frames-dir required for conditioned mode")
            sys.exit(1)

        frames_dir = Path(args.frames_dir)
        ref_frames = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
        if not ref_frames:
            print(f"ERROR: No reference frames found in {frames_dir}")
            sys.exit(1)

        from src.generate.vram import model_scope
        from src.generate.conditioned import generate_conditioned_batch

        print(f"Mode B: Conditioned generation ({len(prompts)} images, {len(ref_frames)} ref frames)")
        with model_scope("conditioned", config) as pipe:
            paths = generate_conditioned_batch(pipe, prompts, ref_frames, config, run_dir)
        print(f"Generated {len(paths)} images in {run_dir / 'images'}")

    elif config.mode == "faceid":
        from src.generate.vram import model_scope
        from src.generate.faceid import (
            discover_face_dataset, load_face_labels,
            assign_identities, generate_faceid_batch,
        )

        # Discover face dataset
        face_dataset_dir = args.face_dataset or "data/face id dataset"
        print(f"Loading face ID dataset from {face_dataset_dir}...")
        subjects = discover_face_dataset(face_dataset_dir)
        labels = load_face_labels(face_dataset_dir)
        print(f"  Found {len(subjects)} subjects, {sum(len(v) for v in subjects.values())} reference images")

        # Assign identities to prompts
        prompts = assign_identities(prompts, subjects, labels, seed=config.prompt.diversity_seed)
        id_counts = {}
        for p in prompts:
            sid = p.get("identity_id", "none")
            id_counts[sid] = id_counts.get(sid, 0) + 1
        print(f"  Assigned {len(id_counts)} unique identities across {len(prompts)} prompts")

        # Save identity assignments
        assignments_path = run_dir / "identity_assignments.json"
        assignments_path.write_text(json.dumps(
            [{"prompt_id": p["prompt_id"], "identity_id": p.get("identity_id"),
              "identity_ref": p.get("identity_ref_path"),
              "identity_meta": p.get("identity_meta", {})} for p in prompts],
            ensure_ascii=False, indent=2,
        ))

        print(f"Mode C: Face-ID generation ({len(prompts)} images, {len(subjects)} identities)")
        with model_scope("faceid", config) as pipe:
            paths = generate_faceid_batch(pipe, prompts, config, run_dir)
        print(f"Generated {len(paths)} images in {run_dir / 'images'}")


if __name__ == "__main__":
    main()
