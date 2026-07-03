#!/usr/bin/env python3
"""CLI: Full end-to-end pipeline (generate → annotate → validate → package)."""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config


def run_stage(script: str, args: list[str], stage_name: str):
    """Run a pipeline stage as a subprocess."""
    print(f"\n{'='*60}")
    print(f"  Stage: {stage_name}")
    print(f"{'='*60}\n")
    cmd = [sys.executable, str(Path(__file__).parent / script)] + args
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: {stage_name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Full synthetic dataset pipeline")
    parser.add_argument("--config", default="config/default.yaml", nargs="+")
    parser.add_argument("--scenario", type=int, choices=[1, 2], default=1)
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--mode", choices=["text2img", "conditioned", "faceid"], default="text2img")
    parser.add_argument("--frames-dir", default=None)
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    overrides = {"scenario": args.scenario, "mode": args.mode}
    if args.num_images:
        overrides.setdefault("prompt", {})["num_prompts"] = args.num_images
    if args.run_id:
        overrides["run_id"] = args.run_id

    config = load_config(*args.config, overrides=overrides)
    run_dir = config.run_dir

    print(f"Pipeline: scenario={args.scenario}, mode={args.mode}, images={args.num_images}")
    print(f"Output: {run_dir}")

    # Stage 1: Generate
    gen_args = ["--config"] + args.config + [
        "--mode", args.mode,
        "--scenario", str(args.scenario),
        "--num-images", str(args.num_images),
        "--run-id", str(config.run_id),
    ]
    if args.frames_dir:
        gen_args += ["--frames-dir", args.frames_dir]
    run_stage("run_generate.py", gen_args, "Image Generation")

    # Stage 2: Annotate
    run_stage("run_annotate.py", [
        "--config"] + args.config + [
        "--run-dir", str(run_dir),
        "--scenario", str(args.scenario),
    ], "Annotation")

    # Stage 3: Validate
    if not args.skip_validate:
        run_stage("run_validate.py", [
            "--config"] + args.config + [
            "--run-dir", str(run_dir),
        ], "Validation")

    # Stage 4: Package
    from src.dataset.packager import package_dataset
    print(f"\n{'='*60}")
    print(f"  Stage: Dataset Packaging")
    print(f"{'='*60}\n")
    package_dataset(run_dir)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete! Output: {run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
