"""Mode B: SDXL + ControlNet + IP-Adapter conditioned generation."""

import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from .preprocessing import extract_canny, extract_openpose


def _load_progress(run_dir: Path) -> set[str]:
    progress_file = run_dir / "generate_progress.json"
    if progress_file.exists():
        data = json.loads(progress_file.read_text())
        return set(data.get("completed_ids", []))
    return set()


def _save_progress(run_dir: Path, completed_ids: set[str], total: int):
    progress_file = run_dir / "generate_progress.json"
    progress_file.write_text(json.dumps({
        "stage": "generate_conditioned",
        "total": total,
        "completed": len(completed_ids),
        "completed_ids": sorted(completed_ids),
    }, indent=2))


def generate_conditioned_batch(
    pipe, prompts: list[dict], reference_frames: list[Path],
    config, run_dir: Path,
) -> list[Path]:
    """Generate images conditioned on reference frames.

    Each prompt is paired with a reference frame. If fewer frames than prompts,
    frames are cycled.

    Args:
        pipe: Loaded SDXL+ControlNet pipeline.
        prompts: Prompt dicts.
        reference_frames: Paths to reference frame images.
        config: PipelineConfig.
        run_dir: Output directory.
    """
    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    cond_dir = run_dir / "conditioning"
    cond_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = run_dir / "image_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    completed_ids = _load_progress(run_dir)
    pending = [(i, p) for i, p in enumerate(prompts) if p["prompt_id"] not in completed_ids]

    if not pending:
        print(f"All {len(prompts)} images already generated.")
        return []

    print(f"Generating {len(pending)} conditioned images ({len(completed_ids)} already done)...")

    generated_paths = []
    generator = torch.Generator(device="cuda").manual_seed(config.generation.seed)
    control_mode = config.controlnet.control_mode

    for idx, entry in tqdm(pending, desc="Generating (conditioned)"):
        prompt_id = entry["prompt_id"]
        ref_path = reference_frames[idx % len(reference_frames)]
        ref_image = Image.open(ref_path).convert("RGB")
        ref_image = ref_image.resize((config.generation.width, config.generation.height))

        # Generate conditioning image
        if control_mode == "canny":
            control_image = extract_canny(ref_image)
        elif control_mode == "openpose":
            control_image = extract_openpose(ref_image)
        else:
            control_image = extract_canny(ref_image)

        # Save conditioning image
        control_image.save(cond_dir / f"{prompt_id}_cond.png")

        # Build generation kwargs
        gen_kwargs = dict(
            prompt=entry["prompt"],
            negative_prompt=entry["negative_prompt"],
            image=control_image,
            width=config.generation.width,
            height=config.generation.height,
            num_inference_steps=config.generation.steps,
            guidance_scale=config.generation.guidance_scale,
            controlnet_conditioning_scale=config.controlnet.conditioning_scale,
            generator=generator,
        )

        # IP-Adapter: pass reference image as ip_adapter_image
        if config.ip_adapter.enabled:
            gen_kwargs["ip_adapter_image"] = ref_image

        image = pipe(**gen_kwargs).images[0]

        image_path = image_dir / f"{prompt_id}.png"
        image.save(image_path)
        generated_paths.append(image_path)

        # Metadata
        meta = {
            "prompt_id": prompt_id,
            "prompt": entry["prompt"],
            "reference_frame": str(ref_path),
            "control_mode": control_mode,
            "conditioning_scale": config.controlnet.conditioning_scale,
            "slots": entry.get("slots", {}),
        }
        (meta_dir / f"{prompt_id}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2)
        )

        completed_ids.add(prompt_id)
        _save_progress(run_dir, completed_ids, len(prompts))

    return generated_paths
