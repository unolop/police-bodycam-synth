"""Mode A: SDXL text-to-image generation with compel long-prompt support.

Supports scene-sequential generation:
- Frame 0 of each scene: full text2img generation
- Frames 1+: img2img from previous frame with low denoising strength,
  creating natural temporal consistency while the prompt drives action changes.
"""

import json
from pathlib import Path

import torch
from compel import Compel, ReturnedEmbeddingsType
from PIL import Image
from tqdm import tqdm


def _load_progress(run_dir: Path) -> set[str]:
    progress_file = run_dir / "generate_progress.json"
    if progress_file.exists():
        data = json.loads(progress_file.read_text())
        return set(data.get("completed_ids", []))
    return set()


def _save_progress(run_dir: Path, completed_ids: set[str], total: int):
    progress_file = run_dir / "generate_progress.json"
    progress_file.write_text(json.dumps({
        "stage": "generate",
        "total": total,
        "completed": len(completed_ids),
        "completed_ids": sorted(completed_ids),
    }, indent=2))


def _build_compel(pipe):
    return Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
        truncate_long_prompts=False,
    )


def _encode_and_pad(compel, prompt: str, negative_prompt: str):
    prompt_embeds, pooled_prompt_embeds = compel(prompt)
    neg_embeds, neg_pooled = compel(negative_prompt)

    max_len = max(prompt_embeds.shape[1], neg_embeds.shape[1])
    if prompt_embeds.shape[1] < max_len:
        pad = torch.zeros(1, max_len - prompt_embeds.shape[1], prompt_embeds.shape[2],
                          device=prompt_embeds.device, dtype=prompt_embeds.dtype)
        prompt_embeds = torch.cat([prompt_embeds, pad], dim=1)
    if neg_embeds.shape[1] < max_len:
        pad = torch.zeros(1, max_len - neg_embeds.shape[1], neg_embeds.shape[2],
                          device=neg_embeds.device, dtype=neg_embeds.dtype)
        neg_embeds = torch.cat([neg_embeds, pad], dim=1)

    return prompt_embeds, pooled_prompt_embeds, neg_embeds, neg_pooled


def generate_batch(pipes, prompts: list[dict], config, run_dir: Path) -> list[Path]:
    """Generate images with scene-sequential support.

    Args:
        pipes: Either a single pipeline (independent mode) or a dict with
               'text2img' and 'img2img' pipelines (scene mode).
        prompts: List of prompt dicts, ordered by scene_id then frame_index.
        config: PipelineConfig.
        run_dir: Output directory.
    """
    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = run_dir / "image_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    completed_ids = _load_progress(run_dir)

    # Determine mode
    is_scene_mode = isinstance(pipes, dict)
    if is_scene_mode:
        t2i_pipe = pipes["text2img"]
        i2i_pipe = pipes["img2img"]
        strength = config.scene.img2img_strength
    else:
        t2i_pipe = pipes
        i2i_pipe = None
        strength = None

    compel = _build_compel(t2i_pipe)

    # Group prompts by scene for ordered processing
    if is_scene_mode:
        scenes = {}
        for p in prompts:
            sid = p.get("scene_id", "default")
            scenes.setdefault(sid, []).append(p)
        # Sort frames within each scene
        for sid in scenes:
            scenes[sid].sort(key=lambda x: x.get("frame_index", 0))
        ordered_prompts = []
        for sid in sorted(scenes.keys()):
            ordered_prompts.extend(scenes[sid])
    else:
        ordered_prompts = prompts

    pending = [p for p in ordered_prompts if p["prompt_id"] not in completed_ids]
    if not pending:
        print(f"All {len(prompts)} images already generated.")
        return []

    total = len(prompts)
    print(f"Generating {len(pending)} images ({len(completed_ids)} already done)...")

    generated_paths = []
    base_frame = None  # frame 0 of current scene (used as base for all subsequent frames)
    current_scene_id = None

    for entry in tqdm(pending, desc="Generating"):
        prompt_id = entry["prompt_id"]
        image_path = image_dir / f"{prompt_id}.png"
        scene_id = entry.get("scene_id")
        frame_idx = entry.get("frame_index", 0)

        # Determine seed
        if is_scene_mode:
            seed = entry.get("scene_seed", config.generation.seed)
        else:
            seed = config.generation.seed
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Encode prompts
        prompt_embeds, pooled, neg_embeds, neg_pooled = \
            _encode_and_pad(compel, entry["prompt"], entry["negative_prompt"])

        # Scene transition: reset base_frame when scene changes
        if scene_id != current_scene_id:
            base_frame = None
            current_scene_id = scene_id

        # Choose pipeline: text2img for frame 0, img2img from frame 0 for frames 1+
        if is_scene_mode and frame_idx > 0 and base_frame is not None:
            # img2img from frame 0 (NOT from previous frame — avoids cumulative degradation)
            image = i2i_pipe(
                image=base_frame,
                strength=strength,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=neg_embeds,
                negative_pooled_prompt_embeds=neg_pooled,
                num_inference_steps=config.generation.steps,
                guidance_scale=config.generation.guidance_scale,
                generator=generator,
            ).images[0]
        else:
            # Full text2img generation (frame 0 or independent mode)
            gen_kwargs = dict(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=neg_embeds,
                negative_pooled_prompt_embeds=neg_pooled,
                width=config.generation.width,
                height=config.generation.height,
                num_inference_steps=config.generation.steps,
                guidance_scale=config.generation.guidance_scale,
                generator=generator,
            )
            if entry.get("identity_image") is not None:
                gen_kwargs["ip_adapter_image"] = entry["identity_image"]

            image = t2i_pipe(**gen_kwargs).images[0]

        image.save(image_path)
        generated_paths.append(image_path)

        # Store frame 0 as base for all subsequent frames in this scene
        if frame_idx == 0:
            base_frame = image

        # Save metadata
        meta = {
            "prompt_id": prompt_id,
            "prompt": entry["prompt"],
            "negative_prompt": entry["negative_prompt"],
            "seed": seed,
            "steps": config.generation.steps,
            "guidance_scale": config.generation.guidance_scale,
            "model": config.generation.model,
            "slots": entry.get("slots", {}),
        }
        if is_scene_mode:
            meta["scene_id"] = scene_id
            meta["frame_index"] = frame_idx
            meta["scene_seed"] = entry.get("scene_seed")
            meta["img2img"] = frame_idx > 0
            meta["img2img_strength"] = strength if frame_idx > 0 else None

        (meta_dir / f"{prompt_id}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2)
        )

        completed_ids.add(prompt_id)
        _save_progress(run_dir, completed_ids, total)

    return generated_paths
