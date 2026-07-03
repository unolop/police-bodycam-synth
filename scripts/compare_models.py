#!/usr/bin/env python3
"""Compare RealVisXL (GPU 0) vs FLUX.1-dev (GPU 1) on the same scene prompts."""

import json
import sys
import torch
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_realvis(prompts_data, output_dir, gpu_id=0):
    """Run RealVisXL on specified GPU."""
    import torch
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
    from compel import Compel, ReturnedEmbeddingsType

    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading RealVisXL V5.0...")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    # img2img pipeline sharing components
    i2i_pipe = StableDiffusionXLImg2ImgPipeline(
        vae=pipe.vae, text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2, tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2, unet=pipe.unet, scheduler=pipe.scheduler,
    )
    i2i_pipe.set_progress_bar_config(disable=True)

    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
        truncate_long_prompts=False,
    )

    img_dir = Path(output_dir) / "realvis" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    base_frame = None
    for entry in prompts_data:
        pid = entry["prompt_id"]
        frame_idx = entry.get("frame_index", 0)
        seed = entry.get("scene_seed", 42)
        gen = torch.Generator(device=device).manual_seed(seed)

        prompt_embeds, pooled = compel(entry["prompt"])
        neg_embeds, neg_pooled = compel(entry["negative_prompt"])

        # Pad
        ml = max(prompt_embeds.shape[1], neg_embeds.shape[1])
        if prompt_embeds.shape[1] < ml:
            pad = torch.zeros(1, ml - prompt_embeds.shape[1], prompt_embeds.shape[2],
                              device=prompt_embeds.device, dtype=prompt_embeds.dtype)
            prompt_embeds = torch.cat([prompt_embeds, pad], dim=1)
        if neg_embeds.shape[1] < ml:
            pad = torch.zeros(1, ml - neg_embeds.shape[1], neg_embeds.shape[2],
                              device=neg_embeds.device, dtype=neg_embeds.dtype)
            neg_embeds = torch.cat([neg_embeds, pad], dim=1)

        if frame_idx > 0 and base_frame is not None:
            image = i2i_pipe(
                image=base_frame, strength=0.35,
                prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled,
                negative_prompt_embeds=neg_embeds, negative_pooled_prompt_embeds=neg_pooled,
                num_inference_steps=30, guidance_scale=6.0, generator=gen,
            ).images[0]
        else:
            image = pipe(
                prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled,
                negative_prompt_embeds=neg_embeds, negative_pooled_prompt_embeds=neg_pooled,
                width=1024, height=1024,
                num_inference_steps=30, guidance_scale=6.0, generator=gen,
            ).images[0]

        image.save(img_dir / f"{pid}.png")
        if frame_idx == 0:
            base_frame = image
        print(f"[GPU {gpu_id}] RealVisXL: {pid} done")

    print(f"[GPU {gpu_id}] RealVisXL complete: {len(prompts_data)} images")


def generate_sd35(prompts_data, output_dir, gpu_id=1):
    """Run Stable Diffusion 3.5 Medium on specified GPU."""
    import torch
    from diffusers import StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline

    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading SD 3.5 Medium...")

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=torch.float16,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    # img2img sharing components
    i2i_pipe = StableDiffusion3Img2ImgPipeline(
        transformer=pipe.transformer, scheduler=pipe.scheduler,
        vae=pipe.vae, text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2, text_encoder_3=pipe.text_encoder_3,
        tokenizer=pipe.tokenizer, tokenizer_2=pipe.tokenizer_2,
        tokenizer_3=pipe.tokenizer_3,
    )
    i2i_pipe.set_progress_bar_config(disable=True)

    img_dir = Path(output_dir) / "sd35" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    base_frame = None
    for entry in prompts_data:
        pid = entry["prompt_id"]
        frame_idx = entry.get("frame_index", 0)
        seed = entry.get("scene_seed", 42)
        gen = torch.Generator(device=device).manual_seed(seed)

        prompt = entry["prompt"]
        neg = entry["negative_prompt"]

        if frame_idx > 0 and base_frame is not None:
            image = i2i_pipe(
                image=base_frame, strength=0.35,
                prompt=prompt, negative_prompt=neg,
                num_inference_steps=28, guidance_scale=5.0, generator=gen,
            ).images[0]
        else:
            image = pipe(
                prompt=prompt, negative_prompt=neg,
                width=1024, height=1024,
                num_inference_steps=28, guidance_scale=5.0, generator=gen,
            ).images[0]

        image.save(img_dir / f"{pid}.png")
        if frame_idx == 0:
            base_frame = image
        print(f"[GPU {gpu_id}] SD3.5: {pid} done")

    print(f"[GPU {gpu_id}] SD3.5 complete: {len(prompts_data)} images")


if __name__ == "__main__":
    from src.prompts.scene_planner import generate_scene_prompts

    output_dir = "output/model_compare"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate 1 scene × 5 frames for scenario 1
    prompts = list(generate_scene_prompts(scenario=1, num_scenes=1, frames_per_scene=5, seed=42))
    print(f"Generated {len(prompts)} prompts for comparison")

    # Save prompts
    with open(f"{output_dir}/prompts.json", "w") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    # Run both models in parallel on separate GPUs
    p0 = mp.Process(target=generate_realvis, args=(prompts, output_dir, 0))
    p1 = mp.Process(target=generate_sd35, args=(prompts, output_dir, 1))

    print("Starting parallel generation...")
    p0.start()
    p1.start()
    p0.join()
    p1.join()

    print(f"\nDone! Compare results in:")
    print(f"  RealVisXL: {output_dir}/realvis/images/")
    print(f"  FLUX:      {output_dir}/flux/images/")
