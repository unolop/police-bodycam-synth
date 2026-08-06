#!/usr/bin/env python3
"""Quick sample generation: bodycam scenes with IP-Adapter-FaceID.

Generates 3 scenes × 5 frames = 15 images using selected face identities.
Use this to verify FaceID conditioning quality before full-scale generation.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/sample_faceid_generation.py
    CUDA_VISIBLE_DEVICES=1 python scripts/sample_faceid_generation.py --num-scenes 5 --frames-per-scene 5
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prompts.templates import (
    STYLE_PREFIX, NEGATIVE_PROMPT, TEMPLATE_POI,
    LOCATIONS, TIME_OF_DAY, WEATHER, CROWD_DENSITY,
    SUBJECT_DISTANCES, SUBJECT_DESCRIPTIONS, SUBJECT_ACTIONS,
    ADDITIONAL_OBJECTS, LIGHTING_DETAILS,
)


def build_prompt(seed: int) -> str:
    rng = random.Random(seed)
    return TEMPLATE_POI.format(
        style_prefix=STYLE_PREFIX,
        subject_description=rng.choice(SUBJECT_DESCRIPTIONS),
        subject_distance=rng.choice(SUBJECT_DISTANCES),
        subject_action=rng.choice(SUBJECT_ACTIONS),
        location=rng.choice(LOCATIONS),
        time_of_day=rng.choice(TIME_OF_DAY),
        weather=rng.choice(WEATHER),
        crowd_density=rng.choice(CROWD_DENSITY),
        additional_objects=rng.choice(ADDITIONAL_OBJECTS),
        lighting_detail=rng.choice(LIGHTING_DETAILS),
    )


def load_identity_embeddings(ref_dir: Path) -> list[dict]:
    """Load ArcFace embeddings from selected/ directory."""
    emb_files = sorted(ref_dir.glob("*.npy"))
    if not emb_files:
        raise FileNotFoundError(f"No .npy embeddings found in {ref_dir}")

    identities = []
    for f in emb_files:
        emb = np.load(f)
        identities.append({"name": f.stem, "embedding": emb})
    print(f"Loaded {len(identities)} face identities from {ref_dir}")
    return identities


def load_faceid_pipeline(config):
    """Load RealVisXL + IP-Adapter-FaceID."""
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
    from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
    from huggingface_hub import hf_hub_download
    import torch.nn as nn

    dtype = torch.float16
    device = "cuda"

    print("Loading RealVisXL V5.0...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0",
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16",
    ).to(device)

    print("Downloading IP-Adapter-FaceID weights...")
    lora_path = hf_hub_download(
        repo_id="h94/IP-Adapter-FaceID",
        filename="ip-adapter-faceid_sdxl_lora.safetensors",
    )
    ip_path = hf_hub_download(
        repo_id="h94/IP-Adapter-FaceID",
        filename="ip-adapter-faceid_sdxl.bin",
    )

    print("Fusing LoRA...")
    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora()
    pipe.unload_lora_weights()

    print("Loading IP-Adapter weights...")
    raw_sd = torch.load(ip_path, map_location="cpu", weights_only=True)
    proj_sd = raw_sd["image_proj"]
    ip_sd = raw_sd["ip_adapter"]

    num_tokens = 4

    class FaceIDProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_tokens = num_tokens
            self.output_dim = 2048
            self.proj = nn.Sequential(
                nn.Linear(512, 1024),
                nn.GELU(),
                nn.Linear(1024, 2048 * num_tokens),
            )
            self.norm = nn.LayerNorm(2048)

        def forward(self, x):
            return self.norm(self.proj(x).reshape(-1, self.num_tokens, self.output_dim))

    image_proj = FaceIDProjection()
    image_proj.load_state_dict(proj_sd)
    image_proj = image_proj.to(dtype=dtype, device=device)

    # Set up IP cross-attention processors
    ip_keys = {k for k in ip_sd if ".to_k_ip." in k or ".to_v_ip." in k}
    ip_indices = sorted(set(int(k.split(".")[0]) for k in ip_keys))
    attn_procs, idx = {}, 0

    for name in pipe.unet.attn_processors.keys():
        cross_dim = None if name.endswith("attn1.processor") else pipe.unet.config.cross_attention_dim
        if cross_dim is None:
            attn_procs[name] = pipe.unet.attn_processors[name]
            continue
        parts = name.replace(".processor", "").split(".")
        layer = pipe.unet
        for p in parts:
            layer = getattr(layer, p) if not p.isdigit() else layer[int(p)]
        hidden_size = layer.to_q.weight.shape[0]
        proc = IPAdapterAttnProcessor2_0(
            hidden_size=hidden_size,
            cross_attention_dim=cross_dim,
            num_tokens=[num_tokens],
            scale=[0.3],
        )
        if idx < len(ip_indices):
            ip_idx = ip_indices[idx]
            proc.to_k_ip[0].weight = nn.Parameter(ip_sd[f"{ip_idx}.to_k_ip.weight"].to(dtype))
            proc.to_v_ip[0].weight = nn.Parameter(ip_sd[f"{ip_idx}.to_v_ip.weight"].to(dtype))
        attn_procs[name] = proc.to(device=device, dtype=dtype)
        idx += 1

    pipe.unet.set_attn_processor(attn_procs)

    class PassthroughProj(nn.Module):
        def forward(self, x):
            return x[0] if isinstance(x, list) else x

    pipe.unet.encoder_hid_proj = PassthroughProj()
    pipe.unet.config.encoder_hid_dim_type = "ip_image_proj"
    pipe.set_progress_bar_config(disable=False)

    # img2img pipeline sharing same components
    img2img = StableDiffusionXLImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
    )
    img2img.set_progress_bar_config(disable=True)

    print(f"Pipeline ready ({idx} IP cross-attention layers)")
    return pipe, img2img, image_proj


def generate_scene(pipe, img2img, image_proj, prompt, identity, scene_seed, frames, strength, out_dir, scene_id):
    """Generate one scene: frame 0 text2img, frames 1+ img2img from frame 0."""
    device, dtype = "cuda", torch.float16

    # Prepare face embedding
    emb = torch.tensor(identity["embedding"], dtype=dtype, device=device).unsqueeze(0)
    face_cond = image_proj(emb).unsqueeze(0)  # (1, 1, num_tokens, 2048)
    # Classifier-free guidance requires [negative_embed, positive_embed]
    ip_embeds = [torch.cat([torch.zeros_like(face_cond), face_cond])]  # (2, 1, num_tokens, 2048)

    # Frame 0: text2img
    generator = torch.Generator(device=device).manual_seed(scene_seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        ip_adapter_image_embeds=ip_embeds,
        num_inference_steps=30,
        guidance_scale=7.5,
        width=1024, height=1024,
        generator=generator,
    )
    base_frame = result.images[0]
    base_frame.save(out_dir / f"scene{scene_id:03d}_frame00_{identity['name']}.png")

    # Frames 1+: img2img from base_frame
    for fi in range(1, frames):
        result = img2img(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=base_frame,
            ip_adapter_image_embeds=ip_embeds,
            strength=strength,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=torch.Generator(device=device).manual_seed(scene_seed + fi),
        )
        result.images[0].save(out_dir / f"scene{scene_id:03d}_frame{fi:02d}_{identity['name']}.png")

    return base_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-scenes", type=int, default=3)
    parser.add_argument("--frames-per-scene", type=int, default=5)
    parser.add_argument("--img2img-strength", type=float, default=0.35)
    parser.add_argument("--identity-ref-dir", type=str,
                        default=str(PROJECT_ROOT / "data/celeb-k/face_pool_controlled/selected"))
    parser.add_argument("--out-dir", type=str,
                        default=str(PROJECT_ROOT / "output/sample_faceid"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load identities
    identities = load_identity_embeddings(Path(args.identity_ref_dir))

    # Load pipeline
    pipe, img2img, image_proj = load_faceid_pipeline(None)

    print(f"\nGenerating {args.num_scenes} scenes × {args.frames_per_scene} frames "
          f"= {args.num_scenes * args.frames_per_scene} images")
    print(f"Output: {out_dir}\n")

    rng = random.Random(args.seed)
    for scene_id in range(args.num_scenes):
        scene_seed = args.seed + scene_id * 100
        prompt = build_prompt(scene_seed)
        identity = identities[scene_id % len(identities)]

        print(f"Scene {scene_id+1}/{args.num_scenes} | ID: {identity['name']} | seed: {scene_seed}")
        print(f"  Prompt: {prompt[:120]}...")

        generate_scene(
            pipe, img2img, image_proj,
            prompt=prompt,
            identity=identity,
            scene_seed=scene_seed,
            frames=args.frames_per_scene,
            strength=args.img2img_strength,
            out_dir=out_dir,
            scene_id=scene_id,
        )
        print(f"  Saved {args.frames_per_scene} frames")

    print(f"\nDone. {args.num_scenes * args.frames_per_scene} images saved to {out_dir}")


if __name__ == "__main__":
    main()
