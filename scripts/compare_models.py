#!/usr/bin/env python3
"""Model comparison: RealVisXL V5.0 vs Juggernaut XL v9 with IP-Adapter-FaceID.

Runs identical prompts, seeds, and face IDs through both models.
Outputs per-model images + side-by-side comparison grid.

Usage:
    python scripts/compare_models.py --num-scenes 3
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prompts.templates import (
    STYLE_PREFIX, NEGATIVE_PROMPT, TEMPLATE_POI,
    LOCATIONS, TIME_OF_DAY, WEATHER, CROWD_DENSITY,
    SUBJECT_DISTANCES, SUBJECT_DESCRIPTIONS, SUBJECT_ACTIONS,
    ADDITIONAL_OBJECTS, LIGHTING_DETAILS,
)

MODELS = {
    "realvisxl": "SG161222/RealVisXL_V5.0",
    "juggernaut": "RunDiffusion/Juggernaut-XL-v9",
}

# Extended negative prompt — suppress extra limbs
NEGATIVE_PROMPT_EXT = (
    NEGATIVE_PROMPT
    + ", extra arms, extra limbs, three arms, deformed arms, "
    + "mutated hands, extra hands, fused fingers, missing fingers, bad hands"
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


def load_identities(ref_dir: Path) -> list[dict]:
    emb_files = sorted(ref_dir.glob("*.npy"))
    if not emb_files:
        raise FileNotFoundError(f"No .npy embeddings found in {ref_dir}")
    ids = [{"name": f.stem, "embedding": np.load(f)} for f in emb_files]
    print(f"Loaded {len(ids)} face identities from {ref_dir}")
    return ids


def load_pipeline(model_id: str):
    from diffusers import StableDiffusionXLPipeline
    from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
    from huggingface_hub import hf_hub_download
    import torch.nn as nn

    dtype, device = torch.float16, "cuda"

    print(f"  Loading {model_id}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=dtype, use_safetensors=True, variant="fp16",
    ).to(device)

    lora_path = hf_hub_download("h94/IP-Adapter-FaceID", "ip-adapter-faceid_sdxl_lora.safetensors")
    ip_path   = hf_hub_download("h94/IP-Adapter-FaceID", "ip-adapter-faceid_sdxl.bin")

    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora()
    pipe.unload_lora_weights()

    raw_sd  = torch.load(ip_path, map_location="cpu", weights_only=True)
    proj_sd = raw_sd["image_proj"]
    ip_sd   = raw_sd["ip_adapter"]
    num_tokens = 4

    class FaceIDProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_tokens = num_tokens
            self.output_dim = 2048
            self.proj = nn.Sequential(
                nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, 2048 * num_tokens),
            )
            self.norm = nn.LayerNorm(2048)
        def forward(self, x):
            return self.norm(self.proj(x).reshape(-1, self.num_tokens, self.output_dim))

    image_proj = FaceIDProjection()
    image_proj.load_state_dict(proj_sd)
    image_proj = image_proj.to(dtype=dtype, device=device)

    ip_indices = sorted(set(int(k.split(".")[0]) for k in ip_sd if ".to_k_ip." in k))
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
        proc = IPAdapterAttnProcessor2_0(
            hidden_size=layer.to_q.weight.shape[0],
            cross_attention_dim=cross_dim,
            num_tokens=[num_tokens], scale=[0.3],
        )
        if idx < len(ip_indices):
            ip_i = ip_indices[idx]
            proc.to_k_ip[0].weight = nn.Parameter(ip_sd[f"{ip_i}.to_k_ip.weight"].to(dtype))
            proc.to_v_ip[0].weight = nn.Parameter(ip_sd[f"{ip_i}.to_v_ip.weight"].to(dtype))
        attn_procs[name] = proc.to(device=device, dtype=dtype)
        idx += 1

    pipe.unet.set_attn_processor(attn_procs)

    class PassthroughProj(nn.Module):
        def forward(self, x): return x[0] if isinstance(x, list) else x

    pipe.unet.encoder_hid_proj = PassthroughProj()
    pipe.unet.config.encoder_hid_dim_type = "ip_image_proj"
    return pipe, image_proj


def generate_one(pipe, image_proj, prompt, identity, seed):
    dtype, device = torch.float16, "cuda"
    emb = torch.tensor(identity["embedding"], dtype=dtype, device=device).unsqueeze(0)
    face_cond = image_proj(emb).unsqueeze(0)
    ip_embeds = [torch.cat([torch.zeros_like(face_cond), face_cond])]
    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT_EXT,
        ip_adapter_image_embeds=ip_embeds,
        num_inference_steps=40,
        guidance_scale=7.5,
        width=1024, height=1024,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    return result.images[0]


def make_grid(rows, out_path):
    """Layout: [Face ID | RealVisXL | Juggernaut XL] per row."""
    thumb, header_h = 300, 40
    W = thumb * 3
    H = header_h + thumb * len(rows)
    grid = Image.new("RGB", (W, H), (20, 20, 20))
    draw = ImageDraw.Draw(grid)
    for col, label in enumerate(["Face ID (DeceiveD)", "RealVisXL V5.0", "Juggernaut XL v9"]):
        draw.text((col * thumb + thumb // 2, header_h // 2), label, fill=(220, 220, 100), anchor="mm")
    for i, row in enumerate(rows):
        y = header_h + i * thumb
        for col, key in enumerate(["face", "realvisxl", "juggernaut"]):
            grid.paste(row[key].resize((thumb, thumb)), (col * thumb, y))
        draw.text((6, y + 6), row["label"], fill=(255, 255, 80))
    grid.save(out_path)
    print(f"Saved grid: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-scenes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--identity-ref-dir", type=str,
                        default=str(PROJECT_ROOT / "data/celeb-k/face_pool_controlled/selected"))
    parser.add_argument("--out-dir", type=str,
                        default=str(PROJECT_ROOT / "output/model_comparison"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    face_dir = PROJECT_ROOT / "data/celeb-k/face_pool_controlled/generated"

    identities = load_identities(Path(args.identity_ref_dir))
    scenes = [
        {"seed": args.seed + i * 100, "prompt": build_prompt(args.seed + i * 100),
         "identity": identities[i % len(identities)]}
        for i in range(args.num_scenes)
    ]

    results = {name: [] for name in MODELS}

    for model_name, model_id in MODELS.items():
        print(f"\n{'='*55}\nModel: {model_name}\n{'='*55}")
        pipe, image_proj = load_pipeline(model_id)
        for i, scene in enumerate(scenes):
            print(f"  Scene {i+1}/{args.num_scenes} | ID: {scene['identity']['name']} | seed: {scene['seed']}")
            img = generate_one(pipe, image_proj, scene["prompt"], scene["identity"], scene["seed"])
            fname = out_dir / f"{model_name}_scene{i:03d}_{scene['identity']['name']}.png"
            img.save(fname)
            results[model_name].append(img)
        del pipe, image_proj
        torch.cuda.empty_cache()
        print("  VRAM cleared.")

    print("\nBuilding comparison grid...")
    rows = []
    for i, scene in enumerate(scenes):
        face_path = face_dir / f"{scene['identity']['name']}.png"
        face_img = Image.open(face_path) if face_path.exists() else Image.new("RGB", (300, 300), (60, 60, 60))
        rows.append({
            "label": scene['identity']['name'],
            "face":       face_img,
            "realvisxl":  results["realvisxl"][i],
            "juggernaut": results["juggernaut"][i],
        })
    make_grid(rows, out_dir / "comparison_grid.png")
    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
