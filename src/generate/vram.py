"""VRAM management: load/unload model families to stay within 24GB."""

import gc
from contextlib import contextmanager
from pathlib import Path

import torch


def clear_vram():
    """Force-free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_text2img_pipeline(config):
    """Load SDXL pipelines for text-to-image generation.

    In scene mode, returns a dict with both text2img and img2img pipelines
    (sharing the same model components for VRAM efficiency).
    Otherwise returns just the text2img pipeline.
    """
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

    dtype = torch.float16 if config.generation.fp16 else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config.generation.model,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if config.generation.fp16 else None,
    )
    pipe = pipe.to("cuda")

    # Load IP-Adapter for face identity conditioning
    if config.scene.identity_enabled and config.ip_adapter.enabled:
        pipe.load_ip_adapter(
            config.ip_adapter.model_repo,
            subfolder=config.ip_adapter.subfolder,
            weight_name=config.ip_adapter.weight_name,
        )
        pipe.set_ip_adapter_scale(config.ip_adapter.scale)
        print(f"  IP-Adapter loaded (scale={config.ip_adapter.scale})")

    pipe.set_progress_bar_config(disable=True)

    if config.scene.enabled:
        # Create img2img pipeline sharing the same model components (no extra VRAM)
        img2img_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            text_encoder_2=pipe.text_encoder_2,
            tokenizer=pipe.tokenizer,
            tokenizer_2=pipe.tokenizer_2,
            unet=pipe.unet,
            scheduler=pipe.scheduler,
        )
        img2img_pipe.set_progress_bar_config(disable=True)
        return {"text2img": pipe, "img2img": img2img_pipe}

    return pipe


def load_conditioned_pipeline(config):
    """Load SDXL + ControlNet (+ optional IP-Adapter) pipeline."""
    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

    dtype = torch.float16 if config.generation.fp16 else torch.float32

    # Load ControlNet
    mode = config.controlnet.control_mode
    if mode == "canny":
        controlnet = ControlNetModel.from_pretrained(
            config.controlnet.canny_model, torch_dtype=dtype, use_safetensors=True,
        )
    elif mode == "openpose":
        controlnet = ControlNetModel.from_pretrained(
            config.controlnet.openpose_model, torch_dtype=dtype,
        )
    else:
        raise ValueError(f"Unsupported control_mode: {mode}")

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        config.generation.model,
        controlnet=controlnet,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if config.generation.fp16 else None,
    )
    pipe = pipe.to("cuda")

    # Optionally load IP-Adapter
    if config.ip_adapter.enabled:
        pipe.load_ip_adapter(
            config.ip_adapter.model_repo,
            subfolder=config.ip_adapter.subfolder,
            weight_name=config.ip_adapter.weight_name,
        )
        pipe.set_ip_adapter_scale(config.ip_adapter.scale)

    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_faceid_pipeline(config):
    """Load SDXL + IP-Adapter-FaceID pipeline with manual weight loading.

    Uses InsightFace embeddings (512-dim) instead of CLIP pixel features,
    which avoids grayscale bleed from grayscale reference images.
    Returns (pipe, image_proj) where image_proj is the FaceID projection module.
    """
    from diffusers import StableDiffusionXLPipeline
    from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file as load_safetensors
    import torch.nn as nn

    dtype = torch.float16 if config.generation.fp16 else torch.float32
    faceid_cfg = config.faceid

    # 1. Load base SDXL pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config.generation.model,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if config.generation.fp16 else None,
    )
    pipe = pipe.to("cuda")
    print(f"  Base model loaded: {config.generation.model}")

    # 2. Download FaceID weights
    lora_path = hf_hub_download(
        repo_id=faceid_cfg.lora_weights,
        filename=faceid_cfg.lora_weight_name,
    )
    ip_path = hf_hub_download(
        repo_id=faceid_cfg.ip_weights,
        filename=faceid_cfg.ip_weight_name,
    )
    print(f"  FaceID weights downloaded")

    # 3. Load and fuse LoRA weights
    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    print(f"  LoRA fused")

    # 4. Load IP-Adapter-FaceID state dict (nested: 'image_proj' and 'ip_adapter')
    raw_sd = torch.load(ip_path, map_location="cpu", weights_only=True)
    proj_sd = raw_sd["image_proj"]
    ip_sd = raw_sd["ip_adapter"]

    # 5. Build FaceID projection (512 → 4×2048)
    class FaceIDProjection(nn.Module):
        def __init__(self, embed_dim=512, hidden_dim=1024, output_dim=2048, num_tokens=4):
            super().__init__()
            self.num_tokens = num_tokens
            self.output_dim = output_dim
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim * num_tokens),
            )
            self.norm = nn.LayerNorm(output_dim)

        def forward(self, x):
            x = self.proj(x).reshape(-1, self.num_tokens, self.output_dim)
            return self.norm(x)

    image_proj = FaceIDProjection(num_tokens=faceid_cfg.num_tokens)
    image_proj.load_state_dict(proj_sd)
    image_proj = image_proj.to(dtype=dtype, device="cuda")
    print(f"  FaceID projection loaded")

    # 6. Set up IP cross-attention processors on UNet
    # ip_sd has interleaved keys: even indices = LoRA (already fused), odd indices = IP cross-attn
    ip_keys = {k for k in ip_sd if ".to_k_ip." in k or ".to_v_ip." in k}
    ip_indices = sorted(set(int(k.split(".")[0]) for k in ip_keys))

    attn_procs = {}
    unet = pipe.unet
    idx = 0
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if cross_attention_dim is None:
            attn_procs[name] = unet.attn_processors[name]
            continue

        # Get hidden size from the corresponding attention layer
        parts = name.replace(".processor", "").split(".")
        layer = unet
        for p in parts:
            layer = getattr(layer, p) if not p.isdigit() else layer[int(p)]
        hidden_size = layer.to_q.weight.shape[0]

        proc = IPAdapterAttnProcessor2_0(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            num_tokens=[faceid_cfg.num_tokens],
            scale=[faceid_cfg.scale],
        )

        if idx < len(ip_indices):
            ip_idx = ip_indices[idx]
            k_weight = ip_sd[f"{ip_idx}.to_k_ip.weight"].to(dtype=dtype)
            v_weight = ip_sd[f"{ip_idx}.to_v_ip.weight"].to(dtype=dtype)
            proc.to_k_ip[0].weight = nn.Parameter(k_weight)
            proc.to_v_ip[0].weight = nn.Parameter(v_weight)

        attn_procs[name] = proc.to(device="cuda", dtype=dtype)
        idx += 1

    unet.set_attn_processor(attn_procs)
    print(f"  IP cross-attention set up ({idx} layers)")

    # 7. Passthrough encoder_hid_proj (embeddings are pre-projected)
    class PassthroughImageProj(nn.Module):
        def forward(self, x):
            if isinstance(x, list):
                return x[0]
            return x

    unet.encoder_hid_proj = PassthroughImageProj()
    unet.config.encoder_hid_dim_type = "ip_image_proj"

    pipe.set_progress_bar_config(disable=True)
    return pipe, image_proj


@contextmanager
def model_scope(stage: str, config):
    """Context manager: load models for a stage, yield, then fully unload.

    Stages: 'text2img', 'conditioned', 'faceid', 'annotate', 'validate'
    """
    clear_vram()
    model = None
    try:
        if stage == "text2img":
            model = load_text2img_pipeline(config)
        elif stage == "conditioned":
            model = load_conditioned_pipeline(config)
        elif stage == "faceid":
            model = load_faceid_pipeline(config)
        elif stage == "annotate":
            model = _load_annotation_models(config)
        elif stage == "validate":
            model = _load_validation_models(config)
        else:
            raise ValueError(f"Unknown stage: {stage}")
        yield model
    finally:
        del model
        clear_vram()


def _load_annotation_models(config):
    """Load all annotation models (they fit together in ~6.5GB)."""
    models = {}

    # Grounding DINO (via transformers AutoModel — avoids groundingdino-py compat issues)
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    dino_processor = AutoProcessor.from_pretrained(config.annotation.dino_model)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        config.annotation.dino_model
    ).to("cuda")
    models["dino"] = dino_model
    models["dino_processor"] = dino_processor

    # SAM2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from huggingface_hub import hf_hub_download
    sam2_ckpt = hf_hub_download(config.annotation.sam2_model, filename="sam2_hiera_large.pt")
    sam2_cfg = "sam2_hiera_l.yaml"
    sam2 = build_sam2(sam2_cfg, sam2_ckpt, device="cuda")
    models["sam2"] = SAM2ImagePredictor(sam2)

    # YOLOv8
    from ultralytics import YOLO
    models["yolo"] = YOLO(config.annotation.yolo_model)

    # ArcFace
    from insightface.app import FaceAnalysis
    face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(config.annotation.face_det_size,) * 2)
    models["face"] = face_app

    return models


def _load_validation_models(config):
    """Load validation models."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.validation.clip_model,
        pretrained=config.validation.clip_pretrained,
    )
    model = model.to("cuda").eval()
    tokenizer = open_clip.get_tokenizer(config.validation.clip_model)
    return {"clip_model": model, "clip_preprocess": preprocess, "clip_tokenizer": tokenizer}
