"""Configuration system: YAML → dataclass loader with overlay support."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class GenerationConfig:
    model: str = "SG161222/RealVisXL_V5.0"
    steps: int = 30
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 1024
    batch_size: int = 1
    seed: int = 42
    fp16: bool = True


@dataclass
class ControlNetConfig:
    canny_model: str = "diffusers/controlnet-canny-sdxl-1.0"
    openpose_model: str = "thibaud/controlnet-openpose-sdxl-1.0"
    conditioning_scale: float = 0.5
    control_mode: str = "canny"


@dataclass
class IPAdapterConfig:
    enabled: bool = False
    model_repo: str = "h94/IP-Adapter"
    subfolder: str = "sdxl_models"
    weight_name: str = "ip-adapter_sdxl_vit-h.safetensors"
    scale: float = 0.6


@dataclass
class FaceIDConfig:
    ip_weights: str = "h94/IP-Adapter-FaceID"  # HF repo for ip-adapter-faceid_sdxl.bin
    ip_weight_name: str = "ip-adapter-faceid_sdxl.bin"
    lora_weights: str = "h94/IP-Adapter-FaceID"  # HF repo for LoRA
    lora_weight_name: str = "ip-adapter-faceid_sdxl_lora.safetensors"
    scale: float = 0.3
    num_tokens: int = 4  # projection tokens
    insightface_model: str = "buffalo_l"


@dataclass
class SceneConfig:
    enabled: bool = True
    frames_per_scene: int = 10
    num_scenes: int = 840  # 840 scenes × 10 frames = 8400 images
    # Per-frame variation: how much the scene changes between frames
    pose_variation: float = 0.3  # 0=identical, 1=completely different pose/action
    img2img_strength: float = 0.35  # denoising strength for frames 1+ (lower = more consistent)
    # Identity consistency
    identity_enabled: bool = False
    num_identities: int = 20  # pool of unique face identities
    identity_ref_dir: Optional[str] = None  # dir of reference face images


@dataclass
class PromptConfig:
    num_prompts: int = 100
    diversity_seed: int = 42


@dataclass
class ExtractionConfig:
    video_dir: Optional[str] = None
    frames_per_cut: int = 1
    output_dir: Optional[str] = None


@dataclass
class AnnotationConfig:
    dino_model: str = "IDEA-Research/grounding-dino-base"
    dino_box_threshold: float = 0.3
    dino_text_threshold: float = 0.25
    sam2_model: str = "facebook/sam2-hiera-large"
    yolo_model: str = "yolov8x.pt"
    yolo_conf: float = 0.25
    face_det_size: int = 640


@dataclass
class ValidationConfig:
    clip_model: str = "ViT-H-14"
    clip_pretrained: str = "laion2b_s32b_b79k"
    min_clip_score: float = 0.2
    fid_real_path: Optional[str] = None
    # SSIM/LPIPS safety thresholds (above = too similar to reference → re-identification risk)
    ssim_safety_threshold: float = 0.6
    lpips_safety_threshold: float = 0.3  # LPIPS < 0.3 means perceptually too similar
    # Face embedding quality
    min_face_detection_rate: float = 0.85
    min_embedding_cosine_spread: float = 0.3  # min std of pairwise cosine similarities


@dataclass
class PipelineConfig:
    mode: str = "text2img"
    scenario: int = 1
    output_dir: str = "output"
    run_id: Optional[str] = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    controlnet: ControlNetConfig = field(default_factory=ControlNetConfig)
    ip_adapter: IPAdapterConfig = field(default_factory=IPAdapterConfig)
    faceid: FaceIDConfig = field(default_factory=FaceIDConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    @property
    def run_dir(self) -> Path:
        if self.run_id is None:
            from datetime import datetime
            model_short = self.generation.model.split("/")[-1].replace("-", "_")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"{model_short}_s{self.scenario}_n{self.prompt.num_prompts}_{ts}"
        return Path(self.output_dir) / self.run_id


def _deep_update(base: dict, overlay: dict) -> dict:
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _dict_to_dataclass(dc_class, data: dict):
    if not isinstance(data, dict):
        return data
    filtered = {}
    for f in dc_class.__dataclass_fields__:
        if f in data:
            ft = dc_class.__dataclass_fields__[f].type
            # Check if field type is itself a dataclass
            if hasattr(ft, '__dataclass_fields__'):
                filtered[f] = _dict_to_dataclass(ft, data[f])
            else:
                filtered[f] = data[f]
    return dc_class(**filtered)


def load_config(*yaml_paths: str, overrides: Optional[dict] = None) -> PipelineConfig:
    """Load config from YAML files (later files overlay earlier), then apply overrides."""
    merged = {}
    for path in yaml_paths:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        _deep_update(merged, data)
    if overrides:
        _deep_update(merged, overrides)

    # Build nested dataclasses
    cfg = PipelineConfig(
        mode=merged.get("mode", "text2img"),
        scenario=merged.get("scenario", 1),
        output_dir=merged.get("output_dir", "output"),
        run_id=merged.get("run_id"),
        generation=_dict_to_dataclass(GenerationConfig, merged.get("generation", {})),
        controlnet=_dict_to_dataclass(ControlNetConfig, merged.get("controlnet", {})),
        ip_adapter=_dict_to_dataclass(IPAdapterConfig, merged.get("ip_adapter", {})),
        faceid=_dict_to_dataclass(FaceIDConfig, merged.get("faceid", {})),
        prompt=_dict_to_dataclass(PromptConfig, merged.get("prompt", {})),
        scene=_dict_to_dataclass(SceneConfig, merged.get("scene", {})),
        extraction=_dict_to_dataclass(ExtractionConfig, merged.get("extraction", {})),
        annotation=_dict_to_dataclass(AnnotationConfig, merged.get("annotation", {})),
        validation=_dict_to_dataclass(ValidationConfig, merged.get("validation", {})),
    )
    return cfg
