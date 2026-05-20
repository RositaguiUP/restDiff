from dataclasses import dataclass, field, asdict
from typing import List

@dataclass
class OptimizerConfig:
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 1.25e-4 # 2.5e-3 / 20

@dataclass
class GuidanceConfig:
    sd_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    tile_path: str = "lllyasviel/control_v11f1e_sd15_tile"
    depth_path: str = "lllyasviel/control_v11f1p_sd15_depth"
    guidance_scale: float = 7.5
    controlnet_scales: List[float] = field(default_factory=lambda: [0.5, 1.0])
    ip_adapter_scale: float = 0.5
    num_steps_sample: int = 20
    min_step_percent: float = 0.25
    max_step_percent: float = 0.98

@dataclass
class PipelineConfig:
    # Project Identity
    scene_name: str = "my_scene"
    version: str = "v1"
    data_dir: str = "data/my_scene"
    device: str = "cuda"
    sh_degree: int = 3
    
    warmup_version: str = "v1"
    
    # Execution Budgets
    max_steps_warmup: int = 30000
    max_steps_distill: int = 15000
    ckpt_interval: int = 2500
    log_interval: int = 50
    vis_interval: int = 500
    
    # Loss Optimization Coefficients
    lambda_ssim: float = 0.2
    
    lambda_warmup_rgb: float = 1.0
    lambda_warmup_depth: float = 0.5
    
    lambda_distill_l1: float = 0.5
    lambda_distill_ssim: float = 0.2
    lambda_distill_depth: float = 10.0
    lambda_distill_mse: float = 100.0
    lambda_distill_lpips: float = 10.0
    
    # Embedded Sub-configs
    opts: OptimizerConfig = field(default_factory=OptimizerConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    
    def to_dict(self):
            """Converts dataclass to dict for WandB auto-logging"""
            return asdict(self)