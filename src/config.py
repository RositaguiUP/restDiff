from dataclasses import dataclass, field, asdict
from typing import List, Optional

@dataclass
class OptimizerConfig:
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 1.25e-4 # 2.5e-3 / 20
    
@dataclass
class ScheduleConfig:
    # How long to keep the initial weights before decaying
    hold_steps: int = 10000 
    # How many steps it takes to reach the final weights
    decay_steps: int = 20000 
    
    # Starting weights (Phase 1: Geometry Focus)
    depth_start: float = 0.3
    rgb_start: float = 0.7
    
    # Ending weights (Phase 2: Texture Focus)
    depth_end: float = 0.02
    rgb_end: float = 0.98

@dataclass
class GuidanceConfig:
    # sd_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    # tile_path: str = "lllyasviel/control_v11f1e_sd15_tile"
    # depth_path: str = "lllyasviel/control_v11f1p_sd15_depth"
    # guidance_scale: float = 7.5
    # controlnet_scales: List[float] = field(default_factory=lambda: [0.85, 1.0])
    # ip_adapter_scale: float = 0.5
    # num_steps_sample: int = 20
    # min_step_percent: float = 0.05
    # max_step_percent: float = 0.4
        # Use a photorealistic SDXL fine-tune
    sd_path: str = "SG161222/RealVisXL_V4.0" 
    # SDXL specific ControlNets
    tile_path: str = "xinsir/controlnet-tile-sdxl-1.0"
    depth_path: str = "diffusers/controlnet-depth-sdxl-1.0"
    
    guidance_scale: float = 5.0 # SDXL usually needs lower guidance (4.0 - 5.0)
    controlnet_scales: List[float] = field(default_factory=lambda: [0.6, 0.8]) # Lower slightly to prevent burning
    ip_adapter_scale: float = 0.5
    num_steps_sample: int = 20
    min_step_percent: float = 0.05
    max_step_percent: float = 0.4
    
    

@dataclass
class PipelineConfig:
    # Project Identity
    scene_name: str = "my_scene"
    floor_number: int = 1
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
    
    # --- Evaluation & File I/O Parameters ---
    run_eval: bool = True  # Toggle evaluation during training
    test_every: int = 8
    eval_steps: List[int] = field(default_factory=lambda: [14999, 29999])
    use_color_correction_metric: bool = True
    color_correct_method: str = "affine" # "affine" or "quadratic"
    
    # --- PLY Export ---
    save_ply: bool = False
    ply_steps: List[int] = field(default_factory=lambda: [14999, 29999])
    render_traj_path: str = "ellipse"
    
    # --- Densification Strategy ---
    strategy_type: str = "default" # "default" or "mcmc"
    init_opa: float = 0.1
    init_scale: float = 1.0
    opacity_reg: float = 0.0
    scale_reg: float = 0.0
    
    mcmc_cap_max: int = 5_000_000
    refine_stop_iter: int = 15000
    
    # Loss Optimization Coefficients
    lambda_ssim: float = 0.2
    
    # Updated Distillation Weights
    lambda_distill_rgb: float = 15.0     # Very low: Stop fighting the deblurring process
    lambda_distill_depth: float = 3.0   # High: Trust the LiDAR constraints
    lambda_distill_mse: float = 1.0     # Baseline pixel matching
    lambda_distill_lpips: float = 1.25   # High: Force photorealistic, sharp textures
    
    # Embedded Sub-configs
    opts: OptimizerConfig = field(default_factory=OptimizerConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    
    def __post_init__(self):
        """Automatically apply paper hyper-parameters if MCMC is selected"""
        if self.strategy_type == "mcmc":
            self.init_opa = 0.5
            self.init_scale = 0.1
            self.opacity_reg = 0.01
            self.scale_reg = 0.01
    
    def to_dict(self):
            """Converts dataclass to dict for WandB auto-logging"""
            return asdict(self)