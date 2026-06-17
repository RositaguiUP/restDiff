import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    DDIMScheduler,
    EulerDiscreteScheduler 
)

from src.config import GuidanceConfig

class StableDiffusionControlNetGuidance(nn.Module):
    def __init__(self, cfg: GuidanceConfig, device: str = "cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.weights_dtype = torch.float16 # FP16 for VRAM efficiency

        print("[INFO] Loading ControlNets (Tile + Depth)...")
        controlnet_tile = ControlNetModel.from_pretrained(cfg.tile_path, torch_dtype=self.weights_dtype)
        controlnet_depth = ControlNetModel.from_pretrained(cfg.depth_path, torch_dtype=self.weights_dtype)
        
        self.is_sdxl = cfg.sd_path.startswith("SG161222/RealVisXL_V4.0")
        
        if self.is_sdxl:
            print("[INFO] Loading Stable Diffusion XL Pipeline...")
            self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                cfg.sd_path,
                controlnet=[controlnet_tile, controlnet_depth],
                torch_dtype=self.weights_dtype,
                variant="fp16",
                use_safetensors=True
            ).to(self.device)
            
            print("[INFO] Loading IP-Adapter for SDXL...")
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter", 
                subfolder="sdxl_models",
                weight_name="ip-adapter_sdxl.bin"
            )
            print("[INFO] Using Euler Scheduler for SDXL...")
            self.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        else:
            print("[INFO] Loading Stable Diffusion 1.5 Pipeline...")
            self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                cfg.sd_path,
                controlnet=[controlnet_tile, controlnet_depth],
                torch_dtype=self.weights_dtype,
                safety_checker=None,
            ).to(self.device)
            
            print("[INFO] Loading IP-Adapter for SD1.5...")
            self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
            self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
            
        self.pipe.set_ip_adapter_scale(cfg.ip_adapter_scale)
        self.pipe.enable_xformers_memory_efficient_attention()
        
        self.pipe.scheduler = self.scheduler
        self.pipe.set_progress_bar_config(disable=True)
        
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * cfg.max_step_percent)

    def _get_target_resolution(self, h, w):
        """Calculates nearest multiple of 8 resolution maintaining aspect ratio"""
        max_res = 1024 if self.is_sdxl else 512
        scale = max_res / max(h, w)
        new_h = int(round(h * scale / 8) * 8)
        new_w = int(round(w * scale / 8) * 8)
        return new_h, new_w

    @torch.no_grad()
    def multi_step(self, rgb: torch.Tensor, scan_rgb: torch.Tensor, scan_depth: torch.Tensor, prompt: str, current_step_ratio: float):
        """
        rgb: [H, W, 3] Sharp 3DGS Render
        scan_rgb: [H, W, 3] Blurry GT Scan
        scan_depth: [H, W] Smooth GT Scan Depth
        """
        # 1. Format Inputs to [B, C, H, W]
        rgb_BCHW = rgb.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
        orig_h, orig_w = rgb_BCHW.shape[2:]
        
        # Determine rectangular resolution suitable for the model
        target_h, target_w = self._get_target_resolution(orig_h, orig_w)
        
        rgb_resized = F.interpolate(rgb_BCHW, (target_h, target_w), mode="bilinear", align_corners=False)
        
        # Tile ControlNet & IP-Adapter (Uses original scan RGB)
        ctrl_tile = scan_rgb.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
        ctrl_tile_resized = F.interpolate(ctrl_tile, (target_h, target_w), mode="bilinear", align_corners=False)
        ip_adapter_input = F.interpolate(ctrl_tile, (224, 224), mode="bilinear", align_corners=False).to(self.weights_dtype)
        
        # Depth ControlNet (Normalize ignoring 0s)
        valid_mask = scan_depth > 0
        d_min = scan_depth[valid_mask].min() if valid_mask.sum() > 0 else 0.0
        d_max = scan_depth[valid_mask].max() if valid_mask.sum() > 0 else 1.0
        
        ctrl_depth = torch.where(valid_mask, (scan_depth - d_min) / (d_max - d_min + 1e-8), torch.zeros_like(scan_depth))
        ctrl_depth = ctrl_depth.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).clamp(0, 1)
        ctrl_depth_resized = F.interpolate(ctrl_depth, (target_h, target_w), mode="nearest")

        # 2. Anneal Timestep
        t = current_step_ratio * self.min_step + (1 - current_step_ratio) * self.max_step
        strength = t / self.num_train_timesteps

        # 3. Generate Pseudo-GT
        out_images = self.pipe(
            prompt=[prompt],
            negative_prompt=["blurry, motion blur, out of focus, distorted, artifact, worst quality"],
            image=rgb_resized.to(self.weights_dtype), 
            control_image=[ctrl_tile_resized.to(self.weights_dtype), ctrl_depth_resized.to(self.weights_dtype)],
            ip_adapter_image=ip_adapter_input,
            controlnet_conditioning_scale=self.cfg.controlnet_scales,
            strength=strength,
            num_inference_steps=self.cfg.num_steps_sample,
            guidance_scale=self.cfg.guidance_scale,
            output_type="pt"
        ).images

        # Return to original resolution [H, W, 3] to calculate losses properly
        pseudo_gt = F.interpolate(out_images, (orig_h, orig_w), mode="bilinear", align_corners=False)
        return pseudo_gt.squeeze(0).permute(1, 2, 0).clamp(0, 1).to(torch.float32)