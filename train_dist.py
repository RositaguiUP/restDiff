import os
import time
import argparse
import torch
import numpy as np
from tqdm import tqdm
import wandb
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

from src.config import PipelineConfig
from src.dataset import CustomGSDataset
from src.losses import LossEngine
from src.model import create_splats_with_optimizers
from src.guidance import StableDiffusionControlNetGuidance
from src.utils import PipelineHelpers


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: Distillation Refinement Script")
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--warmup_version", type=str, required=True)
    parser.add_argument("--lambda_distill_rgb", type=float, default=1.0)
    parser.add_argument("--lambda_distill_depth", type=float, default=10.0)
    return parser.parse_args()

def run_distillation():
    # DETERMINISTIC ORDERING
    torch.manual_seed(42)
    np.random.seed(42)
    
    args = parse_args()
    cfg = PipelineConfig(
        scene_name=args.scene_name,
        version=args.version,
        data_dir=args.data_dir,
        warmup_version=args.warmup_version,
        lambda_distill_rgb=args.lambda_distill_rgb,
        lambda_distill_depth=args.lambda_distill_depth
    )
    
    # Dynamic Directories
    stage = "distill"
    base_dir = f"results/{cfg.scene_name}/{stage}/{cfg.version}"
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    wandb_dir = os.path.join(base_dir, "wandb")
    os.makedirs(wandb_dir, exist_ok=True)
    
    # 1. Setup Dataset, Losses, Helpers and Metrics
    dataset = CustomGSDataset(data_dir=cfg.data_dir, device=cfg.device)
    losses = LossEngine(device=cfg.device)
    helpers = PipelineHelpers(device=cfg.device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(cfg.device)
    
    # 2. Setup Splats & Strategy
    splats, optimizers = create_splats_with_optimizers(
        ply_path=os.path.join(cfg.data_dir, "pointcloud.ply"), cfg=cfg.opts, device=cfg.device, sh_degree=cfg.sh_degree
    )
    strategy = DefaultStrategy(verbose=False)
    
    # --- LOAD CHECKPOINT ---
    # 3. Restore State Weights from Warmup Results
    print(f"[INFO] Restoring Checkpoint State")
    warmup_ckpt = f"results/{cfg.scene_name}/warmup/{cfg.warmup_version}/checkpoints/ckpt_warmup_latest.pt"
    checkpoint = torch.load(warmup_ckpt, map_location=cfg.device)
    for k, v in splats.items(): v.data = checkpoint["splats"][k]
    for name, opt in optimizers.items(): opt.load_state_dict(checkpoint["optimizers"][name])
    strategy_state = checkpoint["strategy_state"]
    start_step = checkpoint["step"]
    
    # 4. Setup Guidance
    guidance = StableDiffusionControlNetGuidance(cfg=cfg.guidance, device=cfg.device)
    prompt = "4k image, photorealistic, cinematic lighting, sharp, high resolution" # Modify dynamically if needed
    
    preserve_steps = [start_step + 7500, start_step + 15000]
    target_end_step = start_step + cfg.max_steps_distill
    
    # Project and Run Naming integration
    run_id = f"{cfg.version}-{time.strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="thesis-gsplat-distillation",
        name=f"{cfg.scene_name}-{cfg.version}",
        id=run_id,
        dir=wandb_dir,
        config=cfg.to_dict()
    )
    
    # 5. Main Distillation Loop
    for step in tqdm(range(start_step, target_end_step)):
        step_ratio = (step - start_step) / cfg.max_steps_distill
        idx = torch.randint(0, len(dataset), (1,)).item()
        data = dataset[idx]
        
        c2w = data["camtoworld"].unsqueeze(0) 
        K = data["K"].unsqueeze(0)            
        gt_img = data["image"]       # [H, W, 3]         
        gt_depth = data["depth"]     # [H, W]
        
        colors_sh = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        scales = torch.exp(splats["scales"])
        opacities = torch.sigmoid(splats["opacities"])
        
        # Render (Asking for RGB + Expected Depth)
        render_colors, _, info = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=scales,
            opacities=opacities,
            colors=colors_sh,
            viewmats=torch.linalg.inv(c2w),
            Ks=K,
            width=dataset.W,
            height=dataset.H,
            sh_degree=3,
            packed=False,
            absgrad=strategy.absgrad,
            render_mode="RGB+ED" 
        )
        
        pred_img = render_colors.squeeze(0)[..., :3]
        pred_depth = render_colors.squeeze(0)[..., 3]
        
        strategy.step_pre_backward(params=splats, optimizers=optimizers, state=strategy_state, step=step, info=info)
        
        # --- A. ANCHOR LOSSES (Geometry & Physical) ---
        gs_loss = losses.compute_official_gs_losses(pred_img, gt_img)
        rgb_loss = cfg.lambda_warmup_rgb * gs_loss
        
        depth_loss = cfg.lambda_warmup_depth * losses.compute_pearson_depth_loss(pred_depth, gt_depth)
        
        # --- B. DISTILLATION PASS (Diffusion) ---
        # Generate the Perfect Pseudo-GT using ControlNet
        pseudo_gt = guidance.multi_step(
            rgb=pred_img.detach(), 
            scan_rgb=gt_img, 
            scan_depth=gt_depth, 
            prompt=prompt, 
            current_step_ratio=step_ratio
        )
        
        # Compute Distillation Losses
        pred_img_bchw = pred_img.permute(2, 0, 1).unsqueeze(0)
        pseudo_gt_bchw = pseudo_gt.permute(2, 0, 1).unsqueeze(0)
        
        # Clamp the 3DGS render to safely remove SH evaluation outliers
        pred_img_bchw_clamped = torch.clamp(pred_img_bchw, 0.0, 1.0)

        # pseudo_gt is already clamped by the diffusion pipeline, but we clamp it here to be absolutely safe
        pseudo_gt_bchw_clamped = torch.clamp(pseudo_gt_bchw, 0.0, 1.0)

        loss_mse = cfg.lambda_distill_mse * F.mse_loss(pred_img_bchw_clamped, pseudo_gt_bchw_clamped)
        loss_lpips = cfg.lambda_distill_lpips * lpips_metric(pred_img_bchw_clamped, pseudo_gt_bchw_clamped).mean()
        
        total_loss = rgb_loss + depth_loss + loss_mse + loss_lpips
        total_loss.backward()

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
            
        strategy.step_post_backward(params=splats, optimizers=optimizers, state=strategy_state, step=step, info=info, packed=False)

        # Logging
        if step % cfg.log_interval == 0:
            helpers.log_telemetry(step, total_loss.item(), rgb_loss.item(), depth_loss.item(), pred_img, gt_img, len(splats["means"]), extra_losses={
                "losses/loss_distill_mse": loss_mse.item(),
                "losses/loss_distill_lpips": loss_lpips.item()
            })
            
        if step % cfg.vis_interval == 0:
            extra_imgs = {"distill_vis/pseudo_ground_truth": wandb.Image(pseudo_gt.detach().cpu().numpy(), caption=f"Pseudo GT Step {step}")}
            helpers.log_visuals(step, pred_img, gt_img, pred_depth, gt_depth, prefix="distill_vis", extra_images=extra_imgs)
                        
        
        if step > start_step and step % cfg.ckpt_interval == 0:
            helpers.save_checkpoint(splats, optimizers, strategy_state, step, os.path.join(cfg.data_dir, "checkpoints_distilled"), "distill")

    # --- NEW: Final Checkpoint Saving ---
    print("\n[INFO] Distillation complete. Saving final state...")
    helpers.save_checkpoint(splats, optimizers, strategy_state, target_end_step, os.path.join(cfg.data_dir, "checkpoints_distilled"), "distill")
    wandb.finish()

if __name__ == "__main__":
    run_distillation()