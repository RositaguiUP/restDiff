import os
import argparse
import random
import torch
import numpy as np
from tqdm import tqdm
import wandb

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from src.config import PipelineConfig
from src.dataset import CustomGSDataset
from src.losses import LossEngine, DynamicLossScheduler
from src.model import create_splats_with_optimizers
from src.utils import PipelineHelpers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--strategy_type", type=str, default="default", choices=["default", "mcmc"])
    parser.add_argument("--depth_start", type=float, default=0.3)
    parser.add_argument("--depth_end", type=float, default=0.02)
    parser.add_argument("--hold_steps", type=int, default=10000)
    parser.add_argument("--decay_steps", type=int, default=15000)
    return parser.parse_args()
    
def run_warmup():
    
    print("[INFO] Starting Warmup Stage")
    
    # DETERMINISTIC ORDERING: Ensures same camera idx sequence per run
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    args = parse_args()
    
    # Setup configs
    cfg = PipelineConfig(
        scene_name=args.scene_name,
        version=args.version,
        data_dir=args.data_dir,
        strategy_type=args.strategy_type,
    )
    
    # Override schedule config with CLI args
    cfg.schedule.depth_start = args.depth_start
    cfg.schedule.depth_end = args.depth_end
    cfg.schedule.rgb_start = 1.0 - args.depth_start
    cfg.schedule.rgb_end = 1.0 - args.depth_end
    cfg.schedule.hold_steps = args.hold_steps
    cfg.schedule.decay_steps = args.decay_steps
    
    # Dynamic Directories
    stage = "warmup"
    base_dir = f"results/{cfg.scene_name}/{stage}/{cfg.version}"
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. Setup Dataset, Losses, and Helpers
    dataset = CustomGSDataset(data_dir=cfg.data_dir, device=cfg.device)
    losses = LossEngine(device=cfg.device)
    loss_scheduler = DynamicLossScheduler(cfg.schedule)
    helpers = PipelineHelpers(device=cfg.device)
    
    # 2. Setup Splats & Strategy ( Densification logic )
    splats, optimizers = create_splats_with_optimizers(
        ply_path=os.path.join(cfg.data_dir, "pointcloud.ply"), cfg=cfg, device=cfg.device, sh_degree=cfg.sh_degree
    )
    
    if cfg.strategy_type == "mcmc":
        strategy = MCMCStrategy(verbose=True)
        strategy.init_opa = cfg.init_opa
        strategy.init_scale = cfg.init_scale
        strategy.opacity_reg = cfg.opacity_reg
        strategy.scale_reg = cfg.scale_reg
        strategy_state = strategy.initialize_state()
    else:
        strategy = DefaultStrategy(verbose=True)
        strategy_state = strategy.initialize_state(scene_scale=1.0)
        
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / cfg.max_steps_warmup))

    # Project and Run Naming integration
    name = f"{cfg.scene_name}-{cfg.version}"
    wandb.init(
        project="thesis-gsplat-warmup",
        name=name,
        dir=base_dir,
        config=cfg.to_dict() # Auto-populates Wandb Config UI
    )
    
    # 3. Main Training Loop
    for step in tqdm(range(cfg.max_steps_warmup)):
        # Fetch random camera frame
        idx = torch.randint(0, len(dataset), (1,)).item()
        data = dataset[idx]
        
        c2w = data["camtoworld"].unsqueeze(0) # [1, 4, 4]
        K = data["K"].unsqueeze(0)            # [1, 3, 3]
        gt_img = data["image"]                # [H, W, 3]
        gt_depth = data["depth"]              # [H, W]
        
        # SH Degree scheduling
        sh_degree_to_use = min(step // 1000, cfg.sh_degree)
        
        # Prepare attributes
        colors_sh = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        scales = torch.exp(splats["scales"])
        opacities = torch.sigmoid(splats["opacities"])
        
        # Rasterization (gsplat expects World-to-Camera viewmats)
        viewmats = torch.linalg.inv(c2w)
        
        render_colors, _, info = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=scales,
            opacities=opacities,
            colors=colors_sh,
            viewmats=viewmats,
            Ks=K,
            width=dataset.W,
            height=dataset.H,
            sh_degree=sh_degree_to_use,
            packed=False,
            absgrad=(
                strategy.absgrad
                if cfg.strategy_type == "default"
                else False
            ),
            render_mode="RGB+ED"
        )
        
        rendered = render_colors.squeeze(0)  # Shape becomes [H, W, 4]
        pred_img = rendered[..., :3]  # Channels 0, 1, 2 (RGB)
        pred_depth = rendered[..., 3]  # Channel 3 (Expected Projection Depth)

        # --- Pre-Backward Strategy Step ---
        strategy.step_pre_backward(
            params=splats, optimizers=optimizers, state=strategy_state, step=step, info=info
        )
        
        # Calculate Losses
        gs_loss = losses.compute_official_gs_losses(pred_img, gt_img)
        depth_loss = losses.compute_pearson_depth_loss(pred_depth, gt_depth)
        
        # Get dynamic weights for this specific step
        current_w_rgb, current_w_depth = loss_scheduler.get_weights(step)
        
        rgb_weighted = current_w_rgb * gs_loss
        depth_weighted = current_w_depth * depth_loss
        total_loss = rgb_weighted + depth_weighted
        
        # --- MCMC Regularization ---
        if cfg.strategy_type == "mcmc":
            # Scale the penalty by the total magnitude of your custom losses
            loss_multiplier = current_w_rgb + current_w_depth
            
            if cfg.opacity_reg > 0.0:
                total_loss += (cfg.opacity_reg * loss_multiplier) * torch.sigmoid(splats["opacities"]).mean()
            if cfg.scale_reg > 0.0:
                total_loss += (cfg.scale_reg * loss_multiplier) * torch.exp(splats["scales"]).mean()

        total_loss.backward()

        # Optimization & Densification Steps
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # --- Post-Backward Dynamic Strategy Stepping ---
        scene_scale = 1.0
        if len(splats["means"]) > 0:
            scene_scale = torch.max(
                splats["means"].max(dim=0).values - splats["means"].min(dim=0).values
            ).item()

        if cfg.strategy_type == "mcmc":
            strategy = MCMCStrategy(verbose=True)
            strategy_state = strategy.initialize_state()
        else:
            strategy = DefaultStrategy(verbose=True)
            strategy_state = strategy.initialize_state(scene_scale=scene_scale)

        # Logging
        if step % cfg.log_interval == 0:
            helpers.log_telemetry(step, total_loss.item(), rgb_weighted.item(), depth_weighted.item(), pred_img, gt_img, len(splats["means"]), extra_losses={
                "losses/loss_depth": depth_weighted.item(),
                "losses/loss_depth_raw": depth_loss.item(),
                "weights/lambda_rgb": current_w_rgb,
                "weights/lambda_depth": current_w_depth
            })
            
        if step % cfg.vis_interval == 0:
            helpers.log_visuals(step, pred_img, gt_img, pred_depth, gt_depth, prefix="warmup_vis")
            
        if step > 0 and step % cfg.ckpt_interval == 0:
            helpers.save_checkpoint(splats, optimizers, strategy_state, step, ckpt_dir, "warmup", preserve_steps=[7500, 15000])
    
    print("\n[INFO] Training complete. Saving final state...")
    helpers.save_checkpoint(splats, optimizers, strategy_state, cfg.max_steps_warmup, ckpt_dir, "warmup", preserve_steps=[7500, 15000])
    wandb.finish()

if __name__ == "__main__":
    run_warmup()