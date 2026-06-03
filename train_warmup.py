import os
import argparse
import random
import time
import torch
import numpy as np
from tqdm import tqdm
import wandb

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat import export_splats

from src.config import PipelineConfig
from src.dataset import CustomGSDataset
from evaluation import EvaluationEngine
from src.losses import LossEngine, DynamicLossScheduler
from src.model import create_splats_with_optimizers
from src.utils import PipelineHelpers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--floor_number", type=int, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--run_eval", action="store_true", help="Flag to enable evaluation during training")
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
        floor_number=args.floor_number,
        version=args.version,
        data_dir=args.data_dir,
        run_eval=args.run_eval,
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
    base_dir = f"results/{cfg.scene_name}/{stage}/{cfg.version}/{cfg.floor_number}"
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    stats_dir = os.path.join(base_dir, "stats")
    ply_dir = os.path.join(base_dir, "ply")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(ply_dir, exist_ok=True)
    
    # 1. Setup Dataset, Losses, and Helpers    
    train_dataset = CustomGSDataset(data_dir=cfg.data_dir, device=cfg.device, split="train", test_every=cfg.test_every)    
    losses = LossEngine(device=cfg.device)
    loss_scheduler = DynamicLossScheduler(cfg.schedule)
    helpers = PipelineHelpers(device=cfg.device)
    
    # 2. Setup Splats & Strategy ( Densification logic )
    splats, optimizers = create_splats_with_optimizers(
        ply_path=os.path.join(cfg.data_dir, "pointcloud.ply"), cfg=cfg, device=cfg.device, sh_degree=cfg.sh_degree
    )
    
    if cfg.strategy_type == "mcmc":
        strategy = MCMCStrategy(cap_max=cfg.mcmc_cap_max, verbose=True)
        strategy.init_opa = cfg.init_opa
        strategy.init_scale = cfg.init_scale
        strategy.opacity_reg = cfg.opacity_reg
        strategy.scale_reg = cfg.scale_reg
        strategy_state = strategy.initialize_state()
    else:
        strategy = DefaultStrategy(verbose=True)
        strategy_state = strategy.initialize_state(scene_scale=1.0)  # Use a fixed scale for the default strategy to prevent giant Gaussians filling the whole screen
        
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / cfg.max_steps_warmup))
        
    # Project and Run Naming integration
    name = f"{cfg.scene_name}-{cfg.version}-f{cfg.floor_number}"
    wandb.init(
        project="thesis-gsplat-warmup",
        name=name,
        dir=base_dir,
        config=cfg.to_dict() # Auto-populates Wandb Config UI
    )
    
    eval_engine = None
    val_dataset = None
    
    if cfg.run_eval:
        val_dataset = CustomGSDataset(data_dir=cfg.data_dir, device=cfg.device, split="test", test_every=cfg.test_every)
        eval_engine = EvaluationEngine(cfg)
    
    start_time = time.time()
    
    # 3. Main Training Loop
    for step in tqdm(range(cfg.max_steps_warmup)):
        # Fetch random camera frame
        idx = torch.randint(0, len(train_dataset), (1,)).item()
        data = train_dataset[idx]
        
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
            width=train_dataset.W,
            height=train_dataset.H,
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
        # if cfg.strategy_type == "mcmc":
        #     # Scale the penalty by the total magnitude of your custom losses
        loss_multiplier = current_w_rgb + current_w_depth
        
        if cfg.opacity_reg > 0.0:
            total_loss += (cfg.opacity_reg * loss_multiplier) * torch.sigmoid(splats["opacities"]).mean()
        if cfg.scale_reg > 0.0:
            total_loss += (cfg.scale_reg * loss_multiplier) * torch.exp(splats["scales"]).mean()

        total_loss.backward()
        
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
            
        if step > 0 and (step + 1) % cfg.ckpt_interval == 0:
            helpers.save_checkpoint(splats, optimizers, strategy_state, step, ckpt_dir, "warmup", preserve_steps=cfg.eval_steps)

        if cfg.save_ply and (step in cfg.ply_steps or step == cfg.max_steps_warmup - 1):
            export_splats(
                means=splats["means"], scales=splats["scales"], quats=splats["quats"], 
                opacities=splats["opacities"], sh0=splats["sh0"], shN=splats["shN"],
                format="ply", save_to=os.path.join(ply_dir, f"point_cloud_{step}.ply")
            )

        # Optimization & Densification Steps
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # --- Post-Backward Dynamic Strategy Stepping ---
        # Run post-backward steps after backward and optimizer
        if cfg.strategy_type == "mcmc":
            strategy.step_post_backward(
                params=splats,
                optimizers=optimizers,
                state=strategy_state,
                step=step,
                info=info,
                lr=current_lr
            )
        else:
            strategy.step_post_backward(
                params=splats,
                optimizers=optimizers,
                state=strategy_state,
                step=step,
                info=info,
                packed=False,
            )
        
        
        if cfg.run_eval and eval_engine is not None and step > 0 and step in cfg.eval_steps:
            eval_engine.evaluate(step, splats, val_dataset, start_time, stats_dir)
          
        
            
    print("\n[INFO] Training complete")
    wandb.finish()

if __name__ == "__main__":
    run_warmup()