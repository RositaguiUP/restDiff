import os
import time
import argparse
import torch
import wandb
import numpy as np
from tqdm import tqdm

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

from src.config import PipelineConfig, OptimizerConfig
from src.dataset import CustomGSDataset
from src.losses import LossEngine
from src.model import create_splats_with_optimizers
from src.utils import PipelineHelpers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--lambda_warmup_rgb", type=float, default=1.0)
    parser.add_argument("--lambda_warmup_depth", type=float, default=5.0)
    return parser.parse_args()
    
def run_warmup():
    
    print("[INFO] Starting Warmup Stage")
    
    # DETERMINISTIC ORDERING: Ensures same camera idx sequence per run
    torch.manual_seed(42)
    np.random.seed(42)
    
    args = parse_args()
    
    # Instantiate parsed property instances
    cfg = PipelineConfig(
        scene_name=args.scene_name,
        version=args.version,
        data_dir=args.data_dir,
        lambda_warmup_rgb=args.lambda_warmup_rgb,
        lambda_warmup_depth=args.lambda_warmup_depth
    )
    
    # Dynamic Directories
    stage = "warmup"
    base_dir = f"results/{cfg.scene_name}/{stage}/{cfg.version}"
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    wandb_dir = os.path.join(base_dir, "wandb")
    os.makedirs(wandb_dir, exist_ok=True)
    
    # 1. Setup Dataset, Losses, and Helpers
    dataset = CustomGSDataset(data_dir=cfg.data_dir, device=cfg.device)
    losses = LossEngine(device=cfg.device)
    helpers = PipelineHelpers(device=cfg.device)
    
    # 2. Setup Splats & Strategy ( Densification logic )
    splats, optimizers = create_splats_with_optimizers(
        ply_path=os.path.join(cfg.data_dir, "pointcloud.ply"), cfg=cfg.opts, device=cfg.device, sh_degree=cfg.sh_degree
    )
    
    strategy = DefaultStrategy(verbose=True)
    strategy_state = strategy.initialize_state(scene_scale=1.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / cfg.max_steps_warmup))

    # Project and Run Naming integration
    run_id = f"{cfg.version}-{time.strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="thesis-gsplat-warmup",
        name=f"{cfg.scene_name}-{cfg.version}",
        id=run_id,
        dir=wandb_dir,
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
            absgrad=strategy.absgrad,
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
        rgb_loss = cfg.lambda_warmup_rgb * gs_loss
        
        depth_loss = cfg.lambda_warmup_depth * losses.compute_pearson_depth_loss(pred_depth, gt_depth)
        
        total_loss = rgb_loss + depth_loss
        total_loss.backward()

        # Optimization & Densification Steps
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
            
        scheduler.step()
        
        # --- Post-Backward Strategy Step ---
        strategy.step_post_backward(
            params=splats, optimizers=optimizers, state=strategy_state, step=step, info=info, packed=False
        )

        # Logging
        if step % cfg.log_interval == 0:
            helpers.log_telemetry(step, total_loss.item(), rgb_loss.item(), depth_loss.item(), pred_img, gt_img, len(splats["means"]), extra_losses={
                "losses/loss_depth": depth_loss.item()
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