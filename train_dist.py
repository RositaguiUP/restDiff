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
from evaluation.evaluation import EvaluationEngine
from src.losses import LossEngine, DynamicLossScheduler
from src.model import create_splats_with_optimizers
from src.guidance import StableDiffusionControlNetGuidance
from src.utils import PipelineHelpers

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: Distillation Refinement Script")
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--floor_number", type=int, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--warmup_version", type=str, required=True)
    parser.add_argument("--run_eval", action="store_true", help="Flag to enable evaluation")
    parser.add_argument("--strategy_type", type=str, default="default", choices=["default", "mcmc"])
    parser.add_argument("--lambda_distill_rgb", type=float, default=1.0)
    parser.add_argument("--lambda_distill_depth", type=float, default=10.0)
    # Loss schedule overrides
    parser.add_argument("--depth_start", type=float, default=0.3)
    parser.add_argument("--depth_end", type=float, default=0.02)
    parser.add_argument("--hold_steps", type=int, default=10000)
    parser.add_argument("--decay_steps", type=int, default=15000)
    return parser.parse_args()

def run_distillation():
    print("[INFO] Starting Distillation Stage")
    
    # DETERMINISTIC ORDERING
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    args = parse_args()
    
    cfg = PipelineConfig(
        scene_name=args.scene_name,
        floor_number=args.floor_number,
        version=args.version,
        data_dir=args.data_dir,
        warmup_version=args.warmup_version,
        run_eval=args.run_eval,
        strategy_type=args.strategy_type,
        lambda_distill_rgb=args.lambda_distill_rgb,
        lambda_distill_depth=args.lambda_distill_depth
    )
    
    # Override schedule config with CLI args
    cfg.schedule.depth_start = args.depth_start
    cfg.schedule.depth_end = args.depth_end
    cfg.schedule.rgb_start = 1.0 - args.depth_start
    cfg.schedule.rgb_end = 1.0 - args.depth_end
    cfg.schedule.hold_steps = args.hold_steps
    cfg.schedule.decay_steps = args.decay_steps
    
    # Dynamic Directories
    stage = "distill"
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
        
    # 2. Setup Splats & Strategy
    splats, optimizers = create_splats_with_optimizers(
        ply_path=os.path.join(cfg.data_dir, "pointcloud.ply"), cfg=cfg, device=cfg.device, sh_degree=cfg.sh_degree
    )
    
    if cfg.strategy_type == "mcmc":
        strategy = MCMCStrategy(cap_max=cfg.mcmc_cap_max, verbose=True)
        strategy.init_opa = cfg.init_opa
        strategy.init_scale = cfg.init_scale
        strategy.opacity_reg = cfg.opacity_reg
        strategy.scale_reg = cfg.scale_reg
    else:
        strategy = DefaultStrategy(refine_stop_iter=cfg.refine_stop_iter, verbose=True)
    
    # --- LOAD CHECKPOINT ---
    print(f"[INFO] Restoring Checkpoint State from Warmup")
    warmup_ckpt = f"results/{cfg.scene_name}/warmup/{cfg.warmup_version}/{cfg.floor_number}/checkpoints/ckpt_warmup_29999.pt"
    checkpoint = torch.load(warmup_ckpt, map_location=cfg.device)
    for k, v in splats.items(): v.data = checkpoint["splats"][k]
    for name, opt in optimizers.items(): opt.load_state_dict(checkpoint["optimizers"][name])
    
    strategy_state = checkpoint["strategy_state"]
    start_step = checkpoint["step"]
    
    # Setup LR Scheduler resuming from start_step
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / cfg.max_steps_distill))
    for _ in range(start_step):
        scheduler.step()
    
    # 4. Setup Guidance
    guidance = StableDiffusionControlNetGuidance(cfg=cfg.guidance, device=cfg.device)
    prompt = "4k image, photorealistic, cinematic lighting, sharp, high resolution, precise indoor architecture" 
    
    preserve_steps = [start_step + 7500, start_step + 15000]
    target_end_step = start_step + cfg.max_steps_distill
    
    eval_engine = None
    val_dataset = None
    if cfg.run_eval:
        val_dataset = CustomGSDataset(data_dir=cfg.data_dir, device=cfg.device, split="test", test_every=cfg.test_every)
        eval_engine = EvaluationEngine(cfg)
    
    # Project and Run Naming integration
    name = f"{cfg.scene_name}-{cfg.version}-f{cfg.floor_number}"
    wandb.init(
        project="thesis-gsplat-distillation",
        name=name,
        dir=base_dir,
        config=cfg.to_dict()
    )
    
    start_time = time.time()
    
    # 5. Main Distillation Loop
    for step in tqdm(range(start_step, target_end_step)):
        step_ratio = (step - start_step) / cfg.max_steps_distill
        idx = torch.randint(0, len(train_dataset), (1,)).item()
        data = train_dataset[idx]
        
        c2w = data["camtoworld"].unsqueeze(0) 
        K = data["K"].unsqueeze(0)            
        gt_img = data["image"]               
        gt_depth = data["depth"]             
        
        sh_degree_to_use = min(step // 1000, cfg.sh_degree)
        colors_sh = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        scales = torch.exp(splats["scales"])
        opacities = torch.sigmoid(splats["opacities"])
        
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
            absgrad=(strategy.absgrad if cfg.strategy_type == "default" else False),
            render_mode="RGB+ED" 
        )
        
        rendered = render_colors.squeeze(0)
        pred_img = rendered[..., :3]
        pred_depth = rendered[..., 3]
        
        strategy.step_pre_backward(params=splats, optimizers=optimizers, state=strategy_state, step=step, info=info)
        
        # --- A. ANCHOR LOSSES (Geometry & Physical) ---
        gs_loss = losses.compute_official_gs_losses(pred_img, gt_img, lambda_ssim=cfg.lambda_ssim)
        depth_loss = losses.compute_pearson_depth_loss(pred_depth, gt_depth)
        
        current_w_rgb, current_w_depth = loss_scheduler.get_weights(step)
        rgb_weighted = current_w_rgb * gs_loss
        depth_weighted = current_w_depth * depth_loss
        
        # --- B. DISTILLATION PASS (Diffusion) ---
        pseudo_gt = guidance.multi_step(
            rgb=pred_img.detach(), 
            scan_rgb=gt_img, 
            scan_depth=gt_depth, 
            prompt=prompt, 
            current_step_ratio=step_ratio
        )
        
        loss_mse, loss_lpips = losses.compute_distillation_losses(pred_img, pseudo_gt)
        mse_weighted = cfg.lambda_distill_mse * loss_mse
        lpips_weighted = cfg.lambda_distill_lpips * loss_lpips
        
        total_loss = rgb_weighted + depth_weighted + mse_weighted + lpips_weighted
        
        # --- MCMC Regularization ---
        loss_multiplier = current_w_rgb + current_w_depth
        if cfg.strategy_type == "mcmc":
            if cfg.opacity_reg > 0.0:
                total_loss += (cfg.opacity_reg * loss_multiplier) * torch.sigmoid(splats["opacities"]).mean()
            if cfg.scale_reg > 0.0:
                total_loss += (cfg.scale_reg * loss_multiplier) * torch.exp(splats["scales"]).mean()

        total_loss.backward()

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
            
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
            
        # --- Post-Backward Dynamic Strategy Stepping ---
        if cfg.strategy_type == "mcmc":
            strategy.step_post_backward(params=splats, optimizers=optimizers, state=strategy_state, step=step, info=info, lr=current_lr)
        else:
            strategy.step_post_backward(params=splats, optimizers=optimizers, state=strategy_state, step=step, info=info, packed=False)

        # Logging
        if step % cfg.log_interval == 0:
            helpers.log_telemetry(step, total_loss.item(), rgb_weighted.item(), depth_weighted.item(), pred_img, gt_img, len(splats["means"]), extra_losses={
                "losses/loss_depth_raw": depth_loss.item(),
                "losses/loss_distill_mse": mse_weighted.item(),
                "losses/loss_distill_lpips": lpips_weighted.item(),
                "weights/lambda_rgb": current_w_rgb,
                "weights/lambda_depth": current_w_depth
            })
            
        if step % cfg.vis_interval == 0:
            extra_imgs = {"distill_vis/pseudo_ground_truth": wandb.Image(pseudo_gt.detach().cpu().numpy(), caption=f"Pseudo GT Step {step}")}
            helpers.log_visuals(step, pred_img, gt_img, pred_depth, gt_depth, prefix="distill_vis", extra_images=extra_imgs)
                        
        if step > start_step and step % cfg.ckpt_interval == 0:
            helpers.save_checkpoint(splats, optimizers, strategy_state, step, ckpt_dir, "distill", preserve_steps)
            
        if cfg.save_ply and (step in cfg.ply_steps or step == target_end_step - 1):
            export_splats(
                means=splats["means"], scales=splats["scales"], quats=splats["quats"], 
                opacities=splats["opacities"], sh0=splats["sh0"], shN=splats["shN"],
                format="ply", save_to=os.path.join(ply_dir, f"point_cloud_{step}.ply")
            )
            
        if cfg.run_eval and eval_engine is not None and step > start_step and step in cfg.eval_steps:
            eval_engine.evaluate(step, splats, val_dataset, start_time, stats_dir)
            
        # Free up VRAM to respect environment quotas between loops
        torch.cuda.empty_cache()

    print("\n[INFO] Distillation complete. Saving final state...")
    helpers.save_checkpoint(splats, optimizers, strategy_state, target_end_step, ckpt_dir, "distill", preserve_steps)
    wandb.finish()

if __name__ == "__main__":
    run_distillation()