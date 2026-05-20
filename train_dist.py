import os
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

from src.dataset import CustomGSDataset
from src.model import create_splats_with_optimizers
from src.guidance import StableDiffusionControlNetGuidance

def compute_pearson_depth_loss(pred_depth, gt_depth):
    """Computes the Pearson correlation loss for depth constraints."""
    valid_mask = gt_depth > 0
    if valid_mask.sum() < 10:
        return torch.tensor(0.0, device=pred_depth.device)
    
    rend_d = pred_depth[valid_mask]
    gt_d = gt_depth[valid_mask]
    
    rend_centered = rend_d - rend_d.mean()
    gt_centered = gt_d - gt_d.mean()
    
    cov = (rend_centered * gt_centered).sum()
    std_rend = torch.sqrt((rend_centered ** 2).sum() + 1e-8)
    std_gt = torch.sqrt((gt_centered ** 2).sum() + 1e-8)
    
    pearson_corr = cov / (std_rend * std_gt)
    return 1.0 - pearson_corr

def save_checkpoint(splats, optimizers, strategy_state, step, output_dir):
    """Saves the current state of Gaussians, optimizers, and densification strategy."""
    os.makedirs(output_dir, exist_ok=True)
    
    splat_state = {k: v.data for k, v in splats.items()}
    
    checkpoint = {
        "step": step,
        "splats": splat_state,
        "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
        "strategy_state": strategy_state,
    }
    
    ckpt_path = os.path.join(output_dir, f"ckpt_distill_{step:05d}.pt")
    latest_path = os.path.join(output_dir, "ckpt_distill_latest.pt")
    
    torch.save(checkpoint, ckpt_path)
    torch.save(checkpoint, latest_path)
    print(f"\n[INFO] Saved distillation checkpoint to {ckpt_path}")

def train_stage3():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "data/6VSV7_695_v2"
    ckpt_path = os.path.join(data_dir, "checkpoints", "ckpt_latest.pt")
    output_dir = os.path.join(data_dir, "checkpoints_distilled_2")
    
    # 1. Setup Data & Metrics
    dataset = CustomGSDataset(data_dir=data_dir, device=device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
    
    # 2. Setup Splats & Strategy
    splats, optimizers = create_splats_with_optimizers(
        ply_path=os.path.join(data_dir, "pointcloud.ply"), device=device
    )
    strategy = DefaultStrategy(verbose=False)
    
    # --- LOAD CHECKPOINT ---
    print(f"[INFO] Loading Checkpoint from Stage 2: {ckpt_path}")
    checkpoint = torch.load(ckpt_path)
    for k, v in splats.items():
        v.data = checkpoint["splats"][k]
    for name, opt in optimizers.items():
        opt.load_state_dict(checkpoint["optimizers"][name])
    strategy_state = checkpoint["strategy_state"]
    start_step = checkpoint["step"]
    
    # 3. Setup Guidance
    guidance = StableDiffusionControlNetGuidance(device=device)
    prompt = "4k image, photorealistic, cinematic lighting, sharp, high resolution" # Modify dynamically if needed

    # Hyperparameters
    max_steps = start_step + 15000 # Run distillation for 15,000 steps
    lambda_rgb = 0.3           # Anchor weight
    lambda_depth = 0.2         # Anchor weight
    lambda_distill_mse = 0.01  # Distillation weight
    lambda_distill_lpips = 0.02 # Distillation weight

    wandb.init(project="thesis-gsplat-distillation", name="run-04-sds-controlnet")

    # 4. Main Distillation Loop
    for step in tqdm(range(start_step, max_steps)):
        step_ratio = (step - start_step) / (max_steps - start_step)
        
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
        render_colors, render_alphas, info = rasterization(
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
        
        pred_img = render_colors.squeeze(0)[..., :3]  # [H, W, 3]
        pred_depth = render_colors.squeeze(0)[..., 3] # [H, W]
        
        strategy.step_pre_backward(params=splats, optimizers=optimizers, state=strategy_state, step=step, info=info)
        
        # --- A. ANCHOR LOSSES (Geometry & Physical) ---
        loss_rgb = F.l1_loss(pred_img, gt_img)
        loss_depth = compute_pearson_depth_loss(pred_depth, gt_depth)
        
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
        
        loss_distill_mse = F.mse_loss(pred_img_bchw, pseudo_gt_bchw)
        # loss_distill_lpips = lpips_metric(pred_img_bchw * 2 - 1, pseudo_gt_bchw * 2 - 1).mean()
        # Clamp the 3DGS render to safely remove SH evaluation outliers
        pred_img_bchw_clamped = torch.clamp(pred_img_bchw, 0.0, 1.0)

        # pseudo_gt is already clamped by the diffusion pipeline, but we clamp it here to be absolutely safe
        pseudo_gt_bchw_clamped = torch.clamp(pseudo_gt_bchw, 0.0, 1.0)

        # Pass raw [0, 1] tensors to torchmetrics
        loss_distill_lpips = lpips_metric(pred_img_bchw_clamped, pseudo_gt_bchw_clamped).mean()


        # Combine
        total_loss = (
            (lambda_rgb * loss_rgb) + 
            (lambda_depth * loss_depth) + 
            (lambda_distill_mse * loss_distill_mse) + 
            (lambda_distill_lpips * loss_distill_lpips)
        )
        
        total_loss.backward()

        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
        strategy.step_post_backward(params=splats, optimizers=optimizers, state=strategy_state, step=step, info=info, packed=False)

        # Logging
        if step % 20 == 0:
            wandb.log({
                "loss_total": total_loss.item(),
                "loss_rgb": loss_rgb.item(),
                "loss_depth": loss_depth.item(),
                "loss_distill_mse": loss_distill_mse.item(),
                "loss_distill_lpips": loss_distill_lpips.item(),
            })
            
        if step % 500 == 0:
            # Log the Render vs the Diffusion Pseudo-GT to see the refinement
            render_np = torch.clamp(pred_img, 0.0, 1.0).detach().cpu().numpy()
            pseudo_np = torch.clamp(pseudo_gt, 0.0, 1.0).detach().cpu().numpy()
            wandb.log({
                "render_sharp": wandb.Image(render_np, caption=f"3DGS Render (Step {step})"),
                "pseudo_gt": wandb.Image(pseudo_np, caption=f"ControlNet Target (Step {step})")
            })
        
        if step > start_step and step % 2500 == 0:
            save_checkpoint(splats, optimizers, strategy_state, step, output_dir)

    # --- NEW: Final Checkpoint Saving ---
    print("\n[INFO] Distillation complete. Saving final state...")
    save_checkpoint(splats, optimizers, strategy_state, max_steps, output_dir)

    wandb.finish()

if __name__ == "__main__":
    train_stage3()