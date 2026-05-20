import os
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from torchmetrics.image import StructuralSimilarityIndexMeasure
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

from src.dataset import CustomGSDataset
from src.model import create_splats_with_optimizers


def save_checkpoint(splats, optimizers, strategy_state, step, output_dir):
    """Saves the current state of Gaussians, optimizers, and densification strategy."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract raw data tensors from nn.ParameterDict
    splat_state = {k: v.data for k, v in splats.items()}
    
    checkpoint = {
        "step": step,
        "splats": splat_state,
        "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
        "strategy_state": strategy_state,
    }
    
    # Save a temporary/periodic file and a "latest" tag file
    ckpt_path = os.path.join(output_dir, f"ckpt_{step:05d}.pt")
    latest_path = os.path.join(output_dir, "ckpt_latest.pt")
    
    torch.save(checkpoint, ckpt_path)
    torch.save(checkpoint, latest_path)
    print(f"\n[INFO] Saved checkpoint to {ckpt_path}")
    
    
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "data/6VSV7_695_v2"
    output_dir = os.path.join(data_dir, "checkpoints")
    
    # 1. Setup Data & Metrics
    dataset = CustomGSDataset(data_dir=data_dir, device=device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # 2. Setup Splats & Strategy ( Densification logic )
    splats, optimizers = create_splats_with_optimizers(
        ply_path=os.path.join(data_dir, "pointcloud.ply"),
        device=device
    )
    
    strategy = DefaultStrategy(verbose=True)
    strategy_state = strategy.initialize_state(scene_scale=1.0)
    
    # 3. Setup Scheduler (Exponential decay for positions)
    max_steps = 30000
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
    )

    wandb.init(project="thesis-gsplat-warmup", name="run-03-official-strategy")

    # 4. Main Training Loop
    for step in tqdm(range(max_steps)):
        # Fetch random camera frame
        idx = torch.randint(0, len(dataset), (1,)).item()
        data = dataset[idx]
        
        c2w = data["camtoworld"].unsqueeze(0) # [1, 4, 4]
        K = data["K"].unsqueeze(0)            # [1, 3, 3]
        gt_img = data["image"]                # [H, W, 3]
        
        # SH Degree scheduling
        sh_degree_to_use = min(step // 1000, 3)
        
        # Prepare attributes
        colors_sh = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        scales = torch.exp(splats["scales"])
        opacities = torch.sigmoid(splats["opacities"])
        
        # Rasterization (gsplat expects World-to-Camera viewmats)
        viewmats = torch.linalg.inv(c2w)
        
        render_colors, render_alphas, info = rasterization(
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
            absgrad=strategy.absgrad
        )
        
        pred_img = render_colors.squeeze(0) # [H, W, 3]
        
        # --- Pre-Backward Strategy Step ---
        strategy.step_pre_backward(
            params=splats, optimizers=optimizers, state=strategy_state, step=step, info=info
        )
        
        # Calculate Loss (L1 + SSIM)
        l1loss = F.l1_loss(pred_img, gt_img)
        
        # SSIM expects [B, C, H, W]
        pred_img_b = pred_img.permute(2, 0, 1).unsqueeze(0)
        gt_img_b = gt_img.permute(2, 0, 1).unsqueeze(0)
        ssimloss = 1.0 - ssim_metric(pred_img_b, gt_img_b)
        
        total_loss = 0.8 * l1loss + 0.2 * ssimloss
        total_loss.backward()

        # Optimization & Densification Steps
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
        scheduler.step()
        
        # --- Post-Backward Strategy Step ---
        strategy.step_post_backward(
            params=splats, optimizers=optimizers, state=strategy_state, step=step, info=info, packed=False
        )

        # Logging
        if step % 50 == 0:
            wandb.log({
                "loss": total_loss.item(),
                "l1_loss": l1loss.item(),
                "ssim_loss": ssimloss.item(),
                "num_GS": len(splats["means"])
            })
            
        if step % 1000 == 0:
            rendered_img_np = torch.clamp(pred_img, 0.0, 1.0).detach().cpu().numpy()
            wandb.log({"render": wandb.Image(rendered_img_np, caption=f"Step {step}")})
        
        if step > 0 and step % 5000 == 0:
            save_checkpoint(splats, optimizers, strategy_state, step, output_dir)
    
    print("\n[INFO] Training complete. Saving final state...")
    save_checkpoint(splats, optimizers, strategy_state, max_steps, output_dir)     
    wandb.finish()

if __name__ == "__main__":
    train()