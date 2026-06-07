import os
import glob
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class PipelineHelpers:
    def __init__(self, device: str = "cuda"):
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
        
        self.cmap = plt.get_cmap("turbo")
        self.start_time = time.time()

    def colormap_depth(self, depth_tensor: torch.Tensor) -> np.ndarray:
        """Converts structural depth maps into Turbo RGB visualizations."""
        depth_np = depth_tensor.detach().cpu().numpy()
        valid_mask = depth_np > 0
        
        if valid_mask.sum() > 0:
            d_min, d_max = depth_np[valid_mask].min(), depth_np[valid_mask].max()
            depth_norm = np.zeros_like(depth_np)
            depth_norm[valid_mask] = (depth_np[valid_mask] - d_min) / (d_max - d_min + 1e-8)
        else:
            depth_norm = depth_np
            
        return (self.cmap(depth_norm)[..., :3] * 255).astype(np.uint8)

    def log_telemetry(self, step: int, total_loss: float, rgb_loss: float, depth_loss: float, 
                      pred_img: torch.Tensor, gt_img: torch.Tensor, num_gs: int, extra_metrics: dict = None, extra_losses: dict = None):
        """Computes and updates numerical analytical trackers."""
        pred_bchw = pred_img.permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
        gt_bchw = gt_img.permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
        
        # Performance evaluation benchmarks run purely against target true scans
        psnr_val = self.psnr_metric(pred_bchw, gt_bchw).item()
        ssim_val = self.ssim_metric(pred_bchw, gt_bchw).item()
        lpips_val = self.lpips_metric(pred_bchw, gt_bchw).item()
        mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        elapsed_min = (time.time() - self.start_time) / 60.0
        
        metrics = {
            "metrics/psnr": psnr_val,
            "metrics/ssim": ssim_val,
            "metrics/lpips": lpips_val,
            "metrics/num_gaussians": num_gs,
            "metrics/vram_gb": mem_gb,
            "metrics/time_minutes": elapsed_min
        }
        
        if extra_metrics:
            metrics.update(extra_metrics)
            
        losses = {
            "losses/loss_total": total_loss,
            "losses/loss_rgb": rgb_loss,
            "losses/loss_depth": depth_loss,
        }
        
        if extra_losses:
            losses.update(extra_losses)
            
        wandb.log({**metrics, **losses}, step=step)

    def log_visuals(self, step: int, pred_img: torch.Tensor, gt_img: torch.Tensor, 
                    pred_depth: torch.Tensor, gt_depth: torch.Tensor, prefix: str = "vis", extra_images: dict = None):
        """Generates multi-modal rendering verification telemetry frames."""
        render_rgb_np = torch.clamp(pred_img, 0.0, 1.0).detach().cpu().numpy()
        target_rgb_np = torch.clamp(gt_img, 0.0, 1.0).detach().cpu().numpy()
        
        render_depth_rgb = self.colormap_depth(pred_depth)
        target_depth_rgb = self.colormap_depth(gt_depth)
        
        log_dict = {
            f"{prefix}/rendered_rgb": wandb.Image(render_rgb_np, caption=f"Render RGB Step {step}"),
            f"{prefix}/target_rgb": wandb.Image(target_rgb_np, caption=f"Target RGB Step {step}"),
            f"{prefix}/rendered_depth": wandb.Image(render_depth_rgb, caption=f"Rendered Depth Step {step}"),
            f"{prefix}/target_depth": wandb.Image(target_depth_rgb, caption=f"Target Depth Step {step}")
        }
        if extra_images:
            log_dict.update(extra_images)
            
        wandb.log(log_dict, step=step)

    @staticmethod
    def save_checkpoint(splats, optimizers, strategy_state, step: int, output_dir: str, name_tag: str, preserve_steps: list):
        os.makedirs(output_dir, exist_ok=True)
        checkpoint = {
            "step": step,
            "splats": {k: v.data for k, v in splats.items()},
            "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
            "strategy_state": strategy_state,
        }
        
        ckpt_path = os.path.join(output_dir, f"ckpt_{name_tag}_{step:05d}.pt")
        latest_path = os.path.join(output_dir, f"ckpt_{name_tag}_latest.pt")
        
        torch.save(checkpoint, ckpt_path)
        
        # Only create/update latest if this step is not the highest preserved step
        max_preserved = max(preserve_steps) if preserve_steps else None
        if step != max_preserved:
            torch.save(checkpoint, latest_path)

        
        # Cleanup: Delete old files that aren't 'latest' or in the preserve list
        all_ckpts = glob.glob(os.path.join(output_dir, f"ckpt_{name_tag}_*.pt"))
        for f in all_ckpts:
            if "latest" in f: 
                if step == max_preserved:
                    os.remove(f)
                else:
                    continue
            try:
                f_step = int(f.split("_")[-1].split(".")[0])
                if f_step != step and f_step not in preserve_steps:
                    os.remove(f)
            except ValueError:
                pass