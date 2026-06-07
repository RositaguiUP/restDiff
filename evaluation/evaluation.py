import os
import time
import json
import argparse

import torch
import numpy as np
import wandb

from gsplat.rendering import rasterization

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.dataset import CustomGSDataset
from src.config import PipelineConfig
from utils.gsplat_org.color_correct import color_correct_affine, color_correct_quadratic

class EvaluationEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(cfg.device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(cfg.device)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(cfg.device)
        
    @torch.no_grad()
    def evaluate(self, step, splats, val_dataset, start_time, stats_dir):
        print(f"\n[INFO] Running Evaluation at Step {step}...")
        torch.cuda.synchronize()
        
        psnrs, ssims, lpips_vals = [], [], []
        cc_psnrs, cc_ssims, cc_lpips_vals = [], [], []
        
        for i in range(len(val_dataset)):
            data = val_dataset[i]
            c2w = data["camtoworld"].unsqueeze(0)
            K = data["K"].unsqueeze(0)
            gt_img = data["image"] # [H, W, 3]
            
            viewmats = torch.linalg.inv(c2w)
            colors_sh = torch.cat([splats["sh0"], splats["shN"]], dim=1)
            
            render_colors, _, _ = rasterization(
                means=splats["means"],
                quats=splats["quats"],
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=colors_sh,
                viewmats=viewmats,
                Ks=K,
                width=val_dataset.W,
                height=val_dataset.H,
                sh_degree=self.cfg.sh_degree,
                packed=False,
                render_mode="RGB"
            )
            
            pred_img = torch.clamp(render_colors.squeeze(0)[..., :3], 0.0, 1.0)
            
            # Reshape for torchmetrics: [B, C, H, W]
            gt_img_p = gt_img.permute(2, 0, 1).unsqueeze(0)
            pred_img_p = pred_img.permute(2, 0, 1).unsqueeze(0)
            
            psnrs.append(self.psnr_metric(pred_img_p, gt_img_p).item())
            ssims.append(self.ssim_metric(pred_img_p, gt_img_p).item())
            lpips_vals.append(self.lpips_metric(pred_img_p, gt_img_p).item())
            
            if self.cfg.use_color_correction_metric:
                if self.cfg.color_correct_method == "affine":
                    cc_pred = color_correct_affine(pred_img.unsqueeze(0), gt_img.unsqueeze(0))
                else:
                    cc_pred = color_correct_quadratic(pred_img.unsqueeze(0), gt_img.unsqueeze(0))
                
                cc_pred_p = cc_pred.permute(0, 3, 1, 2)
                cc_psnrs.append(self.psnr_metric(cc_pred_p, gt_img_p).item())
                cc_ssims.append(self.ssim_metric(cc_pred_p, gt_img_p).item())
                cc_lpips_vals.append(self.lpips_metric(cc_pred_p, gt_img_p).item())

        # Resource Metrics
        elapsed_time = time.time() - start_time
        vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        num_gs = len(splats["means"])
        
        eval_stats = {
            "step": step,
            "psnr": np.mean(psnrs),
            "ssim": np.mean(ssims),
            "lpips": np.mean(lpips_vals),
            "num_gaussians": num_gs,
            "vram_gb": vram_gb,
            "training_time_hrs": elapsed_time / 3600.0
        }
        
        if self.cfg.use_color_correction_metric:
            eval_stats.update({
                "cc_psnr": np.mean(cc_psnrs),
                "cc_ssim": np.mean(cc_ssims),
                "cc_lpips": np.mean(cc_lpips_vals),
            })
            
        if self.cfg.use_color_correction_metric:
            eval_stats.update({
                "cc_psnr": np.mean(cc_psnrs), "cc_ssim": np.mean(cc_ssims), "cc_lpips": np.mean(cc_lpips_vals),
            })
            
        os.makedirs(stats_dir, exist_ok=True)
        with open(os.path.join(stats_dir, f"val_step{step:04d}.json"), "w") as f:
            json.dump(eval_stats, f, indent=4)

        if wandb.run is not None:
            wandb.log({f"val/{k}": v for k, v in eval_stats.items() if k not in ["step", "num_gaussians", "vram_gb", "training_time_hrs"]}, step=step)
            wandb.log({f"resources/{k}": v for k, v in eval_stats.items() if k in ["num_gaussians", "vram_gb", "training_time_hrs"]}, step=step)
        
        print(f"PSNR: {eval_stats['psnr']:.3f} | Num GS: {num_gs} | VRAM: {vram_gb:.2f}GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone GS Evaluation Script")
    parser.add_argument("--stage", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--floor_number", type=int, required=True)
    # parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint number (e.g., ckpt_warmup_29999.pt)")
    args = parser.parse_args()
    
    print("starting evaluation")

    data_dir = f"data/{args.scene_name}/{args.floor_number}"

    cfg = PipelineConfig(scene_name=args.scene_name, version=args.version, data_dir=data_dir, floor_number=args.floor_number)
    stats_dir = f"results/{cfg.scene_name}/{args.stage}/{args.version}/{args.floor_number}/stats"
    
    val_dataset = CustomGSDataset(data_dir=cfg.data_dir, device=cfg.device, split="test", test_every=cfg.test_every)
    eval_engine = EvaluationEngine(cfg)
    
    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    stats_dir = f"results/{args.scene_name}/{args.stage}/{args.version}/{args.floor_number}/checkpoints/ckpt_{args.stage}_{args.ckpt}.pt"
    checkpoint = torch.load(args.ckpt, map_location=cfg.device, weights_only=True)
    splats = checkpoint["splats"]
    step = checkpoint.get("step", cfg.max_steps_warmup)
    
    eval_engine.evaluate(step, splats, val_dataset, start_time=time.time(), stats_dir=stats_dir)
    print("[INFO] Standalone evaluation complete.")
    
# python evaluation/evaluation.py --stage warmup --version v6.0 --scenes 6VSV7_695_v2 --floor_number 0  --ckpt 29999 --data_dir data/6VSV7_695_v2/0