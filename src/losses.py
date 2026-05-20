import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure

class LossEngine:
    def __init__(self, device: str = "cuda"):
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def compute_official_gs_losses(self, pred_img: torch.Tensor, gt_img: torch.Tensor, lambda_ssim: float = 0.2):
        """Computes the official GSPLAT loss: L1 + (1 - SSIM)."""
        l1_loss, ssim_loss = self.compute_rgb_losses(pred_img, gt_img)
        total_loss = (1 - lambda_ssim) * l1_loss + lambda_ssim * ssim_loss
        return total_loss
    
    def compute_rgb_losses(self, pred_img: torch.Tensor, gt_img: torch.Tensor):
        """Calculates structural and pixel absolute differences."""
        l1 = F.l1_loss(pred_img, gt_img)
        
        pred_bchw = pred_img.permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
        gt_bchw = gt_img.permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
        ssim = 1.0 - self.ssim_metric(pred_bchw, gt_bchw)
        
        return l1, ssim

    def compute_pearson_depth_loss(self, pred_depth: torch.Tensor, gt_depth: torch.Tensor):
        """Computes scale-invariant Pearson depth alignment."""
        valid_mask = gt_depth > 0
        if valid_mask.sum() < 10:
            return torch.tensor(0.0, device=pred_depth.device)
        
        p_d = pred_depth[valid_mask]
        g_d = gt_depth[valid_mask]
        
        p_centered = p_d - p_d.mean()
        g_centered = g_d - g_d.mean()
        
        cov = (p_centered * g_centered).sum()
        std_p = torch.sqrt((p_centered ** 2).sum() + 1e-8)
        std_g = torch.sqrt((g_centered ** 2).sum() + 1e-8)
        
        pearson_corr = cov / (std_p * std_g)
        return 1.0 - pearson_corr