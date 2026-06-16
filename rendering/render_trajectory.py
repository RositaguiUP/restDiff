import os
import torch
import imageio
import numpy as np
from tqdm import tqdm
from gsplat.rendering import rasterization
from datasets.traj import generate_interpolated_path # using gsplat's built in interpolator
from src.dataset import CustomGSDataset
from src.config import PipelineConfig

@torch.no_grad()
def render_trajectory(ckpt_path, data_dir, output_path, device="cuda"):
    dataset = CustomGSDataset(data_dir=data_dir, device=device, split="train")
    
    # Extract all Camera to World matrices from the dataset
    c2w_all = torch.stack([dataset[i]["camtoworld"] for i in range(len(dataset))]).cpu().numpy()
    
    # Generate a smooth interpolated path
    c2w_smooth = generate_interpolated_path(c2w_all[:, :3, :], 1) # [N, 3, 4]
    
    # Convert back to 4x4
    c2w_smooth = np.concatenate([
        c2w_smooth, 
        np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(c2w_smooth), axis=0)
    ], axis=1)
    
    c2w_smooth = torch.from_numpy(c2w_smooth).float().to(device)
    K = dataset.K.unsqueeze(0)
    
    # Load Splats
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    splats = checkpoint["splats"]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=30)
    
    print("Rendering Trajectory Video...")
    for i in tqdm(range(len(c2w_smooth))):
        c2w = c2w_smooth[i].unsqueeze(0)
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
            width=dataset.W,
            height=dataset.H,
            sh_degree=3,
            packed=False,
            render_mode="RGB"
        )
        
        img = torch.clamp(render_colors.squeeze(0)[..., :3], 0.0, 1.0).cpu().numpy()
        writer.append_data((img * 255).astype(np.uint8))
        
    writer.close()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    render_trajectory(args.ckpt, args.data_dir, args.output)
    
# python rendering/render_trajectory.py --ckpt results/2F5Z7_007/0/warmup/v6.0/checkpoints/ckpt_warmup_29999.pt --data_dir data/2F5Z7_007/0 --output results/2F5Z7_007/0/warmup/v6.0/renders/trajectory_interpolated/video.mp4