import os
import torch
import math
import numpy as np
from torch import Tensor
from sklearn.neighbors import NearestNeighbors


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Converts RGB to Spherical Harmonics (SH) DC component."""
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0
  
def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)

def create_splats_with_optimizers(ply_path, num_random_pts=100000, sh_degree=3, device="cuda"):
    """Initializes GS parameters and Adam optimizers mirroring simple_trainer.py"""
    
    # 1. Load Point Cloud
    if os.path.exists(ply_path):
        from plyfile import PlyData
        plydata = PlyData.read(ply_path)
        v = plydata['vertex']
        points = np.vstack([v['x'], v['y'], v['z']]).T
        if 'red' in v:
            rgbs = np.vstack([v['red'], v['green'], v['blue']]).T / 255.0
        else:
            rgbs = np.random.rand(points.shape[0], 3)
        
        points = torch.from_numpy(points).float().to(device)
        rgbs = torch.from_numpy(rgbs).float().to(device)
        print(f"[INFO] Loaded {points.shape[0]} points from {ply_path}")
    else:
        points = (torch.rand((num_random_pts, 3)) * 2 - 1).to(device)
        rgbs = torch.rand((num_random_pts, 3)).to(device)
        print("[INFO] PLY not found. Using random initialization.")

    N = points.shape[0]

    # 2. Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * 1.0).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # 3. Initialize other attributes
    quats = torch.rand((N, 4), device=device)
    quats[:, 0] = 1.0
    opacities = torch.logit(torch.full((N,), 0.1, device=device))
    
    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3), device=device)
    colors[:, 0, :] = rgb_to_sh(rgbs)

    # 4. Parameter Dictionary
    splats = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(points),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(colors[:, :1, :]),
        "shN": torch.nn.Parameter(colors[:, 1:, :])
    })

    # 5. Optimizers Setup
    params_groups = [
        ("means", splats["means"], 1.6e-4),
        ("scales", splats["scales"], 5e-3),
        ("quats", splats["quats"], 1e-3),
        ("opacities", splats["opacities"], 5e-2),
        ("sh0", splats["sh0"], 2.5e-3),
        ("shN", splats["shN"], 2.5e-3 / 20),
    ]

    optimizers = {
        name: torch.optim.Adam([{"params": param, "lr": lr, "name": name}], eps=1e-15)
        for name, param, lr in params_groups
    }

    return splats, optimizers