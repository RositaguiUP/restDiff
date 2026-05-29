import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from gsplat.rendering import rasterization

@torch.no_grad()
def render_custom_poses(ckpt_path, poses_json_path, output_dir, device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Splats
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    splats = checkpoint["splats"]
    
    with open(poses_json_path, "r") as f:
        meta = json.load(f)
        
    W, H = int(meta["w"]), int(meta["h"])
    K = torch.tensor([
        [meta["fl_x"], 0, meta["cx"]],
        [0, meta["fl_y"], meta["cy"]],
        [0, 0, 1]
    ], dtype=torch.float32, device=device).unsqueeze(0)
    
    gl_to_cv = torch.tensor([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
    ], dtype=torch.float32, device=device)

    print(f"Rendering {len(meta['frames'])} novel views...")
    for idx, frame in enumerate(tqdm(meta["frames"])):
        c2w_gl = torch.tensor(frame["transform_matrix"], dtype=torch.float32, device=device)
        c2w_cv = (c2w_gl @ gl_to_cv).unsqueeze(0)
        viewmats = torch.linalg.inv(c2w_cv)
        
        colors_sh = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        
        render_colors, _, _ = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors_sh,
            viewmats=viewmats,
            Ks=K,
            width=W,
            height=H,
            sh_degree=3,
            packed=False,
            render_mode="RGB"
        )
        
        img = torch.clamp(render_colors.squeeze(0)[..., :3], 0.0, 1.0).cpu().numpy()
        Image.fromarray((img * 255).astype(np.uint8)).save(os.path.join(output_dir, f"render_{idx:04d}.png"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render custom poses from a Gaussian Splatting checkpoint.")
    parser.add_argument("--scene", type=str, required=True, help="Scene name (e.g., bicycle)")
    parser.add_argument("--version", type=str, required=True, help="Experiment version (e.g., v1)")
    parser.add_argument("--stage", type=str, required=True, help="Stage of training (e.g., fine)")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint identifier (e.g., 30000)")
    parser.add_argument("--poses_filename", type=str, required=True, help="Name of the poses JSON file without extension")
    args = parser.parse_args()
    
    # Automatically construct paths based on the provided arguments
    ckpt_path = f"results/{args.scene}/{args.stage}/{args.version}/checkpoints/ckpt_{args.stage}_{args.ckpt}.pt"
    poses_json_path = f"data/{args.scene}/poses_to_render/{args.poses_filename}.json"
    output_dir = f"results/{args.scene}/{args.stage}/{args.version}/renders/{args.poses_filename}"
    
    print(f"--- Paths Configured ---")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Poses JSON: {poses_json_path}")
    print(f"Output Dir: {output_dir}")
    print(f"------------------------\n")
    
    render_custom_poses(ckpt_path, poses_json_path, output_dir)
    
# python render_custom_poses.py --scene 6VSV7_695_v2 --version v5.1 --stage warmup --ckpt 30000 --poses_filename selected
# results/6VSV7_695_v2/warmup/v5.1/checkpoints/ckpt_warmup_30000.pt
# --poses_json data/6VSV7_695_v2/poses_to_render/selected.json --output_dir path/to/output/dir

# version, scene, ckpt, poses_json