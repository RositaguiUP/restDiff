import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from gsplat.rendering import rasterization

@torch.no_grad()
def render_custom_poses(ckpt_path, poses_json_path, output_dir, device="cuda", rot_k=-1):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Splats
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    splats = checkpoint["splats"]
    
    with open(poses_json_path, "r") as f:
        meta = json.load(f)
        
    # Get global W/H if missing from frames
    global_W = int(meta.get("w", 0))
    global_H = int(meta.get("h", 0))

    print(f"Rendering {len(meta['frames'])} novel views...")
    for idx, frame in enumerate(tqdm(meta["frames"])):
        c2w_cv = torch.tensor(frame["pose"], dtype=torch.float32, device=device)
        viewmats = torch.linalg.inv(c2w_cv).unsqueeze(0)
        
        colors_sh = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        
        # Retrieve per-frame intrinsics/dimensions if available, otherwise fallback to global meta
        fl_x = frame.get("fl_x", meta.get("fl_x"))
        fl_y = frame.get("fl_y", meta.get("fl_y"))
        cx = frame.get("cx", meta.get("cx"))
        cy = frame.get("cy", meta.get("cy"))
        W = frame.get("w", global_W)
        H = frame.get("h", global_H)
        
        K = torch.tensor([
            [fl_x, 0, cx],
            [0, fl_y, cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=device).unsqueeze(0)
        
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
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Check orientation and apply rotation if it's a portrait image
        if frame.get("orientation", "landscape") == "portrait":
            # rot_k=-1 rotates 90 degrees clockwise. rot_k=1 is counter-clockwise.
            img_uint8 = np.rot90(img_uint8, k=rot_k)

        Image.fromarray(img_uint8).save(os.path.join(output_dir, f"render_{idx:04d}.png"))
        
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render custom poses from a Gaussian Splatting checkpoint.")
    parser.add_argument("--scene", type=str, required=True, help="Scene name (e.g., bicycle)")
    parser.add_argument("--floor", type=str, required=True, help="Floor name (e.g., 0)")
    parser.add_argument("--version", type=str, required=True, help="Experiment version (e.g., v1)")
    parser.add_argument("--stage", type=str, required=True, help="Stage of training (e.g., fine)")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint identifier (e.g., 30000)")
    parser.add_argument("--poses_filename", type=str, required=True, help="Name of the poses JSON file without extension")
    parser.add_argument("--rot_k", type=int, default=-1, help="Number of 90-degree rotations for portrait images. -1 = 90deg CW, 1 = 90deg CCW")
    args = parser.parse_args()
    
    # Automatically construct paths based on the provided arguments
    ckpt_path = f"results/{args.scene}/{args.stage}/{args.version}/{args.floor}/checkpoints/ckpt_{args.stage}_{args.ckpt}.pt"
    poses_json_path = f"data/{args.scene}/{args.floor}/poses_to_render/{args.poses_filename}.json"
    output_dir = f"results/{args.scene}/{args.stage}/{args.version}/{args.floor}/renders/{args.poses_filename}"
    
    print(f"--- Paths Configured ---")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Poses JSON: {poses_json_path}")
    print(f"Output Dir: {output_dir}")
    print(f"Rotation k: {args.rot_k}")
    print(f"------------------------\n")
    
    render_custom_poses(ckpt_path, poses_json_path, output_dir, device="cuda", rot_k=args.rot_k)
    
# python rendering/render_custom_poses.py --scene 2F5Z7_007 --floor 0 --version v6.0 --stage warmup --ckpt 29999 --poses_filename trajectory_inter_15 --rot_k 1
# results/2F5Z7_007/0/warmup/v6.0/checkpoints/ckpt_warmup_29999.pt
# --poses_json data/2F5Z7_007/0/trajectory_inter_15.json --output_dir path/to/output/dir

# version, scene, ckpt, poses_json  data/2F5Z7_007/0/trajectory_inter_15.json