import os
import torch
import argparse
from gsplat import export_splats

def export_checkpoint_to_ply(ckpt_path, output_path):
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    
    # Load the checkpoint
    # Note: Use weights_only=True if you are on a recent torch version
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # Extract the splats dictionary
    # Assuming your checkpoint structure is {'splats': {...}, 'step': ...}
    splats = checkpoint["splats"]
    
    print(f"[INFO] Exporting {len(splats['means'])} Gaussians to {output_path}...")
    
    # Export to PLY
    export_splats(
        means=splats["means"],
        scales=splats["scales"],
        quats=splats["quats"],
        opacities=splats["opacities"],
        sh0=splats["sh0"],
        shN=splats["shN"],
        format="ply",
        save_to=output_path
    )
    print("[INFO] Export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the .pt file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the .ply file")
    args = parser.parse_args()
    
    export_checkpoint_to_ply(args.ckpt, args.output)
    
    
# python export_ply.py \
#     --ckpt results/my_scene/warmup/v5.1/checkpoints/ckpt_warmup_30000.pt \
#     --output results/my_scene/warmup/v5.1/final_model.ply