import argparse
import json
import random
import sys
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from gsplat.rendering import rasterization

from src.dataset import CustomGSDataset

def main():
    parser = argparse.ArgumentParser(description="Extract novel view test poses, render RGB+D, and plot comparisons.")
    parser.add_argument("--stage", required=True, help="Target stage (e.g., 'warmup')")
    parser.add_argument("--version", required=True, help="Target version (e.g., 'v5.1')")
    parser.add_argument("--steps", type=int, nargs="+", required=True, help="One or more step integers")
    parser.add_argument("--scenes", nargs="*", help="Specific scenes to parse. Parses all if omitted.")
    parser.add_argument("--results-dir", default="results", help="Base results directory")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--test-every", type=int, default=8, help="Test split slice interval")
    parser.add_argument("--max-visualizations", type=int, default=None, help="Max random frames to sample and plot per run")
    parser.add_argument("--device", default="cuda", help="Computation target device")
    
    args = parser.parse_args()
    device = args.device
    results_base = Path(args.results_dir)
    data_base = Path(args.data_dir)

    if args.scenes:
        scenes = [results_base / scene for scene in args.scenes]
    else:
        scenes = [d for d in results_base.iterdir() if d.is_dir()]

    for step in args.steps:
        print(f"\n--- Processing Visualization Pass for Step: {step} ---")
        
        for scene_path in scenes:
            scene_name = scene_path.name
            target_dir = scene_path / args.stage / args.version

            if not target_dir.exists():
                continue

            # Find valid floor setups containing metric definitions
            for floor_path in target_dir.iterdir():
                if not floor_path.is_dir() or floor_path.name in ["checkpoints", "renders"]:
                    continue
                
                floor_name = floor_path.name
                
                # Checkpoint path is nested inside the floor folder
                ckpt_path = results_base / scene_name / args.stage / args.version / floor_name / "checkpoints" / f"ckpt_{args.stage}_{step}.pt"
                if not ckpt_path.exists():
                    print(f"    [-] Checkpoint file not found: {ckpt_path}")
                    continue
                
                # Check for poses.json (matches CustomGSDataset expectation)
                scene_data_dir = data_base / scene_name
                if (scene_data_dir / floor_name / "poses.json").exists():
                    scene_data_dir = scene_data_dir / floor_name
                
                if not (scene_data_dir / "poses.json").exists():
                    print(f"    [-] poses.json file not found at: {scene_data_dir}")
                    continue
                
                # Initialize CustomGSDataset to automatically handle loading and K-matrix logic
                dataset = CustomGSDataset(data_dir=str(scene_data_dir), device=device, split="test", test_every=args.test_every)
                
                if len(dataset) == 0:
                    continue
                
                
                # Handle deterministic sampling
                indices = list(range(len(dataset)))
                if args.max_visualizations is not None and len(indices) > args.max_visualizations:
                    print(f"    [!] Sampling {args.max_visualizations} deterministic random frames from {len(indices)} available test poses.")
                    random.seed(42)
                    indices = random.sample(indices, args.max_visualizations)

                # Save the custom poses output configuration file based on sampled subset
                poses_out_dir = scene_data_dir / "poses_to_render"
                poses_out_dir.mkdir(parents=True, exist_ok=True)
                
                subset_frames = [dataset.frames[i] for i in indices]
                poses_payload = {
                    "w": dataset.W, 
                    "h": dataset.H,
                    # We pass the fallback intrinsics from the dataset's top-level metadata if they exist
                    "fl_x": dataset.meta.get("fl_x"), "fl_y": dataset.meta.get("fl_y"),
                    "cx": dataset.meta.get("cx"), "cy": dataset.meta.get("cy"),
                    "frames": subset_frames
                }
                
                poses_json_path = poses_out_dir / f"test_step_{step}.json"
                with open(poses_json_path, "w") as f:
                    json.dump(poses_payload, f, indent=4)

                # Load Checkpoint and prepare Splats
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
                splats = checkpoint["splats"]
                colors_sh = torch.cat([splats["sh0"], splats["shN"]], dim=1)

                # Setup comparison file destination
                compare_output_dir = floor_path / "comparisons" / f"step_{step}"
                compare_output_dir.mkdir(parents=True, exist_ok=True)

                print(f"    [+] Plotting Room: {scene_name} | Floor: {floor_name}")
                
                for idx in tqdm(indices, desc="     Progress"):
                    # Retrieve all data directly from dataset tensors
                    data = dataset[idx]
                    frame_meta = dataset.frames[idx]
                    
                    c2w = data["camtoworld"]
                    viewmats = torch.linalg.inv(c2w).unsqueeze(0)
                    
                    # K is handled directly by dataset logic
                    K = data["K"].unsqueeze(0)

                    # Rasterization
                    render_outputs, _, _ = rasterization(
                        means=splats["means"],
                        quats=splats["quats"],
                        scales=torch.exp(splats["scales"]),
                        opacities=torch.sigmoid(splats["opacities"]),
                        colors=colors_sh,
                        viewmats=viewmats,
                        Ks=K,
                        width=dataset.W, height=dataset.H,
                        sh_degree=3, packed=False,
                        render_mode="RGB+ED" # Note: Switched to RGB+ED matching standard expected depth output
                    )

                    render_outputs = render_outputs.squeeze(0)
                    rendered_rgb = torch.clamp(render_outputs[..., :3], 0.0, 1.0).cpu().numpy()
                    rendered_depth = render_outputs[..., 3].cpu().numpy()
                    
                    # Extract ground truth from dataset tensors (no redundant PIL/numpy file reads)
                    target_rgb = data["image"].cpu().numpy()
                    target_depth = data["depth"].cpu().numpy()
                    
                    # Construct 2x2 visual breakdown matrix
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    vmin = min(target_depth.min(), rendered_depth.min())
                    vmax = max(target_depth.max(), rendered_depth.max())

                    orig_filename = Path(frame_meta["file_path"]).name
                    orig_stem = Path(frame_meta["file_path"]).stem

                    axes[0, 0].imshow(target_rgb)
                    axes[0, 0].set_title(f"Target RGB ({orig_filename})", fontsize=9, fontweight="bold")
                    axes[0, 0].axis("off")

                    axes[0, 1].imshow(rendered_rgb)
                    axes[0, 1].set_title(f"Rendered RGB (Step {step})", fontsize=9, fontweight="bold")
                    axes[0, 1].axis("off")

                    im_t_depth = axes[1, 0].imshow(target_depth, cmap="plasma", vmin=vmin, vmax=vmax)
                    axes[1, 0].set_title("Target Depth Map", fontsize=9, fontweight="bold")
                    axes[1, 0].axis("off")
                    fig.colorbar(im_t_depth, ax=axes[1, 0], fraction=0.046, pad=0.04)

                    im_r_depth = axes[1, 1].imshow(rendered_depth, cmap="plasma", vmin=vmin, vmax=vmax)
                    axes[1, 1].set_title("Rendered Depth (gsplat)", fontsize=9, fontweight="bold")
                    axes[1, 1].axis("off")
                    fig.colorbar(im_r_depth, ax=axes[1, 1], fraction=0.046, pad=0.04)
                    
                    plt.suptitle(f"Scene: {scene_name} | Floor: {floor_name} | Step: {step} | Frame: {orig_filename}", fontsize=11, fontweight="bold", y=0.97)
                    plt.tight_layout()
                    plt.savefig(compare_output_dir / f"{orig_stem}.png", dpi=150, bbox_inches="tight")
                    plt.close(fig)

if __name__ == "__main__":
    main()