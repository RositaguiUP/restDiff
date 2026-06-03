import argparse
import json
import os
import sys
import random
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from gsplat.rendering import rasterization

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
                
                # Sourcing transforms.json (checks root or floor level folders)
                scene_data_dir = data_base / scene_name
                if (scene_data_dir / floor_name / "transforms.json").exists():
                    scene_data_dir = scene_data_dir / floor_name
                
                transforms_path = scene_data_dir / "transforms.json"
                if not transforms_path.exists():
                    print(f"    [-] transforms.json file not found at: {transforms_path}")
                    continue

                with open(transforms_path, "r") as f:
                    meta = json.load(f)

                # Split tracking matched to CustomGSDataset logic
                all_frames = meta["frames"]
                test_frames = [f for i, f in enumerate(all_frames) if i % args.test_every == 0]

                if not test_frames:
                    continue

                # Apply deterministic pseudo-random cap restriction
                if args.max_visualizations is not None and len(test_frames) > args.max_visualizations:
                    print(f"    [!] Sampling {args.max_visualizations} deterministic random frames from {len(test_frames)} available test poses.")
                    # Setting seed right before sampling ensures consistency across runs and steps
                    random.seed(42)
                    test_frames = random.sample(test_frames, args.max_visualizations)

                # Save the custom poses output configuration file
                poses_out_dir = scene_data_dir / "poses_to_render"
                poses_out_dir.mkdir(parents=True, exist_ok=True)
                poses_filename = f"test_step_{step}"
                poses_json_path = poses_out_dir / f"{poses_filename}.json"

                poses_payload = {
                    "w": meta.get("w"), "h": meta.get("h"),
                    "fl_x": meta.get("fl_x"), "fl_y": meta.get("fl_y"),
                    "cx": meta.get("cx"), "cy": meta.get("cy"),
                    "frames": test_frames
                }
                with open(poses_json_path, "w") as f:
                    json.dump(poses_payload, f, indent=4)

                # Load Checkpoint and prepare Splats
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
                splats = checkpoint["splats"]
                colors_sh = torch.cat([splats["sh0"], splats["shN"]], dim=1)

                W, H = int(poses_payload["w"]), int(poses_payload["h"])
                K = torch.tensor([
                    [poses_payload["fl_x"], 0, poses_payload["cx"]],
                    [0, poses_payload["fl_y"], poses_payload["cy"]],
                    [0, 0, 1]
                ], dtype=torch.float32, device=device).unsqueeze(0)

                # Setup comparison file destination
                compare_output_dir = floor_path / "comparisons" / f"step_{step}"
                compare_output_dir.mkdir(parents=True, exist_ok=True)

                print(f"    [+] Plotting Room: {scene_name} | Floor: {floor_name}")
                for frame in enumerate(tqdm(test_frames, desc="     Progress")):
                    c2w_cv = torch.tensor(frame["transform_matrix"], dtype=torch.float32, device=device)
                    viewmats = torch.linalg.inv(c2w_cv).unsqueeze(0)

                    # Simultaneous RGB and Depth evaluation via multi-channel rasterization
                    render_outputs, _, _ = rasterization(
                        means=splats["means"],
                        quats=splats["quats"],
                        scales=torch.exp(splats["scales"]),
                        opacities=torch.sigmoid(splats["opacities"]),
                        colors=colors_sh,
                        viewmats=viewmats,
                        Ks=K,
                        width=W, height=H,
                        sh_degree=3, packed=False,
                        render_mode="RGB+D"
                    )

                    render_outputs = render_outputs.squeeze(0)
                    rendered_rgb = torch.clamp(render_outputs[..., :3], 0.0, 1.0).cpu().numpy()
                    rendered_depth = render_outputs[..., 3].cpu().numpy()
                    
                    # Read ground-truth resources
                    target_rgb_path = scene_data_dir / frame["file_path"]
                    target_rgb = np.array(Image.open(target_rgb_path).convert("RGB")) / 255.0 if target_rgb_path.exists() else np.zeros((H, W, 3))

                    target_depth_path = scene_data_dir / frame["depth_file_path"]
                    target_depth = np.load(target_depth_path) if target_depth_path.exists() else np.zeros((H, W))

                    # Construct 2x2 visual breakdown matrix
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    vmin = min(target_depth.min(), rendered_depth.min())
                    vmax = max(target_depth.max(), rendered_depth.max())

                    axes[0, 0].imshow(target_rgb)
                    axes[0, 0].set_title(f"Target RGB ({Path(frame['file_path']).name})", fontsize=9, fontweight="bold")
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
                    
                    # Extract original RGB scan name configuration
                    orig_filename = Path(frame["file_path"]).name
                    orig_stem = Path(frame["file_path"]).stem
                    
                    plt.suptitle(f"Scene: {scene_name} | Floor: {floor_name} | Step: {step} | Frame: {orig_filename}", fontsize=11, fontweight="bold", y=0.97)
                    plt.tight_layout()
                    plt.savefig(compare_output_dir / f"{orig_stem}.png", dpi=150, bbox_inches="tight")
                    plt.close(fig)

if __name__ == "__main__":
    main()