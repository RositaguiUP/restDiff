import os
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def crop_and_resize_tensor(img_tensor, is_depth=False, target_size=512):
    """Crops a [H, W, C] or [H, W] tensor to square and resizes it."""
    if is_depth:
        t = img_tensor.unsqueeze(0).unsqueeze(0)
    else:
        t = img_tensor.permute(2, 0, 1).unsqueeze(0)

    H, W = t.shape[2], t.shape[3]
    size = min(H, W)
    start_y, start_x = (H - size) // 2, (W - size) // 2
    
    t = t[:, :, start_y:start_y+size, start_x:start_x+size]
    
    mode = "nearest" if is_depth else "bilinear"
    t = F.interpolate(t, size=(target_size, target_size), mode=mode, align_corners=False if not is_depth else None)
    
    if is_depth:
        return t.squeeze() 
    else:
        return t.squeeze(0).permute(1, 2, 0) 
    
def process_npy_depth(npy_path, base_is_landscape):
    depth_array = np.load(npy_path)
    depth_array = np.nan_to_num(depth_array)
    
    depth_h, depth_w = depth_array.shape
    depth_is_landscape = depth_w > depth_h
    if base_is_landscape != depth_is_landscape:
        print("    [!] Orientation mismatch detected for depth map. Rotating 90 degrees.")
        # .copy() ensures memory contiguity for PyTorch tensor conversion
        depth_array = np.rot90(depth_array, k=-1).copy()
        
    depth_tensor = torch.from_numpy(depth_array).float()
    depth_tensor = crop_and_resize_tensor(depth_tensor, is_depth=True, target_size=512)
    depth_array = depth_tensor.numpy()

    d_min, d_max = depth_array.min(), depth_array.max()
    if d_max > d_min:
        depth_norm = (depth_array - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_array)
        
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    depth_img = Image.fromarray(depth_uint8).convert("RGB")
    return depth_img

def load_and_process_rgb(img_path, base_is_landscape=None):
    pil_img = Image.open(img_path).convert("RGB")
    
    # Orientation Check (only if base_is_landscape is provided, meaning this is a condition image)
    if base_is_landscape is not None:
        img_w, img_h = pil_img.size
        img_is_landscape = img_w > img_h
        if base_is_landscape != img_is_landscape:
            print(f"    [!] Orientation mismatch detected for {os.path.basename(img_path)}. Rotating 90 degrees.")
            pil_img = pil_img.transpose(Image.ROTATE_270)
            
    tensor = torch.from_numpy(np.array(pil_img)).float() / 255.0
    tensor = crop_and_resize_tensor(tensor, is_depth=False, target_size=512)
    return Image.fromarray((tensor.cpu().numpy().clip(0, 1) * 255).astype(np.uint8))

def save_comparison(base_img, tile_img, depth_img, pred_img, output_path, description, model_label, cond_str):
    """Saves a comparison plot dynamically based on the available conditions."""
    num_plots = 4 if depth_img is not None else 3
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 5))
    
    fig.suptitle(f"{description} | Model: {model_label}", fontsize=16, fontweight='bold', y=0.98)
    
    axes[0].imshow(base_img)
    axes[0].set_title("Base Image (Render RGB)", fontsize=14)
    axes[0].axis("off")
    
    axes[1].imshow(tile_img)
    axes[1].set_title(f"Tile Condition ({cond_str})", fontsize=14)
    axes[1].axis("off")
    
    if depth_img is not None:
        axes[2].imshow(depth_img)
        axes[2].set_title(f"Depth Condition ({cond_str})", fontsize=14)
        axes[2].axis("off")
        
        axes[3].imshow(pred_img)
        axes[3].set_title("Prediction", fontsize=14)
        axes[3].axis("off")
    else:
        axes[2].imshow(pred_img)
        axes[2].set_title("Prediction", fontsize=14)
        axes[2].axis("off")
        
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

# ==========================================
# MAIN EXECUTION
# ==========================================
def main(args):
    # 1. LOAD MODELS (Once per script execution)
    print(f"[+] Loading ControlNet (Tile) from: {args.controlnet_tile_path}")
    controlnet_tile = ControlNetModel.from_pretrained(args.controlnet_tile_path, torch_dtype=torch.float16)
    
    if "multi" in args.model_label:
        print(f"[+] Loading ControlNet (Depth) from: {args.controlnet_depth_path}")
        controlnet_depth = ControlNetModel.from_pretrained(args.controlnet_depth_path, torch_dtype=torch.float16)
        controlnets = [controlnet_tile, controlnet_depth]
    else:
        controlnets = controlnet_tile

    print(f"[+] Loading Base Pipeline ({args.base_model})...")
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnets,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    generator = torch.manual_seed(42)
    

    # 2. ITERATE OVER SCENES AND FLOORS
    for scene in args.scenes:
        for floor in args.floors:
            print(f"\n[*] Processing Scene: {scene} | Floor: {floor}")
            
            mode = None
            conditional = None
            gt_type = None
            
            for config_str in args.configs:
                config = json.loads(config_str)
                mode = config['mode']
                conditional = config['conditional']
                gt_type = config.get('gt_type', 'none')
                
                print(f"\n[*] Running Batch Task: {mode} | {conditional} | {gt_type}")
            
                # Sanity check for DSLR + Multi
                if conditional == "gt" and gt_type == "dslr" and mode == "multi":
                    raise ValueError("Configuration Error: DSLR does not have depth maps. Cannot run mode 'multi' with gt_type 'dslr'.")

                # Determine condition string for output directory
                if conditional == "render":
                    cond_str = "render"
                elif conditional == "gt":
                    cond_str = f"gt_{gt_type}"
                else:
                    raise ValueError("Invalid conditional specified.")

                print(f"\n[+] Initialization: Mode={mode.upper()}, Conditional={cond_str.upper()}")

                
                renders_path = f"results/{scene}/{args.stage}/{args.version}/{floor}/renders/{args.poses_version}"
                renders_rgb_dir = os.path.join(renders_path, "rgb")
                
                if not os.path.exists(renders_rgb_dir):
                    print(f"[!] Warning: RGB renders path does not exist. Skipping... ({renders_rgb_dir})")
                    continue
                

                out_dir = f"results/{scene}/{args.stage}/{args.version}/{floor}/predictions/{args.output_subfolder}/{args.poses_version}/{cond_str}/{args.strength}"
                os.makedirs(out_dir, exist_ok=True)

                # Get list of images
                img_files = [f for f in os.listdir(renders_rgb_dir) if f.endswith(('.png', '.jpg'))]
                
                # 3. ITERATE OVER IMAGES IN THE FLOOR
                for img_file in img_files:
                    img_id = os.path.splitext(img_file)[0]
                    
                    # Base image is ALWAYS the RGB render
                    base_img_path = os.path.join(renders_rgb_dir, img_file)
                    
                    # Check orientation layout of the base render before processing targets
                    with Image.open(base_img_path) as tmp_img:
                        base_w, base_h = tmp_img.size
                        base_is_landscape = base_w > base_h
                    
                    # Resolve Conditional Paths
                    tile_path = None
                    depth_path = None

                    if conditional == "render":
                        tile_path = base_img_path
                        if mode == "multi":
                            depth_path = os.path.join(renders_path, "depth", f"{img_id}.npy")
                    
                    elif conditional == "gt":
                        if gt_type == "scan":
                            tile_path = f"data/{scene}/{floor}/rgb/{img_id}.png"
                            if mode == "multi":
                                depth_path = f"data/{scene}/{floor}/depth/{img_id}.npy"
                        
                        elif gt_type == "dslr":
                            tile_path = f"/home/rosita/tests/data/{scene}/photo_alignment/images/images_resized/{img_id}.png"
                            depth_path = None # DSLR has no depth

                    # Verify paths exist before processing
                    if not os.path.exists(tile_path):
                        print(f"[!] Missing Tile Image for ID {img_id}. Skipping... ({tile_path})")
                        continue
                    
                    if mode == "multi" and depth_path and not os.path.exists(depth_path):
                        print(f"[!] Missing Depth NPY for ID {img_id}. Skipping... ({depth_path})")
                        continue

                    # Load & Process Images
                    base_img = load_and_process_rgb(base_img_path, base_is_landscape=None)
                    tile_img = load_and_process_rgb(tile_path, base_is_landscape=base_is_landscape)
                    depth_img = process_npy_depth(depth_path, base_is_landscape) if mode == "multi" else None

                    if mode == "multi":
                        control_images = [tile_img, depth_img]
                        conditioning_scale = [1.0, 1.0]
                    else:
                        control_images = tile_img
                        conditioning_scale = 0.4

                    # Generate
                    print(f"    -> Running inference for {img_id}...")
                    result_img = pipe(
                        prompt=args.prompt,
                        negative_prompt=args.negative_prompt,
                        image=base_img, # Image-to-Image base
                        control_image=control_images, # ControlNet condition(s)
                        controlnet_conditioning_scale=conditioning_scale,
                        strength=args.strength,
                        num_inference_steps=20,
                        guidance_scale=7.5,
                        generator=generator
                    ).images[0]

                    # Save output and comparison
                    pred_out_path = os.path.join(out_dir, f"{img_id}.png")
                    result_img.save(pred_out_path)
                    
                    comp_out_path = os.path.join(out_dir, f"{img_id}_comparison.png")
                    save_comparison(base_img, tile_img, depth_img, result_img, comp_out_path,
                                    description=args.description, model_label=args.model_label, cond_str=cond_str)

                print(f"Images saved in {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Lists
    parser.add_argument("--scenes", nargs='+', default=["2F5Z7_007"], help="List of scenes")
    parser.add_argument("--floors", nargs='+', default=["0"], help="List of floors")
    
    # Path configuration variables
    parser.add_argument("--stage", type=str, default="warmup")
    parser.add_argument("--version", type=str, default="v6.0")
    parser.add_argument("--poses_version", type=str, default="trajectory_inter_10")
    
    # Paths parsed from Bash script execution
    parser.add_argument("--controlnet_tile_path", type=str, required=True)
    parser.add_argument("--controlnet_depth_path", type=str, default="")
    parser.add_argument("--output_subfolder", type=str, required=True, help="e.g. model_version/epoch")
    
    # Plot Text Elements
    parser.add_argument("--model_label", type=str, required=True)
    parser.add_argument("--description", type=str, default="Inferences")
    
    # Run Modes
    parser.add_argument("--configs", nargs='+', required=True, help="List of JSON config strings (e.g., '{\"mode\": \"tile\"}')")
    
    # Pipeline Settings
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--base_model", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--prompt", type=str, default="photorealistic, highly detailed indoor, sharp textures, clean architecture")
    parser.add_argument("--negative_prompt", type=str, default="blurry, artifacts, floating objects, people, distortion, deformed")

    args = parser.parse_args()
    main(args)