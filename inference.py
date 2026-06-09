import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# ==========================================
# CONFIGURATION
# ==========================================
VERSION = "v0"
EPOCH = 0
MODE = "tile_only" # Options: "tile_only" or "multi_controlnet"
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5" # Or your chosen base model
CUSTOM_TILE_PATH = "lllyasviel/control_v11f1e_sd15_tile"
CUSTOM_DEPTH_PATH = "lllyasviel/control_v11f1p_sd15_depth"
# CUSTOM_TILE_PATH = f"/home/rosita/tests/diff/restDiff/finetuned_models/{VERSION}/specialized_controlnet_epoch_{EPOCH}/tile"
# CUSTOM_DEPTH_PATH = f"/home/rosita/tests/diff/restDiff/finetuned_models/{VERSION}/specialized_controlnet_epoch_{EPOCH}/depth"

INPUT_RENDER_RGB = "/home/rosita/tests/diff/restDiff/results/2F5Z7_007/warmup/v6.0/0/renders/render_0006.png"
INPUT_RENDER_DEPTH = "/home/rosita/tests/diff/restDiff/results/2F5Z7_007/warmup/v6.0/0/renders/trajectory_inter_10/render_0005_depth.npy" # Only needed if multi_controlnet
OUTPUT_PATH = f"/home/rosita/tests/diff/restDiff/results/2F5Z7_007/warmup/v6.0/0/predictions/trajectory_inter_10/prediction_0006_{VERSION}_e{EPOCH}.png"

PROMPT = "photorealistic, highly detailed indoor, sharp textures, clean architecture"
NEGATIVE_PROMPT = "blurry, artifacts, floating objects, people, distortion, deformed"

SHOW_COMPARISON = True

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def crop_and_resize_tensor(img_tensor, is_depth=False, target_size=512):
    """Crops a [H, W, C] or [H, W] tensor to square and resizes it."""
    if is_depth:
        # [H, W] -> [1, 1, H, W]
        t = img_tensor.unsqueeze(0).unsqueeze(0)
    else:
        # [H, W, C] -> [1, C, H, W]
        t = img_tensor.permute(2, 0, 1).unsqueeze(0)

    H, W = t.shape[2], t.shape[3]
    size = min(H, W)
    start_y, start_x = (H - size) // 2, (W - size) // 2
    
    # Center crop
    t = t[:, :, start_y:start_y+size, start_x:start_x+size]
    
    # Resize
    mode = "nearest" if is_depth else "bilinear"
    t = F.interpolate(t, size=(target_size, target_size), mode=mode, align_corners=False if not is_depth else None)
    
    if is_depth:
        return t.squeeze() # [H, W]
    else:
        return t.squeeze(0).permute(1, 2, 0) # [H, W, C]
    
def process_npy_depth(npy_path):
    depth_array = np.load(npy_path)
    depth_array = np.nan_to_num(depth_array)
    # Convert to tensor for cropping/resizing
    depth_tensor = torch.from_numpy(depth_array).float()
    depth_tensor = crop_and_resize_tensor(depth_tensor, is_depth=True, target_size=512)
    depth_array = depth_tensor.numpy()

    d_min, d_max = depth_array.min(), depth_array.max()
    if d_max > d_min:
        depth_norm = (depth_array - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_array)
        
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    # Already 512x512 from the tensor operation
    depth_img = Image.fromarray(depth_uint8).convert("RGB")
    return depth_img

def save_comparison(original_img, refined_img, output_path):
    """Saves a side-by-side comparison of the original render and the inference result."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_img)
    axes[0].set_title("Original Render")
    axes[0].axis("off")
    
    axes[1].imshow(refined_img)
    axes[1].set_title("Inference Result")
    axes[1].axis("off")
    
    plt.tight_layout()
    comparison_path = output_path.replace(".png", f"_comparison_{VERSION}_e{EPOCH}.png")
    plt.savefig(comparison_path)
    print(f"[*] Comparison plot saved to {comparison_path}")
    plt.close()
    
# ==========================================
# 2. LOAD MODELS
# ==========================================
print("[+] Loading fine-tuned ControlNets...")
# Use float16 for faster inference and lower VRAM usage
controlnet_tile = ControlNetModel.from_pretrained(CUSTOM_TILE_PATH, torch_dtype=torch.float16)

if MODE == "multi_controlnet":
    controlnet_depth = ControlNetModel.from_pretrained(CUSTOM_DEPTH_PATH, torch_dtype=torch.float16)
    controlnets = [controlnet_tile, controlnet_depth]
else:
    controlnets = controlnet_tile

print(f"[+] Loading Base Pipeline ({BASE_MODEL})...")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnets,
    torch_dtype=torch.float16,
    safety_checker=None # Speed up inference slightly
).to("cuda")

# UniPC is heavily recommended for ControlNet inference (fast and high quality)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# Optional: Enable memory efficient attention if VRAM is tight
pipe.enable_xformers_memory_efficient_attention() 

# ==========================================
# 3. PREPARE INPUTS
# ==========================================
render_rgb_pil = Image.open(INPUT_RENDER_RGB).convert("RGB")
rgb_tensor = torch.from_numpy(np.array(render_rgb_pil)).float() / 255.0
rgb_tensor = crop_and_resize_tensor(rgb_tensor, is_depth=False, target_size=512)
render_rgb_img = Image.fromarray((rgb_tensor.numpy() * 255).astype(np.uint8))

if MODE == "multi_controlnet":
    render_depth_img = process_npy_depth(INPUT_RENDER_DEPTH)
    # Order matters: Must match the order of controlnets list [tile, depth]
    control_images = [render_rgb_img, render_depth_img]
    # ControlNet conditioning scales (how strong each ControlNet is)
    # You may need to tune these. 1.0 is default.
    controlnet_conditioning_scale = [1.0, 1.0] 
else:
    control_images = render_rgb_img
    controlnet_conditioning_scale = 1.0

# ==========================================
# 4. GENERATE PREDICTION
# ==========================================
print("[+] Generating refined image...")
# Generator seed for reproducibility
generator = torch.manual_seed(42)

result = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    image=control_images,
    num_inference_steps=20, # 20-30 is usually plenty for UniPC
    guidance_scale=7.5, # How closely it follows the text prompt
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    generator=generator
).images[0]

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
result.save(OUTPUT_PATH)
print(f"[*] Prediction saved to {OUTPUT_PATH}")

if SHOW_COMPARISON:
    # render_rgb_img is already the 512x512 PIL image used in the pipeline
    save_comparison(render_rgb_img, result, OUTPUT_PATH)