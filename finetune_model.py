import gc
import os
import json
import random
import warnings
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import wandb

from diffusers import StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Suppress verbose warnings from libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# CONFIGURATION 
# ==========================================
DEBUG_MODE = False  # <--- SET TO FALSE WHEN YOU ARE READY FOR THE REAL RUN

MODE = "tile" # Options: "tile" or "multi"
TRAIN_CONDITION = "gt"  # Options: "render" or "gt"
DATASET_TYPE = "scan" # Options: "scan" or "dslr"
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
FINAL_DIR = "./finetuned_models/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


BASE_DIR = "./dataset/hd/warmup/v6.0/29999/" if DATASET_TYPE == "dslr" else "./dataset/warmup/v6.0/29999/"
JSON_FILE = "finetune_meta_hd.json" if DATASET_TYPE == "dslr" else "finetune_meta_all_b40.json"

BATCH_SIZE = 4 if MODE == "tile" else 2
GRADIENT_ACCUMULATION_STEPS = 2 if MODE == "tile" else 4
LEARNING_RATE = 1e-5 
EPOCHS = 10 if not DEBUG_MODE else 2 # Run only 2 epochs in debug mode
VAL_SPLIT_RATIO = 0.10 
PROMPT_DROPOUT_RATE = 0.20 
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


# Initialize these globally at the top of your script
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(DEVICE)

def run_visual_validation(epoch, val_dataloader, vae, text_encoder, tokenizer, unet, controlnet_tile, controlnet_depth, noise_scheduler, device, mode):
    print(f"\n[+] Executing Epoch {epoch} Global Evaluation across Val Set...")

    # 1. Setup Img2Img Pipeline
    if mode == "multi":
        cnet_models = [controlnet_tile, controlnet_depth]
    else:
        cnet_models = controlnet_tile

    val_pipe = StableDiffusionControlNetImg2ImgPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        controlnet=cnet_models, 
        scheduler=UniPCMultistepScheduler.from_config(noise_scheduler.config),
        safety_checker=None, feature_extractor=None
    ).to(device)
    val_pipe.set_progress_bar_config(disable=True)

    # Initialize tracking variables
    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
    num_batches = 0
    log_images = []
    images_logged = 0
    max_images_to_log = 5 # Strictly limit WandB media storage

    # 2. Iterate over the ENTIRE validation dataloader
    with torch.no_grad(), torch.autocast("cuda"):
        for batch in tqdm(val_dataloader, desc="Calculating Metrics"):
            gt_tensors_unnorm = batch["cond_tile"].to(device) * 0.5 + 0.5
            texts = batch["text"]
            
            # Convert base tensors back to PIL for the pipeline
            base_imgs = [transforms.ToPILImage()(img) for img in batch["base_img"]]
            
            cond_imgs = gt_tensors_unnorm

            if mode == "multi":
                depth_imgs = [transforms.ToPILImage()(img) for img in batch["cond_depth"]]
                cnet_images = [[cond_imgs[j], depth_imgs[j]] for j in range(len(cond_imgs))]
            else:
                cnet_images = cond_imgs

            # 3. Img2Img INFERENCE 
            pred_imgs = val_pipe(
                prompt=texts,          
                image=base_imgs,         # <-- The starting image to add noise to (Render)
                control_image=cnet_images, # <-- The structural guidance (Render / Render Depth)
                strength=0.5,            # <-- 0.5 means 50% noise is added to the render
                num_inference_steps=20,
                guidance_scale=7.0
            ).images

            # 4. Calculate Metrics for this batch
            pred_tensors = torch.stack([transforms.ToTensor()(img) for img in pred_imgs]).to(device)

            total_psnr += psnr_metric(pred_tensors, gt_tensors_unnorm).item()
            total_ssim += ssim_metric(
                pred_tensors.unsqueeze(1) if pred_tensors.ndim == 3 else pred_tensors, 
                gt_tensors_unnorm.unsqueeze(1) if gt_tensors_unnorm.ndim == 3 else gt_tensors_unnorm
            ).item()
            total_lpips += lpips_metric(pred_tensors * 2.0 - 1.0, gt_tensors_unnorm * 2.0 - 1.0).item()
            num_batches += 1

            # 5. Grab exactly 5 images total for WandB (across all batches)
            for j in range(len(cond_imgs)):
                if images_logged < max_images_to_log:
                    gt_img = transforms.ToPILImage()(gt_tensors_unnorm[j].cpu())
                    log_images.append(wandb.Image(cond_imgs[j], caption=f"Sample {images_logged}: GT Scan"))
                    if mode == "multi":
                        log_images.append(wandb.Image(depth_imgs[j], caption=f"Sample {images_logged}: GT Depth"))
                    log_images.append(wandb.Image(pred_imgs[j], caption=f"Sample {images_logged}: Enhanced"))
                    log_images.append(wandb.Image(gt_img, caption=f"Sample {images_logged}: 3DGS Render"))
                    images_logged += 1

    # 6. Calculate True Global Averages
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    avg_lpips = total_lpips / num_batches
    
    # Notice: NO step=epoch. We let WandB increment naturally, but chart against the "epoch" metric below.
    wandb.log({
        "Validation/PSNR": psnr_val.item(),
        "Validation/SSIM": ssim_val.item(),
        "Validation/LPIPS": lpips_val.item(),
        "Visualization/Validation Grid": log_images,
        "epoch": epoch
    })
    
    del val_pipe
    torch.cuda.empty_cache()
    gc.collect()


def run_dual_visual_validation(epoch, val_dataloader, vae, text_encoder, tokenizer, unet, controlnet_tile, controlnet_depth, noise_scheduler, device, mode):
    print(f"\n[+] Executing Epoch {epoch} DUAL Evaluation across Val Set...")

    if mode == "multi":
        cnet_models = [controlnet_tile, controlnet_depth]
    else:
        cnet_models = controlnet_tile

    val_pipe = StableDiffusionControlNetImg2ImgPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        controlnet=cnet_models, scheduler=UniPCMultistepScheduler.from_config(noise_scheduler.config),
        safety_checker=None, feature_extractor=None
    ).to(device)
    val_pipe.set_progress_bar_config(disable=True)

    # Track metrics for both runs separately
    metrics = {"render": {"psnr": 0, "ssim": 0, "lpips": 0}, "gt": {"psnr": 0, "ssim": 0, "lpips": 0}}
    num_batches = 0
    log_images = []
    images_logged = 0

    with torch.no_grad(), torch.autocast("cuda"):
        for batch in tqdm(val_dataloader, desc="Calculating Dual Metrics"):
            gt_tensors_unnorm = batch["pixel_values"].to(device) * 0.5 + 0.5
            texts = batch["text"]
            
            # Base Image (Always Render)
            base_imgs = [transforms.ToPILImage()(img * 0.5 + 0.5) for img in batch["base_values"]]
            
            # Prepare both sets of conditions
            # cond_tile_render = [transforms.ToPILImage()(img) for img in batch["cond_tile_render"]]
            # cond_tile_gt = [transforms.ToPILImage()(img) for img in batch["cond_tile_gt"]]
            
            tile_render_tensor = batch["cond_tile_render"]
            tile_gt_tensor = batch["cond_tile_gt"]
            
            if mode == "multi":
                cond_depth_render = [transforms.ToPILImage()(img) for img in batch["cond_depth_render"]]
                cond_depth_gt = [transforms.ToPILImage()(img) for img in batch["cond_depth_gt"]]
                
                depth_render_tensor = batch["cond_depth_render"]
                depth_gt_tensor = batch["cond_depth_gt"]
                
                # cnet_imgs_render = [[cond_tile_render[j], cond_depth_render[j]] for j in range(len(base_imgs))]
                # cnet_imgs_gt = [[cond_tile_gt[j], cond_depth_gt[j]] for j in range(len(base_imgs))]
                # cnet_imgs_render = [cond_tile_render, cond_depth_render]
                # cnet_imgs_gt = [cond_tile_gt, cond_depth_gt]
                cnet_imgs_render = [tile_render_tensor, depth_render_tensor]
                cnet_imgs_gt = [tile_gt_tensor, depth_gt_tensor]
            else:
                cnet_imgs_render = tile_render_tensor
                cnet_imgs_gt = tile_gt_tensor

            # --- INFERENCE 1: Conditioned on Render ---
            pred_imgs_render = val_pipe(
                prompt=texts, image=base_imgs, control_image=cnet_imgs_render, 
                strength=0.5, num_inference_steps=20, guidance_scale=7.0
            ).images
            
            # --- INFERENCE 2: Conditioned on GT ---
            pred_imgs_gt = val_pipe(
                prompt=texts, image=base_imgs, control_image=cnet_imgs_gt, 
                strength=0.5, num_inference_steps=20, guidance_scale=7.0
            ).images

            # Calculate and accumulate metrics for both
            pred_tensors_render = torch.stack([transforms.ToTensor()(img) for img in pred_imgs_render]).to(device)
            pred_tensors_gt = torch.stack([transforms.ToTensor()(img) for img in pred_imgs_gt]).to(device)

            for key, pred in zip(["render", "gt"], [pred_tensors_render, pred_tensors_gt]):
                metrics[key]["psnr"] += psnr_metric(pred, gt_tensors_unnorm).item()
                metrics[key]["ssim"] += ssim_metric(
                    pred.unsqueeze(1) if pred.ndim == 3 else pred, 
                    gt_tensors_unnorm.unsqueeze(1) if gt_tensors_unnorm.ndim == 3 else gt_tensors_unnorm
                ).item()
                metrics[key]["lpips"] += lpips_metric(pred * 2.0 - 1.0, gt_tensors_unnorm * 2.0 - 1.0).item()
            
            num_batches += 1

            # Build the 5-image WandB Visualization Grid
            for j in range(len(base_imgs)):
                if images_logged < 5:
                    gt_target_img = transforms.ToPILImage()(gt_tensors_unnorm[j].cpu())
                    
                    # Logic to identify the depth source for labeling
                    depth_label = "GT Depth" if TRAIN_CONDITION == "gt" else "Render Depth"
                    
                    log_images.append(wandb.Image(base_imgs[j], caption=f"S{images_logged}: Render Base"))
                    
                    if mode == "multi":
                        # Log the specific depth source used for THIS specific inference run
                        log_images.append(wandb.Image(depth_render_tensor[j], caption=f"S{images_logged}: Render Depth"))
                        log_images.append(wandb.Image(depth_gt_tensor[j], caption=f"S{images_logged}: GT Depth"))
                    
                    # We log a sequence: [Render Input] -> [Pred using Render] -> [Pred using GT] -> [GT Target]
                    log_images.append(wandb.Image(gt_target_img, caption=f"S{images_logged}: GT Target"))
                    log_images.append(wandb.Image(pred_imgs_render[j], caption=f"S{images_logged}: Pred (Render Cond)"))
                    log_images.append(wandb.Image(pred_imgs_gt[j], caption=f"S{images_logged}: Pred (GT Cond)"))
                    images_logged += 1

    # Log separate metric lines to WandB so they map as two separate curves on your charts
    wandb.log({
        "Validation_RenderCond/PSNR": metrics["render"]["psnr"] / num_batches,
        "Validation_RenderCond/SSIM": metrics["render"]["ssim"] / num_batches,
        "Validation_RenderCond/LPIPS": metrics["render"]["lpips"] / num_batches,
        
        "Validation_GTCond/PSNR": metrics["gt"]["psnr"] / num_batches,
        "Validation_GTCond/SSIM": metrics["gt"]["ssim"] / num_batches,
        "Validation_GTCond/LPIPS": metrics["gt"]["lpips"] / num_batches,
        
        "Visualization/Dual Comparison Grid": log_images,
        "epoch": epoch
    })
    
    del val_pipe
    torch.cuda.empty_cache()
    gc.collect()
    
# ==========================================
# 1. CUSTOM DATASET CLASS
# ==========================================
class ScanCompletionDataset(Dataset):
    def __init__(self, data_list, mode, target_size=(512, 512), is_train=False):
        self.items = data_list
        self.mode = mode
        self.target_size = target_size
        self.is_train = is_train
        self.base_prompt = "photorealistic, sharp, highly detailed indoor architecture, clear edges, 4k texture"
        
    def __len__(self): return len(self.items)

    def _process_npy_depth(self, npy_path):
        depth_array = np.load(npy_path)
        depth_array = np.nan_to_num(depth_array)
        d_min, d_max = depth_array.min(), depth_array.max()
        depth_norm = (depth_array - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(depth_array)
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        return Image.fromarray(depth_uint8).convert("RGB")

    def __getitem__(self, idx):
        item = self.items[idx]
        gt_rgb = Image.open(item["gt_rgb_path"]).convert("RGB")
        render_rgb = Image.open(item["render_rgb_path"]).convert("RGB")
        
        # Determine if we should flip this specific frame
        flip = self.is_train and random.random() > 0.5
        
        if flip:
            gt_rgb = gt_rgb.transpose(Image.FLIP_LEFT_RIGHT)
            render_rgb = render_rgb.transpose(Image.FLIP_LEFT_RIGHT)
        
        # --- BASE IMAGE & TARGET ---
        # The image to add noise to (always the Render)
        base_tensor = transforms.Compose([
            transforms.Resize(self.target_size), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
        ])(render_rgb)
        
        # The ground truth target for the loss function
        gt_tensor = transforms.Compose([
            transforms.Resize(self.target_size), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
        ])(gt_rgb)
        
        # --- TILE CONDITIONS ---
        cond_tile_render = transforms.Compose([transforms.Resize(self.target_size), transforms.ToTensor()])(render_rgb)
        cond_tile_gt = transforms.Compose([transforms.Resize(self.target_size), transforms.ToTensor()])(gt_rgb)
        
        prompt = self.base_prompt if not (self.is_train and random.random() < PROMPT_DROPOUT_RATE) else ""
        
        batch = {
            "base_values": base_tensor,   # Input to VAE (Render)
            "pixel_values": gt_tensor,    # Target for Loss (GT)
            "cond_tile_render": cond_tile_render, 
            "cond_tile_gt": cond_tile_gt,
            "text": prompt
        }
        
        # --- DEPTH CONDITIONS ---
        if self.mode == "multi":
            # Process Render Depth
            render_depth_img = self._process_npy_depth(item["render_depth_path"])
            # Process GT Depth (You need to make sure you have this path in your JSON!)
            gt_depth_img = self._process_npy_depth(item["gt_depth_path"]) 
            
            if flip:
                render_depth_img = render_depth_img.transpose(Image.FLIP_LEFT_RIGHT)
                gt_depth_img = gt_depth_img.transpose(Image.FLIP_LEFT_RIGHT)
                
            batch["cond_depth_render"] = transforms.Compose([transforms.Resize(self.target_size), transforms.ToTensor()])(render_depth_img)
            batch["cond_depth_gt"] = transforms.Compose([transforms.Resize(self.target_size), transforms.ToTensor()])(gt_depth_img)
            
        return batch

# ==========================================
# 2. SETUP & WANDB
# ==========================================
with open(BASE_DIR + JSON_FILE, "r") as f: 
    full_data = json.load(f)

if DEBUG_MODE:
    print("\n[!!!] DEBUG MODE IS ACTIVE [!!!]")
    print("[!] Slicing dataset to only 20 images for a rapid test run.")
    random.shuffle(full_data)
    full_data = full_data[:20]

split_idx = int(len(full_data) * (1 - VAL_SPLIT_RATIO))
train_data = full_data[:split_idx]
val_data = full_data[split_idx:]

train_dataset = ScanCompletionDataset(train_data, mode=MODE, is_train=True)
val_dataset = ScanCompletionDataset(val_data, mode=MODE, is_train=False)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

run_name = f"{MODE}_{TRAIN_CONDITION}_{DATASET_TYPE}_DEBUG" if DEBUG_MODE else f"{MODE}_{TRAIN_CONDITION}_{DATASET_TYPE}"

FINAL_DIR = os.path.join(FINAL_DIR, run_name)
os.makedirs(FINAL_DIR, exist_ok=True)

wandb.init(
    project="thesis-gs-diffusion", 
    name=run_name, 
    dir=BASE_DIR,
    config={
        "debug_mode": DEBUG_MODE,
        "mode": MODE,
        "base_model": BASE_MODEL,
        "train_condition": TRAIN_CONDITION,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "json_file": JSON_FILE,
        "prompt_dropout_rate": PROMPT_DROPOUT_RATE
    }
)
wandb.define_metric("global_step")
wandb.define_metric("epoch")

wandb.define_metric("train_loss", step_metric="global_step")
wandb.define_metric("Validation/*", step_metric="epoch")
wandb.define_metric("Visualization/*", step_metric="epoch")

# ==========================================
# 3. INITIALIZE MODELS 
# ==========================================
print(f"[+] Loading Base Model: {BASE_MODEL}")
tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder", use_safetensors=True).to(DEVICE)
vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae", use_safetensors=True).to(DEVICE)
unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet", use_safetensors=True).to(DEVICE)
noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

controlnet_tile = ControlNetModel.from_pretrained("lllyasviel/control_v11f1e_sd15_tile").to(DEVICE)
controlnet_tile.train()

# Multi-ControlNet initialization logic
params_to_optimize = list(controlnet_tile.parameters())

if MODE == "multi":
    controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth").to(DEVICE)
    controlnet_depth.train()
    params_to_optimize += list(controlnet_depth.parameters())
else:
    controlnet_depth = None

optimizer = torch.optim.AdamW(params_to_optimize, lr=LEARNING_RATE)



# ==========================================
# 4. TRAINING LOOP
# ==========================================
global_step = 0
for epoch in range(EPOCHS):
    controlnet_tile.train()
    if MODE == "multi": controlnet_depth.train()
        
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    
    optimizer.zero_grad()
    
    epoch_train_loss = 0.0
    num_batches = 0
    
    for step, batch in enumerate(progress_bar):
        latents = vae.encode(batch["base_values"].to(DEVICE)).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=DEVICE).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        if TRAIN_CONDITION == "render":
            cond_tile = batch["cond_tile_render"].to(DEVICE)
            if MODE == "multi": cond_depth = batch["cond_depth_render"].to(DEVICE)
        else:
            cond_tile = batch["cond_tile_gt"].to(DEVICE)
            if MODE == "multi": cond_depth = batch["cond_depth_gt"].to(DEVICE)
        
        
        text_inputs = tokenizer(batch["text"], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").to(DEVICE)
        encoder_hidden_states = text_encoder(text_inputs.input_ids)[0]
        
        # Multi-ControlNet math processing
        if MODE == "multi":
            down_tile, mid_tile = controlnet_tile(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_cond=cond_tile, return_dict=False)
            down_depth, mid_depth = controlnet_depth(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_cond=cond_depth, return_dict=False)
            down_block_res = [t + d for t, d in zip(down_tile, down_depth)]
            mid_block_res = mid_tile + mid_depth
        else:
            down_block_res, mid_block_res = controlnet_tile(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_cond=cond_tile, return_dict=False)
            
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                          down_block_additional_residuals=down_block_res, mid_block_additional_residual=mid_block_res, return_dict=False)[0]
        
        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        loss = loss / GRADIENT_ACCUMULATION_STEPS 
        loss.backward()
        
        current_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
        epoch_train_loss += current_loss
        num_batches += 1
        
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        wandb.log({
            "train_loss": loss.item() * GRADIENT_ACCUMULATION_STEPS, 
            "global_step": global_step
        })
        progress_bar.set_postfix({"loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}"})
        global_step += 1

    # Validation
    run_dual_visual_validation(epoch + 1, val_dataloader, vae, text_encoder, tokenizer, unet, controlnet_tile, controlnet_depth, noise_scheduler, DEVICE, MODE)
    
    # Save checkpoints dynamically based on mode
    epoch_dir = f"controlnet_epoch_{epoch+1}_DEBUG" if DEBUG_MODE else f"controlnet_epoch_{epoch+1}"
    controlnet_tile.save_pretrained(os.path.join(FINAL_DIR, epoch_dir, "tile"))
    if MODE == "multi":
        controlnet_depth.save_pretrained(os.path.join(FINAL_DIR, epoch_dir, "depth"))

final_name = "final_DEBUG" if DEBUG_MODE else "final"
controlnet_tile.save_pretrained(os.path.join(FINAL_DIR, final_name, "tile"))
if MODE == "multi":
    controlnet_depth.save_pretrained(os.path.join(FINAL_DIR, final_name, "depth"))
    
wandb.finish()
print("\n[+] Pipeline execution completed successfully!")