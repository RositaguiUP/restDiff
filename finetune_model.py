import os
import json
import random
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import wandb

# Import Diffusers components
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer

# ==========================================
# CONFIGURATION Toggles
# ==========================================
MODE = "multi_controlnet" # Options: "tile_only" or "multi_controlnet"
# BASE_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE" 
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
BASE_DIR = "./dataset/warmup/v6.0/29999/"
JSON_FILE = "finetune_meta_all_b40.json"
FINAL_DIR = "./finetuned_models/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-6
EPOCHS = 10
VAL_SPLIT_RATIO = 0.10 # 10% of data used for validation
SEED = 42

# Ensure reproducibility for the split
random.seed(SEED)
torch.manual_seed(SEED)

# ==========================================
# 1. CUSTOM DATASET CLASS
# ==========================================
class ScanCompletionDataset(Dataset):
    def __init__(self, data_list, target_size=(512, 512)):
        self.items = data_list
        self.target_size = target_size
        
        # Transforms for Target Image (Ground Truth RGB) -> Normalized to [-1, 1]
        self.target_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # Transforms for Conditioning Inputs (Renders/Depths) -> Kept in [0, 1]
        self.cond_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.items)

    def _process_npy_depth(self, npy_path):
        depth_array = np.load(npy_path)
        depth_array = np.nan_to_num(depth_array)
        d_min, d_max = depth_array.min(), depth_array.max()
        if d_max > d_min:
            depth_norm = (depth_array - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_array)
            
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        depth_img = Image.fromarray(depth_uint8).convert("RGB")
        return depth_img

    def __getitem__(self, idx):
        item = self.items[idx]
        gt_rgb = Image.open(item["gt_rgb_path"]).convert("RGB")
        render_rgb = Image.open(item["render_rgb_path"]).convert("RGB")
        
        batch = {
            "pixel_values": self.target_transform(gt_rgb),
            "cond_tile": self.cond_transform(render_rgb),
            "text": "photorealistic, highly detailed indoor scan, sharp textures, clean architecture"
        }
        
        if MODE == "multi_controlnet":
            render_depth_img = self._process_npy_depth(item["render_depth_path"])
            batch["cond_depth"] = self.cond_transform(render_depth_img)
            
        return batch

# ==========================================
# 2. DATASET SPLITTING & LOADERS
# ==========================================
with open(BASE_DIR + JSON_FILE, "r") as f:
    full_data = json.load(f)

# Shuffle and Split
random.shuffle(full_data)
split_idx = int(len(full_data) * (1 - VAL_SPLIT_RATIO))
train_data = full_data[:split_idx]
val_data = full_data[split_idx:]

print(f"[+] Dataset loaded. Train: {len(train_data)} | Validation: {len(val_data)}")

train_dataset = ScanCompletionDataset(train_data)
val_dataset = ScanCompletionDataset(val_data)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 3. INITIALIZE WANDB
# ==========================================
wandb.init(
    project="thesis-gs-diffusion",
    name=f"run_{MODE}_b{BATCH_SIZE}",
    dir=BASE_DIR,
    config={
        "mode": MODE,
        "base_model": BASE_MODEL,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "JSON_FILE": JSON_FILE
    }
)

# ==========================================
# 4. INITIALIZE MODELS & APPLY FREEZING RULES
# ==========================================
print(f"[+] Loading Base Model Environment from: {BASE_MODEL}")
tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder").to(DEVICE)
vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae").to(DEVICE)
unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet").to(DEVICE)
noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

print("[+] Initializing ControlNet structures...")
controlnet_tile = ControlNetModel.from_pretrained("lllyasviel/control_v11f1e_sd15_tile").to(DEVICE)
controlnet_tile.train()

if MODE == "multi_controlnet":
    controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth").to(DEVICE)
    controlnet_depth.train()
    optimizer = torch.optim.AdamW(
        list(controlnet_tile.parameters()) + list(controlnet_depth.parameters()), 
        lr=LEARNING_RATE
    )
else:
    optimizer = torch.optim.AdamW(controlnet_tile.parameters(), lr=LEARNING_RATE)

# ==========================================
# 5. STEP-BY-STEP TRAINING & VALIDATION LOOP
# ==========================================
print(f"[+] Beginning training pipeline...")
global_step = 0

for epoch in range(EPOCHS):
    
    # --- TRAINING PHASE ---
    controlnet_tile.train()
    if MODE == "multi_controlnet": controlnet_depth.train()
        
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        
        latents = vae.encode(batch["pixel_values"].to(DEVICE)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=DEVICE).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        text_inputs = tokenizer(batch["text"], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").to(DEVICE)
        encoder_hidden_states = text_encoder(text_inputs.input_ids)[0]
        
        cond_tile = batch["cond_tile"].to(DEVICE)
        
        if MODE == "multi_controlnet":
            cond_depth = batch["cond_depth"].to(DEVICE)
            down_tile, mid_tile = controlnet_tile(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_cond=cond_tile, return_dict=False)
            down_depth, mid_depth = controlnet_depth(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_cond=cond_depth, return_dict=False)
            
            down_block_res_samples = [t + d for t, d in zip(down_tile, down_depth)]
            mid_block_res_sample = mid_tile + mid_depth
        else:
            down_block_res_samples, mid_block_res_sample = controlnet_tile(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_cond=cond_tile, return_dict=False)
            
        noise_pred = unet(
            noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples, mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]
        
        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        loss.backward()
        optimizer.step()
        
        # Only step the optimizer every N steps
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Log to wandb
        wandb.log({"train_loss": loss.item(), "epoch": epoch + 1}, step=global_step)
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        global_step += 1

    # --- VALIDATION PHASE ---
    controlnet_tile.eval()
    if MODE == "multi_controlnet": controlnet_depth.eval()
    
    val_loss_total = 0.0
    val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
    
    with torch.no_grad():
        for batch in val_progress_bar:
            latents = vae.encode(batch["pixel_values"].to(DEVICE)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=DEVICE).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            text_inputs = tokenizer(batch["text"], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").to(DEVICE)
            encoder_hidden_states = text_encoder(text_inputs.input_ids)[0]
            
            cond_tile = batch["cond_tile"].to(DEVICE)
            
            if MODE == "multi_controlnet":
                cond_depth = batch["cond_depth"].to(DEVICE)
                down_tile, mid_tile = controlnet_tile(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_cond=cond_tile, return_dict=False)
                down_depth, mid_depth = controlnet_depth(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_cond=cond_depth, return_dict=False)
                down_block_res_samples = [t + d for t, d in zip(down_tile, down_depth)]
                mid_block_res_sample = mid_tile + mid_depth
            else:
                down_block_res_samples, mid_block_res_sample = controlnet_tile(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_cond=cond_tile, return_dict=False)
                
            noise_pred = unet(
                noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples, mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]
            
            val_loss = F.mse_loss(noise_pred, noise, reduction="mean")
            val_loss_total += val_loss.item()
            
    avg_val_loss = val_loss_total / len(val_dataloader)
    wandb.log({"val_loss": avg_val_loss, "epoch": epoch + 1}, step=global_step)
    print(f"[*] Epoch {epoch+1} completed. Average Val Loss: {avg_val_loss:.4f}")

    # Save tracking artifacts at specified intervals
    epoch_dir = f"specialized_controlnet_epoch_{epoch+1}"
    controlnet_tile.save_pretrained(os.path.join(FINAL_DIR, epoch_dir, "tile"))
    if MODE == "multi_controlnet":
        controlnet_depth.save_pretrained(os.path.join(FINAL_DIR, epoch_dir, "depth"))

controlnet_tile.save_pretrained(os.path.join(FINAL_DIR, "tile"))
if MODE == "multi_controlnet":
    controlnet_depth.save_pretrained(os.path.join(FINAL_DIR, "depth"))
print(f"[*] Final fine-tuned models saved to {FINAL_DIR}")
wandb.finish()
print("[+] Training task execution finalized.")