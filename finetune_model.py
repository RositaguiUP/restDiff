import gc
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

from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# ==========================================
# CONFIGURATION 
# ==========================================
DEBUG_MODE = False  # <--- SET TO FALSE WHEN YOU ARE READY FOR THE REAL RUN

MODE = "tile_only" # Options: "tile_only" or "multi_controlnet"
# BASE_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE" 
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
BASE_DIR = "./dataset/warmup/v6.0/29999/"
JSON_FILE = "finetune_meta_all_b40.json"
FINAL_DIR = "./finetuned_models/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4 
LEARNING_RATE = 1e-5 
EPOCHS = 15 if not DEBUG_MODE else 2 # Run only 2 epochs in debug mode
VAL_SPLIT_RATIO = 0.10 
PROMPT_DROPOUT_RATE = 0.20 
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


# Initialize these globally at the top of your script
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(DEVICE)

def run_visual_validation(epoch, global_step, val_dataloader, vae, text_encoder, tokenizer, unet, controlnet_tile, noise_scheduler, device):
    print(f"\n[+] Executing Epoch {epoch} Evaluation Grid & Metrics...")
    
    # Initialize metrics 
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)

    # 1. Grab a fixed batch of images from the Dataloader
    val_iterator = iter(val_dataloader)
    batch = next(val_iterator) 
    
    # 2. Generate the enhanced images
    val_pipe = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        controlnet=controlnet_tile, 
        scheduler=UniPCMultistepScheduler.from_config(noise_scheduler.config),
        safety_checker=None, feature_extractor=None
    ).to(device)
    val_pipe.set_progress_bar_config(disable=True)

    with torch.no_grad(), torch.autocast("cuda"):
        # Convert batch tensors to PIL Images for the pipeline
        cond_imgs = [transforms.ToPILImage()(img) for img in batch["cond_tile"]]
        
        pred_imgs = val_pipe(
            prompt=batch["text"],
            image=cond_imgs,
            num_inference_steps=20,
            guidance_scale=7.0
        ).images

    # 3. Calculate Advanced Metrics
    pred_tensors = torch.stack([transforms.ToTensor()(img) for img in pred_imgs]).to(device)
    gt_tensors_unnorm = batch["pixel_values"].to(device) * 0.5 + 0.5 

    psnr_val = psnr_metric(pred_tensors, gt_tensors_unnorm)
    ssim_val = ssim_metric(pred_tensors.unsqueeze(1) if pred_tensors.ndim == 3 else pred_tensors, 
                           gt_tensors_unnorm.unsqueeze(1) if gt_tensors_unnorm.ndim == 3 else gt_tensors_unnorm)
    
    # LPIPS expects inputs in range [-1, 1]
    lpips_val = lpips_metric(pred_tensors * 2.0 - 1.0, gt_tensors_unnorm * 2.0 - 1.0)

    # 4. Create the Visual Grid for WandB
    log_images = []
    for i in range(len(pred_imgs)):
        gt_img = transforms.ToPILImage()(gt_tensors_unnorm[i].cpu())
        log_images.append(wandb.Image(cond_imgs[i], caption=f"Sample {i}: 3DGS Render"))
        log_images.append(wandb.Image(pred_imgs[i], caption=f"Sample {i}: Enhanced"))
        log_images.append(wandb.Image(gt_img, caption=f"Sample {i}: GT Scan"))

    # Log everything
    wandb.log({
        "Eval/PSNR (Higher=Better)": psnr_val.item(),
        "Eval/SSIM (Structure Preserved)": ssim_val.item(),
        "Eval/LPIPS (Photorealism, Lower=Better)": lpips_val.item(),
        "Validation Grid": log_images,
        "epoch": epoch
    }, step=global_step)
    
    del val_pipe
    torch.cuda.empty_cache()
    gc.collect()

# ==========================================
# 1. CUSTOM DATASET CLASS
# ==========================================
class ScanCompletionDataset(Dataset):
    def __init__(self, data_list, target_size=(512, 512), is_train=False):
        self.items = data_list
        self.target_size = target_size
        self.is_train = is_train
        self.base_prompt = "photorealistic, sharp, highly detailed indoor architecture, clear edges, 4k texture"
        
    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        gt_rgb = Image.open(item["gt_rgb_path"]).convert("RGB")
        render_rgb = Image.open(item["render_rgb_path"]).convert("RGB")
        
        # --- DATA AUGMENTATION ---
        # 50% chance to flip images left-to-right. 
        # MUST happen to both identically so ControlNet geometry matches!
        if self.is_train and random.random() > 0.5:
            gt_rgb = gt_rgb.transpose(Image.FLIP_LEFT_RIGHT)
            render_rgb = render_rgb.transpose(Image.FLIP_LEFT_RIGHT)
            
        gt_tensor = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])(gt_rgb)
        
        cond_tensor = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ])(render_rgb)
        
        # --- PROMPT DROPOUT (EXPERT 2) ---
        prompt = self.base_prompt
        if self.is_train and random.random() < PROMPT_DROPOUT_RATE:
            prompt = ""
            
        batch = {"pixel_values": gt_tensor, "cond_tile": cond_tensor, "text": prompt}
        return batch

# ==========================================
# 2. SETUP & WANDB
# ==========================================
with open(BASE_DIR + JSON_FILE, "r") as f: 
    full_data = json.load(f)

# --- DEBUG MODE INJECTION ---
if DEBUG_MODE:
    print("\n[!!!] DEBUG MODE IS ACTIVE [!!!]")
    print("[!] Slicing dataset to only 20 images for a rapid test run.")
    random.shuffle(full_data)
    full_data = full_data[:20]

split_idx = int(len(full_data) * (1 - VAL_SPLIT_RATIO))
train_data = full_data[:split_idx]
val_data = full_data[split_idx:]

train_dataset = ScanCompletionDataset(train_data, is_train=True)
val_dataset = ScanCompletionDataset(val_data, is_train=False)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

run_name = f"run_{MODE}_b{BATCH_SIZE}_DEBUG" if DEBUG_MODE else f"run_{MODE}_b{BATCH_SIZE}"
wandb.init(project="thesis-gs-diffusion", name=run_name, dir=BASE_DIR)

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
optimizer = torch.optim.AdamW(controlnet_tile.parameters(), lr=LEARNING_RATE)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
global_step = 0
for epoch in range(EPOCHS):
    controlnet_tile.train()
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        latents = vae.encode(batch["pixel_values"].to(DEVICE)).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=DEVICE).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        text_inputs = tokenizer(batch["text"], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").to(DEVICE)
        encoder_hidden_states = text_encoder(text_inputs.input_ids)[0]
        
        cond_tile = batch["cond_tile"].to(DEVICE)
        down_block_res, mid_block_res = controlnet_tile(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_cond=cond_tile, return_dict=False)
            
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                          down_block_additional_residuals=down_block_res, mid_block_additional_residual=mid_block_res, return_dict=False)[0]
        
        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        loss = loss / GRADIENT_ACCUMULATION_STEPS 
        loss.backward()
        
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        wandb.log({"train_loss": loss.item() * GRADIENT_ACCUMULATION_STEPS, "epoch": epoch + 1}, step=global_step)
        progress_bar.set_postfix({"loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}"})
        global_step += 1

    # Validation & Model Checkpointing
    run_visual_validation(epoch + 1, global_step, val_dataloader, vae, text_encoder, tokenizer, unet, controlnet_tile, noise_scheduler, DEVICE)
    
    epoch_dir = f"controlnet_epoch_{epoch+1}_DEBUG" if DEBUG_MODE else f"controlnet_epoch_{epoch+1}"
    controlnet_tile.save_pretrained(os.path.join(FINAL_DIR, epoch_dir, "tile"))

final_name = "final_tile_DEBUG" if DEBUG_MODE else "final_tile"
controlnet_tile.save_pretrained(os.path.join(FINAL_DIR, final_name))
wandb.finish()
print("\n[+] Pipeline execution completed successfully!")