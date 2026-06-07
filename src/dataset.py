import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CustomGSDataset(Dataset):
    def __init__(self, data_dir: str, device: str = "cuda", split: str = "train", test_every: int = 8):
        self.data_dir = data_dir
        self.device = device
        self.split = split

        with open(os.path.join(data_dir, "poses.json"), "r") as f:
            self.meta = json.load(f)
            
        all_frames = self.meta["frames"]
        
        # Split the data
        if split == "train":
            self.frames = [f for i, f in enumerate(all_frames) if i % test_every != 0]
        elif split == "test":
            self.frames = [f for i, f in enumerate(all_frames) if i % test_every == 0]
        else:
            self.frames = all_frames
        
        # Try top-level image size, otherwise fallback to first frame
        if "w" in self.meta and self.meta["w"] is not None:
            self.W = int(self.meta["w"])
            self.H = int(self.meta["h"])
        else:
            # Fallback to first frame sizes
            first = all_frames[0]
            self.W = int(first.get("w", 0))
            self.H = int(first.get("h", 0))

        # Keep a dataset-level K fallback (from top-level meta) but primary usage will be per-frame K
        self.K = None
        if "K" in self.meta and self.meta["K"] is not None:
            try:
                K_np = np.array(self.meta["K"], dtype=np.float32)
                if K_np.shape == (3, 3):
                    self.K = torch.tensor(K_np, dtype=torch.float32, device=self.device)
            except Exception:
                self.K = None

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: int):
        frame = self.frames[idx]
        
        # Load RGB
        img_path = os.path.join(self.data_dir, frame["file_path"])
        img = Image.open(img_path).convert("RGB")
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        
        # Load Raw Metric Depth
        depth_path = os.path.join(self.data_dir, frame["depth_file_path"])
        depth_array = np.load(depth_path)
        depth_tensor = torch.from_numpy(depth_array).float()
        
        # Load and Convert Camera Matrix (C2W) (Already in OpenCV format)
        c2w_cv = torch.tensor(frame["pose"], dtype=torch.float32, device=self.device)

        # Build per-frame intrinsic matrix (prefer frame-level `K`)
        K_tensor = None
        if "K" in frame and frame["K"] is not None:
            K_np = np.array(frame["K"], dtype=np.float32)
            if K_np.shape == (3, 3):
                K_tensor = torch.tensor(K_np, dtype=torch.float32, device=self.device)
        else:
            # fall back to per-frame scalar intrinsics
            fl_x = frame.get("fl_x") or self.meta.get("fl_x")
            fl_y = frame.get("fl_y") or self.meta.get("fl_y")
            cx = frame.get("cx") or self.meta.get("cx")
            cy = frame.get("cy") or self.meta.get("cy")
            if fl_x is not None and fl_y is not None and cx is not None and cy is not None:
                K_tensor = torch.tensor([
                    [float(fl_x), 0.0, float(cx)],
                    [0.0, float(fl_y), float(cy)],
                    [0.0, 0.0, 1.0]
                ], dtype=torch.float32, device=self.device)

        if K_tensor is None and self.K is not None:
            K_tensor = self.K

        return {
            "image": img_tensor.to(self.device),
            "depth": depth_tensor.to(self.device),
            "camtoworld": c2w_cv,
            "K": K_tensor
        }