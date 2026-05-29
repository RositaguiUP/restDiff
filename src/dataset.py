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

        with open(os.path.join(data_dir, "transforms.json"), "r") as f:
            self.meta = json.load(f)
            
        all_frames = self.meta["frames"]
        
        # Split the data
        if split == "train":
            self.frames = [f for i, f in enumerate(all_frames) if i % test_every != 0]
        elif split == "test":
            self.frames = [f for i, f in enumerate(all_frames) if i % test_every == 0]
        else:
            self.frames = all_frames
        
        self.W = int(self.meta["w"])
        self.H = int(self.meta["h"])
        
        # Build Intrinsic Matrix (K)
        self.K = torch.tensor([
            [self.meta["fl_x"], 0, self.meta["cx"]],
            [0, self.meta["fl_y"], self.meta["cy"]],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)
        
        # Matrix to convert OpenGL (Right-Up-Back) to OpenCV (Right-Down-Forward)
        self.gl_to_cv = torch.tensor([
            [1,  0,  0,  0],
            [0, -1,  0,  0],
            [0,  0, -1,  0],
            [0,  0,  0,  1]
        ], dtype=torch.float32, device=self.device)

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
        
        # Load and Convert Camera Matrix (C2W) (OpenGL -> OpenCV)
        c2w_gl = torch.tensor(frame["transform_matrix"], dtype=torch.float32, device=self.device)
        c2w_cv = c2w_gl @ self.gl_to_cv
        
        return {
            "image": img_tensor.to(self.device),
            "depth": depth_tensor.to(self.device),
            "camtoworld": c2w_cv,
            "K": self.K
        }