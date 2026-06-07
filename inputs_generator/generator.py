import json
import numpy as np
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from rich import print

# Custom project imports
from inputs_generator.src.v3dc_io import read_v3dc_sliced
from inputs_generator.src.environment import Environment
# from inputs_generator.src.deblur_utils import MotionDeblurer

class Generator:
    def __init__(self, env_id, floor_number, output_path):
        self.env_id = env_id
        self.floor_number = floor_number
        self.output_path = Path(output_path)
        self.env = Environment(env_id)
        
        # Setup directories
        self.rgb_dir = self.output_path / "rgb"
        self.depth_dir = self.output_path / "depth"
        self.pcd_dir = self.output_path / "pointcloud.ply"

        for d in [self.rgb_dir, self.depth_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _is_blurry(self, image, threshold=15.0):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return score < threshold, score

    def _get_pose_distance(self, w2c_1, w2c_2):
        c1 = -w2c_1[:3, :3].T @ w2c_1[:3, 3]
        c2 = -w2c_2[:3, :3].T @ w2c_2[:3, 3]
        dist = np.linalg.norm(c1 - c2)
        R_diff = w2c_1[:3, :3] @ w2c_2[:3, :3].T
        angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
        return dist, np.degrees(angle)

    def _preprocess_data(self, rgb, depth, img_size=512):
        # Resize depth to match RGB, center crop to square, then resize to final
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        h, w = rgb.shape[:2]
        size = min(h, w)
        start_x, start_y = (w - size) // 2, (h - size) // 2
        # Default: crop to square and resize
        rgb_cropped = cv2.resize(rgb[start_y:start_y+size, start_x:start_x+size], (img_size, img_size))
        depth_cropped = cv2.resize(depth[start_y:start_y+size, start_x:start_x+size], (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        return rgb_cropped, depth_cropped
    
    def _save_depth_visualization(self, idx, depth, depth_path):
        depth_vis = depth / (depth.max() + 1e-6)

        plt.imshow(depth_vis, cmap="turbo")
        plt.colorbar()
        plt.savefig(depth_path / f"{idx:05d}.png")
        plt.close()
    
    def _copy_pointcloud(self):
        """Copies the original point cloud to the output directory."""
        path_org_pcd = self.env.path_pcd(self.floor_number)
        
        if path_org_pcd.exists():
            print(f"Copying point cloud to {self.pcd_dir}...")
            shutil.copy2(path_org_pcd, self.pcd_dir) # copy2 preserves metadata[cite: 1]
        else:
            print(f"Warning: Original point cloud not found at {path_org_pcd}")

    def run(self, crop=True, img_size=512, dist_thresh=0.15, rot_thresh=10.0, blur_thresh=15.0, step=1):        
        # --- STEP 1: FILTERING & SCAN EXTRACTION ---
        print("[bold blue]Starting Step 1: Filtering & Extraction...[/bold blue]")
        # deblurrer = MotionDeblurer()
        
        scan_to_mesh = np.loadtxt(self.env.path_sub_pano(self.floor_number)).reshape(4, 4)
        mesh_to_scan = np.linalg.inv(scan_to_mesh)

        views = read_v3dc_sliced(fp=self.env.path_v3dc(self.floor_number), step=step, read_rgb=True, read_depth=True, orient_img=False)
        
        selected_frames = []
        pending = None

        for i, view in enumerate(tqdm(views, desc="Filtering Frames")):
            if view["img"] is None: continue
            blurry, score = self._is_blurry(view["img"], blur_thresh)
            if blurry: continue

            w2c_mesh = view["viewmat"] @ mesh_to_scan
            current = {"view": view, "score": score, "w2c": w2c_mesh, "id": i}

            if pending is None:
                pending = current
                continue

            d, a = self._get_pose_distance(pending["w2c"], w2c_mesh)
            if d < dist_thresh and a < rot_thresh:
                if score > pending["score"]: pending = current
            else:
                selected_frames.append(pending)
                pending = current
        if pending: selected_frames.append(pending)

        # --- STEP 2: Processing ---
        print("[bold blue]Starting Step 2: Processing...[/bold blue]")
        
        # Prepare for final JSON -- will be filled from first processed frame
        json_data = {
            "camera_model": "OPENCV",
            "w": None, "h": None,
            "frames": []
        }
        first_w = None
        first_h = None
        
        for frame in tqdm(selected_frames, desc="Processing Frames"):
            idx = frame["id"]
            
            # Process Scan RGB/Depth
            # rgb_clean = deblurrer.process(frame["view"]["img"])
            rgb_clean = frame["view"]["img"]
            depth_m = frame["view"]["depth"].astype(np.float32) / 1000.0

            # If cropping/resizing is enabled, use _preprocess_data. Otherwise keep original sizes (only ensure depth matches rgb)
            if crop:
                rgb_f, scan_depth_f = self._preprocess_data(rgb_clean, depth_m, img_size)
                out_h, out_w = img_size, img_size
            else:
                # Resize depth to match rgb resolution, but keep rgb as-is
                scan_depth_f = cv2.resize(depth_m, (rgb_clean.shape[1], rgb_clean.shape[0]), interpolation=cv2.INTER_NEAREST)
                rgb_f = rgb_clean
                out_h, out_w = rgb_f.shape[:2]
            
            # Save Scan Data
            cv2.imwrite(str(self.rgb_dir / f"{idx:05d}.png"), cv2.cvtColor(rgb_f, cv2.COLOR_RGB2BGR))
            np.save(self.depth_dir / f"{idx:05d}.npy", scan_depth_f)
            # self._save_depth_visualization(idx, scan_depth_f, self.depth_dir)

            # Update JSON
            # Convert W2C Scan to C2W (it is in OpenCV format)
            w2c_scan = frame["view"]["viewmat"]
            c2w_cv = np.linalg.inv(w2c_scan)
            # Set intrinsics from first processed frame
            K_frame = frame["view"].get("K")
            fx_frame = float(frame["view"].get("fx"))
            fy_frame = float(frame["view"].get("fy"))
            cx_frame = float(frame["view"].get("cx"))
            cy_frame = float(frame["view"].get("cy"))
            w_frame = int(frame["view"].get("width") or out_w)
            h_frame = int(frame["view"].get("height") or out_h)

            if first_w is None:
                first_w = w_frame
                first_h = h_frame
                # populate top-level json_data image size from first frame
                json_data["w"] = out_w
                json_data["h"] = out_h
            else:
                if w_frame != first_w or h_frame != first_h:
                    print(f"!WARNING! Frame size for frame {idx} ({w_frame},{h_frame}) differs from first frame ({first_w},{first_h}). Using first frame values in metadata.")

            # Build per-frame intrinsics to store with this frame
            if K_frame is not None:
                K_list = np.array(K_frame).tolist()
            else:
                K_list = [[fx_frame, 0.0, cx_frame], [0.0, fy_frame, cy_frame], [0.0, 0.0, 1.0]]

            frame_entry = {
                "id": idx,
                "blurry_score": frame["score"],
                "orientation": frame["view"].get("orientation", "landscape"),
                "file_path": f"rgb/{idx:05d}.png",
                "depth_file_path": f"depth/{idx:05d}.npy",
                "pose": c2w_cv.tolist(),
                "K": K_list,
                "fl_x": fx_frame,
                "fl_y": fy_frame,
                "cx": cx_frame,
                "cy": cy_frame,
                "w": out_w,
                "h": out_h,
            }

            json_data["frames"].append(frame_entry)
            
        with open(self.output_path / "poses.json", "w") as f:
            json.dump(json_data, f, indent=4)
        
        self._copy_pointcloud()
        
        print(f"[bold green]Done! Processed {len(json_data['frames'])} frames.\n Dataset exported to {self.output_path}[/bold green]")