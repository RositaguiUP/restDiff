import os
import json
import argparse
from PIL import Image
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(
        description="Convert info.json into a render_custom_poses-compatible file for multiple scenes and floors."
    )
    # Changed to accept a list of scenes
    parser.add_argument("--scenes", nargs='+', required=True, help="List of scene names")
    parser.add_argument("--outdir", type=str, default="data/dataset", help="Base output directory")
    parser.add_argument("--width", type=int, default=1920, help="Image width")
    parser.add_argument("--height", type=int, default=1080, help="Image height")
    args = parser.parse_args()

    for scene in args.scenes:
        # Construct path based on requirements
        info_json_path = f"/home/rosita/tests/data/{scene}/photo_alignment/images/info.json"
        
        if not os.path.exists(info_json_path):
            print(f"Warning: {info_json_path} not found. Skipping.")
            continue
        
        with open(info_json_path, "r") as f:
            data = json.load(f)
            poses = data.get("results", [])
        
        # Group frames by floor_number
        floors_dict = defaultdict(list)

        for pose_data in poses:
            if pose_data.get("status") != "SUCCESS":
                continue

            floor = str(pose_data.get("floor_number", "default"))
            
            file_path = f"/home/rosita/tests/data/{scene}/photo_alignment/images/images_resized/{pose_data['id']}.jpg"
            
            
            with Image.open(file_path) as img:
                W, H = img.size
            
            focal = pose_data["focal"]

            # Convert normalized focal → pixels
            fl_x = focal * W / 2
            fl_y = focal * H / 2

            # principal point normalized → pixel coords
            pp = pose_data.get("principle_point", [0.0, 0.0])

            cx = W / 2 + pp[0] * W / 2
            cy = H / 2 + pp[1] * H / 2
            
            
            # Append required data
            frame_entry = {
                "id": pose_data["id"],
                "file_path": file_path,
                "floor_number": floor,
                "pose": pose_data["camera_pose"],
                "fl_x": fl_x,
                "fl_y": fl_y,
                "cx": cx,
                "cy": cy,
                "w": W,
                "h": H,
                "K": [
                    [fl_x, 0, cx],
                    [0, fl_y, cy],
                    [0, 0, 1]
                ]
            }
            
            floors_dict[floor].append(frame_entry)

        # Save files per floor
        for floor, frames in floors_dict.items():
            output = {
                "w": args.width,
                "h": args.height,
                "frames": frames,
            }

            # Construct output directory: data/dataset/{scene}/{floor}/
            output_dir = os.path.join(args.outdir, scene, floor)
            os.makedirs(output_dir, exist_ok=True)

            out_path = os.path.join(output_dir, "poses_hd.json")

            with open(out_path, "w") as f:
                json.dump(output, f, indent=4)

            print(f"Scene: {scene}, Floor: {floor} - Saved {len(frames)} poses to {out_path}")

if __name__ == "__main__":
    main()
    
# python rendering/generate_hd_poses.py --scenes 2B2H9_993 2DNBK_922 2F4TK_157 2GFG5_494 2CD65_703 2DPPJ_505 2F6D4_988 2G5TT_538 2J6BP_139
# python rendering/generate_hd_poses.py --scenes 2F5Z7_007 --outdir data