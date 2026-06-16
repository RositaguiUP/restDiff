import json
import random
import os
import argparse

def sample_frames(scene, floor, n):
    # Define paths
    base_dir = f"data/{scene}/{floor}"
    input_path = os.path.join(base_dir, "poses.json")
    output_dir = os.path.join(base_dir, "poses_to_render")
    output_path = os.path.join(output_dir, f"poses_{n}.json")

    # Load the source data
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Sample n frames
    frames = data.get('frames', [])
    if n > len(frames):
        print(f"Warning: Requesting {n} frames, but only {len(frames)} available. Using all.")
        sampled_frames = frames
    else:
        sampled_frames = random.sample(frames, n)

    # Create new dictionary
    new_data = {
        "camera_model": data.get("camera_model"),
        "w": data.get("w"),
        "h": data.get("h"),
        "frames": sampled_frames
    }

    # Save to output location
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=4)
    
    print(f"Successfully saved {len(sampled_frames)} frames to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample frames from poses.json")
    parser.add_argument("scene", help="The scene name")
    parser.add_argument("floor", help="The floor number")
    parser.add_argument("n", type=int, help="Number of random frames to sample")
    
    args = parser.parse_args()
    sample_frames(args.scene, args.floor, args.n)
    
# python rendering/select_from_poses.py <scene> <floor> <n>
# python rendering/select_from_poses.py 2F5Z7_007 0 6