import os
import json
import argparse
import numpy as np
from scipy.interpolate import interp1d


def simplify_path(frames, num_poses):
    """
    Generate N poses from entire trajectory:
    - interpolate only horizontal motion
    - fix vertical coordinate to average height
    """

    poses = np.array([f["pose"] for f in frames])

    translations = poses[:, :3, 3]
    rotations = poses[:, :3, :3]

    # Detect vertical axis automatically:
    ranges = translations.max(axis=0) - translations.min(axis=0)
    vertical_axis = np.argmin(ranges)

    horizontal_axes = [i for i in range(3) if i != vertical_axis]

    print(f"Detected vertical axis: {vertical_axis}")

    # Average height
    avg_height = translations[:, vertical_axis].mean()

    # Parameterize full path
    t_original = np.linspace(0, 1, len(translations))
    t_new = np.linspace(0, 1, num_poses)

    new_trans = np.zeros((num_poses, 3))

    # Interpolate only horizontal axes
    for ax in horizontal_axes:
        interp = interp1d(
            t_original,
            translations[:, ax],
            kind="linear"
        )
        new_trans[:, ax] = interp(t_new)

    # Fix vertical coordinate
    new_trans[:, vertical_axis] = avg_height

    new_frames = []

    for i in range(num_poses):

        idx = round(i * (len(rotations) - 1) / (num_poses - 1))

        pose = np.eye(4)

        # Keep nearest orientation
        pose[:3, :3] = rotations[idx]

        pose[:3, 3] = new_trans[i]

        frame = frames[idx].copy()
        frame["pose"] = pose.tolist()

        new_frames.append(frame)

    return new_frames


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--poses_json",
        required=True
    )

    parser.add_argument(
        "--num_poses",
        type=int,
        default=100
    )

    args = parser.parse_args()

    with open(args.poses_json, "r") as f:
        data = json.load(f)

    frames = data["frames"]

    if len(frames) < 2:
        raise ValueError("Need at least 2 poses")

    data["frames"] = simplify_path(
        frames,
        args.num_poses
    )

    output_dir = os.path.join(
        os.path.dirname(args.poses_json),
        "poses_to_render"
    )

    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(
        output_dir,
        f"trajectory_simple_{args.num_poses}.json"
    )

    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved simplified trajectory:")
    print(out_path)


if __name__ == "__main__":
    main()