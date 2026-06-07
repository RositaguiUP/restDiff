from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def save_turbo_png(image_path):
    image_path = Path(image_path)

    # Load image (works for .npy files)
    img = np.load(image_path)

    # Normalize
    img_vis = img / (img.max() + 1e-6)

    # Output path: same directory, same stem, .png extension
    output_path = image_path.with_suffix(".png")

    # Save figure
    plt.figure()
    plt.imshow(img_vis, cmap="turbo")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    image_path = input("Image path: ").strip()
    save_turbo_png(image_path)