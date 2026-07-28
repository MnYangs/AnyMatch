"""
SGVC PCK Evaluation Script
Computes the Percentage of Correct Keypoints (PCK) metric for novel view synthesis quality.
Uses RoMa feature matching and geometric verification with depth maps.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append("third_party/RoMa_AnyMatch/")
from third_party.RoMa_AnyMatch.romatch import roma_outdoor
from third_party.RoMa_AnyMatch.romatch.utils.utils import warp_kpts

# ---------------------------------------------------------------------------
# Paths Configuration (modify these for your setup)
# ---------------------------------------------------------------------------
ROMA_WEIGHTS_PATH = "/data/Zizhuo_li/MnYang/Roma/RoMa/checkpoints/roma_outdoor.pth"
RESULTS_BASE_DIR = "/data/Zizhuo_li/MnYang/AnyMatch/View_Transformation/AnyMatch_results"
PCK_RESULTS_PATH = os.path.join(RESULTS_BASE_DIR, "index", "images_names_pcks.npz")

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:3")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def read_txt(file_path: str) -> torch.Tensor:
    """Read a matrix from a text file (comma-separated values).

    Args:
        file_path: Path to the text file.

    Returns:
        A torch.Tensor containing the parsed matrix.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    matrix = [list(map(float, line.strip().split(","))) for line in lines]
    return torch.tensor(matrix)


def load_depth(depth_path: str) -> torch.Tensor:
    """Load a depth map from a .npy file.

    Args:
        depth_path: Path to the .npy depth file.

    Returns:
        Depth map as a torch.Tensor.
    """
    return torch.from_numpy(np.load(depth_path))


def compute_geometric_distance(
    depth1: torch.Tensor,
    depth2: torch.Tensor,
    T_1to2: torch.Tensor,
    K1: torch.Tensor,
    K2: torch.Tensor,
    dense_matches: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute geometric distance and PCK metrics between two views.

    Warps keypoints from view 1 to view 2 using depth and camera parameters,
    then compares against the predicted matches from the feature matcher.

    Args:
        depth1: Depth map of view 1 (H, W).
        depth2: Depth map of view 2 (H, W).
        T_1to2: 4x4 transformation matrix from view 1 to view 2.
        K1: 3x3 camera intrinsic matrix of view 1.
        K2: 3x3 camera intrinsic matrix of view 2.
        dense_matches: Dense match tensor (1, H, W, 4) with (x1, y1, x2, y2).

    Returns:
        Tuple containing:
            - gd: Geometric distances for valid matches.
            - pck_1: PCK @ 1 pixel threshold.
            - pck_3: PCK @ 3 pixel threshold.
            - pck_5: PCK @ 5 pixel threshold.
            - prob: Validity mask (1, H, W).
    """
    batch_size, height, width, _ = dense_matches.shape

    with torch.no_grad():
        # Extract keypoints from view 1
        x1 = dense_matches[..., :2].reshape(batch_size, height * width, 2)

        # Warp keypoints to view 2 using depth and camera parameters
        mask, x2 = warp_kpts(
            x1.double(),
            depth1.double(),
            depth2.double(),
            T_1to2.double(),
            K1.double(),
            K2.double(),
        )

        # Convert from normalized to pixel coordinates
        x2 = torch.stack(
            (width * (x2[..., 0] + 1) / 2, height * (x2[..., 1] + 1) / 2), dim=-1
        )
        prob = mask.float().reshape(batch_size, height, width)

    # Extract predicted matches from view 2 and convert to pixel coords
    x2_hat = dense_matches[..., 2:]
    x2_hat = torch.stack(
        (width * (x2_hat[..., 0] + 1) / 2, height * (x2_hat[..., 1] + 1) / 2), dim=-1
    )

    # Compute geometric distances only for valid matches
    gd = (x2_hat - x2.reshape(batch_size, height, width, 2)).norm(dim=-1)
    gd = gd[prob == 1]

    # Compute PCK at different thresholds
    pck_1 = (gd < 1.0).float().mean()
    pck_3 = (gd < 3.0).float().mean()
    pck_5 = (gd < 5.0).float().mean()

    return gd, pck_1, pck_3, pck_5, prob


def get_unprocessed_images(
    image_dir: str, results_path: str
) -> Tuple[List[str], List]:
    """Get list of images that haven't been processed yet.

    Args:
        image_dir: Directory containing the input images.
        results_path: Path to the existing results .npz file (if any).

    Returns:
        Tuple containing:
            - List of unprocessed image names (without extension).
            - Existing results list to append to.
    """
    # Get all JPEG images in the directory
    jpg_file_names = [
        f[:-4] for f in os.listdir(image_dir) if f.lower().endswith(".jpg")
    ]

    # Load existing results if available
    if os.path.exists(results_path):
        data = np.load(results_path)
        existing_results = data["name_pcks"].tolist()
    else:
        existing_results = []

    # Determine which images haven't been processed yet
    processed_names = [sublist[0] for sublist in existing_results]
    unprocessed = list(set(jpg_file_names) - set(processed_names))

    return unprocessed, existing_results


def process_single_image(
    image_name: str,
    roma_model,
    base_dir: str,
    device: torch.device,
) -> Optional[list]:
    """Process a single image pair and compute PCK metrics.

    Args:
        image_name: Base name of the image (without extension).
        roma_model: Loaded RoMa matching model.
        base_dir: Base directory containing image1/, image2/, depth1/, depth2/, etc.
        device: Computation device.

    Returns:
        A list [image_name, pck_1, pck_3, pck_5] or None if processing failed.
    """
    try:
        # Build file paths
        depth1_path = os.path.join(base_dir, "depth1", f"{image_name}.npy")
        depth2_path = os.path.join(base_dir, "depth2", f"{image_name}.npy")
        cam_int_path = os.path.join(base_dir, "cam_int", f"{image_name}.txt")
        cam_ext_path = os.path.join(base_dir, "cam_ext", f"{image_name}.txt")
        im1_path = os.path.join(base_dir, "image1", f"{image_name}.jpg")
        im2_path = os.path.join(base_dir, "image2", f"{image_name}.jpg")

        # Verify all required files exist
        for path in [depth1_path, depth2_path, cam_int_path, cam_ext_path, im1_path, im2_path]:
            if not os.path.exists(path):
                print(f"Warning: Missing file {path}, skipping {image_name}")
                return None

        # Load camera parameters
        K1 = read_txt(cam_int_path).to(device)
        K2 = K1  # Same camera, different pose
        T_1to2 = read_txt(cam_ext_path).to(device)

        # Load depth maps
        depth1 = load_depth(depth1_path).to(device)
        depth2 = load_depth(depth2_path).to(device)

        # Perform feature matching
        warp, certainty = roma_model.match(im1_path, im2_path, device=device)

        # Compute geometric distances and PCK metrics
        _, pck_1, pck_3, pck_5, _ = compute_geometric_distance(
            depth1.unsqueeze(0).double(),
            depth2.unsqueeze(0).double(),
            T_1to2.unsqueeze(0).double(),
            K1.unsqueeze(0).double(),
            K2.unsqueeze(0).double(),
            warp.unsqueeze(0).double(),
        )

        return [
            image_name,
            pck_1.cpu().numpy(),
            pck_3.cpu().numpy(),
            pck_5.cpu().numpy(),
        ]

    except Exception as e:
        print(f"Error processing {image_name}: {e}")
        return None


def main() -> None:
    """Main execution function."""
    print(f"Using device: {DEVICE}")
    print("Loading RoMa model...")

    # Load model weights
    roma_weights = torch.load(
        ROMA_WEIGHTS_PATH, map_location=DEVICE, weights_only=True
    )

    # Initialize RoMa model
    roma_model = roma_outdoor(
        device=DEVICE, weights=roma_weights
    )


    # Determine which images need processing
    image2_dir = os.path.join(RESULTS_BASE_DIR, "image2")
    unprocessed_images, existing_results = get_unprocessed_images(
        image2_dir, PCK_RESULTS_PATH
    )

    if not unprocessed_images:
        print("All images have already been processed. Nothing to do.")
        return

    print(f"Found {len(unprocessed_images)} unprocessed images out of "
          f"{len(os.listdir(image2_dir))} total.")
    print("Processing...")

    # Process each unprocessed image with a progress bar
    for image_name in tqdm(unprocessed_images, desc="Computing PCK"):
        result = process_single_image(image_name, roma_model, RESULTS_BASE_DIR, DEVICE)

        if result is not None:
            existing_results.append(result)

        # Save intermediate results after each image (in case of interruption)
        np.savez(PCK_RESULTS_PATH, name_pcks=existing_results)

    # Compute and display aggregate statistics
    if existing_results:
        pck_1_values = np.array([r[1] for r in existing_results])
        pck_3_values = np.array([r[2] for r in existing_results])
        pck_5_values = np.array([r[3] for r in existing_results])

        print(f"\n{'=' * 50}")
        print(f"Processing complete. Total images evaluated: {len(existing_results)}")
        print(f"{'=' * 50}")
        print(f"Mean PCK @ 1px: {pck_1_values.mean():.4f} ± {pck_1_values.std():.4f}")
        print(f"Mean PCK @ 3px: {pck_3_values.mean():.4f} ± {pck_3_values.std():.4f}")
        print(f"Mean PCK @ 5px: {pck_5_values.mean():.4f} ± {pck_5_values.std():.4f}")
        print(f"{'=' * 50}")
    else:
        print("No results were produced.")


if __name__ == "__main__":
    main()