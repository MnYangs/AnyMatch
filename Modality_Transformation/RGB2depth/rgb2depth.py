import torch
from View_Transformation.MoGe.MoGe.model.v1 import MoGeModel
import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib

def colorize_depth(depth: np.ndarray, mask: np.ndarray = None, normalize: bool = True, cmap: str = 'Spectral') -> np.ndarray:
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        depth = np.where((depth > 0) & mask, depth, np.nan)
    disp = 1 / depth
    if normalize:
        min_disp, max_disp = np.nanquantile(disp, 0.001), np.nanquantile(disp, 0.99)
        disp = (disp - min_disp) / (max_disp - min_disp)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored

if __name__ == "__main__":
    # Get project root directory (script is at Modality_Transformation/RGB2depth/rgb2depth.py, up two levels)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    depth_device = (
        "cuda:3"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Input and output paths (relative to project root)
    input_dir = os.path.join(project_root, "View_Transformation", "AnyMatch_results", "image1")
    output_dir = os.path.join(project_root, "View_Transformation", "AnyMatch_results", "image_depth")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load MoGe depth estimation model
    MoGe_depth = MoGeModel.from_pretrained(
        os.path.join(project_root, "View_Transformation", "MoGe", "checkpoint", "model.pt")
    ).to(depth_device).eval()

    # Supported image formats
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # Get all image files
    image_paths = [
        str(p) for p in Path(input_dir).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_paths.sort()

    print(f"Found {len(image_paths)} images in {input_dir}")

    for image_path in tqdm(image_paths, desc="Converting to depth"):
        try:
            # Read image
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            if image is None:
                print(f"Warning: Cannot read {image_path}, skipping")
                continue

            # Convert to tensor and infer depth
            image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=depth_device).permute(2, 0, 1)
            output = MoGe_depth.infer(image_tensor)
            depth_image = output["depth"].cpu().numpy().astype(np.float32)


            image_name = Path(image_path).stem
            depth_colored = cv2.cvtColor(colorize_depth(depth_image), cv2.COLOR_RGB2BGR)
            vis_save_path = os.path.join(output_dir, f"{image_name}.png")
            cv2.imwrite(vis_save_path, depth_colored)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    print(f"Depth images saved to {output_dir}")