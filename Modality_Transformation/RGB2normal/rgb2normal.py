import cv2
import torch
import os
from View_Transformation.MoGe.MoGe.model.v2 import MoGeModel as MoGeModel2
import numpy as np
from tqdm import tqdm
from pathlib import Path

def colorize_normal(normal: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    if mask is not None:
        normal = np.where(mask[..., None], normal, 0)
    normal = normal * [0.5, -0.5, -0.5] + 0.5
    normal = (normal.clip(0, 1) * 255).astype(np.uint8)
    return normal

if __name__ == "__main__":
    # Get project root directory (script is at Modality_Transformation/RGB2normal/rgb2normal.py, up two levels)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    # Input and output paths (relative to project root)
    input_dir = os.path.join(project_root, "View_Transformation", "AnyMatch_results", "image1")
    output_dir = os.path.join(project_root, "View_Transformation", "AnyMatch_results", "image_normal")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    normal_device = (
        "cuda:3"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    MoGe_normal = MoGeModel2.from_pretrained(os.path.join(project_root, "View_Transformation", "MoGe", "checkpoint", "modelv2.pt")).to(normal_device)

    # Supported image formats
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # Get all image files
    image_paths = [
        str(p) for p in Path(input_dir).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_paths.sort()

    print(f"Found {len(image_paths)} images in {input_dir}")

    for image_path in tqdm(image_paths, desc="Converting to normal"):
        try:
            # Read image
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            if image is None:
                print(f"Warning: Cannot read {image_path}, skipping")
                continue

            # Convert to tensor and infer normal map
            image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=normal_device).permute(2, 0, 1)
            output = MoGe_normal.infer(image_tensor)
            normal_image = output["normal"].cpu().numpy().astype(np.float32)

            image_name = Path(image_path).stem
            normal_colored = cv2.cvtColor(colorize_normal(normal_image), cv2.COLOR_RGB2BGR)
            vis_save_path = os.path.join(output_dir, f"{image_name}.png")
            cv2.imwrite(vis_save_path, normal_colored)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    print(f"Normal images saved to {output_dir}")
