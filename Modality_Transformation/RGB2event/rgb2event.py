import os
import random
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def random_shift(shift_min, shift_max):
    range_choice = random.choice([(shift_min, shift_max), (-shift_max, -shift_min)])
    return random.randint(*range_choice)


def event_transfer_single(image_path, contrast_min=0.05, contrast_max=0.5, shift_min=2, shift_max=4):  #contrast_min=0.05, contrast_max=0.5, shift_min=1, shift_max=3
    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    gray_img_log = np.log1p(gray_img)

    contrast_threshold = random.uniform(contrast_min, contrast_max)

    xshift = random_shift(shift_min, shift_max)
    yshift = random_shift(shift_min, shift_max)

    pic_shape = [gray_img.shape[0], gray_img.shape[1], 3]
    img = np.full(pic_shape, [255, 255, 255], dtype=np.uint8)

    for i in range(abs(yshift), gray_img.shape[0] - abs(yshift)):
        for j in range(abs(xshift), gray_img.shape[1] - abs(xshift)):
            delta_L = gray_img_log[i + yshift, j + xshift] - gray_img_log[i, j]
            if delta_L > contrast_threshold:
                img[i, j] = [255, 0, 0]
            elif delta_L < -contrast_threshold:
                img[i, j] = [0, 0, 255]

    return img

if __name__ == "__main__":
    # Get project root directory (script is at Modality_Transformation/RGB2event/rgb2event.py, up two levels)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    # Input and output paths (relative to project root)
    input_dir = os.path.join(project_root, "View_Transformation", "AnyMatch_results", "image1")
    output_dir = os.path.join(project_root, "View_Transformation", "AnyMatch_results", "image_event")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Supported image formats
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # Get all image files
    image_paths = [
        str(p) for p in Path(input_dir).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_paths.sort()

    print(f"Found {len(image_paths)} images in {input_dir}")

    for image_path in tqdm(image_paths, desc="Converting to event"):
        try:
            # Read image
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            if image is None:
                print(f"Warning: Cannot read {image_path}, skipping")
                continue
            event_image = event_transfer_single(image_path)
            image_name = Path(image_path).stem

            vis_save_path = os.path.join(output_dir, f"{image_name}.png")
            cv2.imwrite(vis_save_path, event_image)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    print(f"Event images saved to {output_dir}")









