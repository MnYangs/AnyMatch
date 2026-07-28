import numpy as np
import torch
import cv2
import os
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from PIL import Image

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "sam_vit_h_4b8939.pth"
IMAGE_PATH = "../../vis_image/00086.png"
OUTPUT_DIR = "output"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "00086.png")

    # Load model
    sam = sam_model_registry["vit_h"](checkpoint=MODEL_PATH)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    # Read image
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Error: Could not read image from {IMAGE_PATH}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    # Generate masks
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image_rgb)

    colored = np.zeros_like(image_rgb)
    for idx, m in enumerate(masks):
        colored[m["segmentation"]] = np.random.randint(0, 255, 3)

    cv2.imwrite(output_path, cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
    print(f"Saved segmentation result to {output_path}")

if __name__ == "__main__":
    main()