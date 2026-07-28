import os
import argparse
import cv2
import math
import numpy as np

import torch
import torchvision
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFilter
from pathlib import Path
from tqdm import tqdm
from diffusers import StableDiffusionInpaintPipeline

from NovelViewTransformation_utils import (
    RGBDRenderer,
    transformation_from_parameters,
)
from MoGe.MoGe.model.v1 import MoGeModel


def get_folder_paths(base_path):
    """Get sorted absolute paths of all image files in the given directory.

    Args:
        base_path: root directory path

    Returns:
        sorted list of absolute paths to image files
    """
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
    data_dir = Path(base_path)
    image_paths = [
        str(p) for p in data_dir.rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_paths.sort()
    return image_paths


def resize_and_center_crop(image, disparity):
    """Resize image and disparity map to 512x512 via center cropping.

    First scale by the shortest edge, then center-crop to a square
    to avoid cutting off too much of the longer edge.

    Args:
        image: input image (H, W, 3)
        disparity: disparity map (H, W)

    Returns:
        image_cropped: cropped image (512, 512, 3)
        disparity_cropped: cropped disparity map (512, 512)
    """
    h, w = image.shape[:2]
    shortest_edge = min(h, w)

    if h < w:
        new_h = shortest_edge
        new_w = int(shortest_edge * (w / h))
    else:
        new_w = shortest_edge
        new_h = int(shortest_edge * (h / w))

    image_resized = cv2.resize(image, (new_w, new_h))
    disparity_resized = cv2.resize(disparity, (new_w, new_h))

    crop_size = min(image_resized.shape[:2])
    start_x = (new_w - crop_size) // 2
    start_y = (new_h - crop_size) // 2

    image_cropped = image_resized[start_y: start_y + crop_size, start_x: start_x + crop_size]
    disparity_cropped = disparity_resized[start_y: start_y + crop_size, start_x: start_x + crop_size]

    return image_cropped, disparity_cropped


def project_point_to_3d(x, y, depth, K):
    """Back-project a pixel to a 3D point in camera space using camera intrinsics and depth.

    Args:
        x: pixel x coordinate
        y: pixel y coordinate
        depth: depth value at this pixel
        K: camera intrinsic matrix (3, 3)

    Returns:
        3D coordinate in camera space (3,)
    """
    inv_K = torch.linalg.inv(K)
    pixel = torch.tensor([x, y, 1.0]).to(K.device)
    normalized_coords = inv_K @ pixel * depth
    return normalized_coords


def transform_to_another_camera(point_3d, T):
    """Transform a 3D point from view 1 coordinate system to view 2 coordinate system.

    Args:
        point_3d: 3D coordinate in view 1 (3,)
        T: 4x4 homogeneous transformation matrix

    Returns:
        3D coordinate in view 2 (3,)
    """
    point_3d_homogeneous = torch.cat([point_3d, torch.tensor([1.0]).to(T.device)])
    transformed_point = T @ point_3d_homogeneous
    return transformed_point[:3]


def project_to_image_plane(point_3d, K):
    """Project a 3D point in camera space to 2D pixel coordinates on the image plane.

    Args:
        point_3d: 3D coordinate in camera space (3,)
        K: camera intrinsic matrix (3, 3)

    Returns:
        2D pixel coordinate on the image plane (2,)
    """
    point_2d_homogeneous = K @ point_3d
    point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
    return point_2d


def setup_seed(seed):
    """Fix all random seeds to ensure experiment reproducibility.

    Args:
        seed: random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_correspondences(im_A_depth, K1, K2, T_1to2, step=10):
    """Compute dense correspondences between two views via 3D projection.

    For each pixel in image A (sampled by step), back-project to 3D using the depth map,
    then project to image B via camera transformation and intrinsics.

    Args:
        im_A_depth: depth map of image A (H, W)
        K1: camera intrinsic matrix of image A (3, 3)
        K2: camera intrinsic matrix of image B (3, 3)
        T_1to2: 4x4 transformation matrix from view 1 to view 2
        step: sampling step size, default 10

    Returns:
        matches_A: array of match point coordinates in image A (N, 2)
        matches_B: array of corresponding match point coordinates in image B (N, 2)
    """
    matches_A = []
    matches_B = []

    for y in range(0, im_A_depth.shape[0], step):
        for x in range(0, im_A_depth.shape[1], step):
            depth_A = im_A_depth[y, x].item()
            if depth_A > 0:
                point_3d_A = project_point_to_3d(x, y, depth_A, K1)
                point_3d_B = transform_to_another_camera(point_3d_A.float(), T_1to2)
                point_2d_B = project_to_image_plane(point_3d_B, K2)

                matches_A.append((x, y))
                matches_B.append((point_2d_B[0].item(), point_2d_B[1].item()))

    return np.array(matches_A), np.array(matches_B)


def draw_matching_image(im_A, im_B, im_B_novel, matches_A, matches_B, num_samples=100):
    """Draw a high-resolution matching visualization image.

    Horizontally concatenate three images (original, inpainted, novel view),
    then draw circles and lines for matched points on the concatenated image.

    Args:
        im_A: original input image (H, W, 3)
        im_B: inpainted novel view image (H, W, 3)
        im_B_novel: un-inpainted novel view image (H, W, 3)
        matches_A: match point coordinates in image A (N, 2)
        matches_B: match point coordinates in image B (N, 2)
        num_samples: number of match points to draw, default 100

    Returns:
        concatenated visualization image (H, W*3, 3)
    """
    im_A_cv = im_A.astype(np.uint8)
    im_B_cv = im_B.astype(np.uint8)
    im_B_cv_o = im_B_novel.astype(np.uint8)

    combined = np.hstack((im_A_cv, im_B_cv, im_B_cv_o))
    offset_x = im_A_cv.shape[1]

    if len(matches_A) > 0:
        selected_indices = np.random.choice(
            range(len(matches_A)), min(num_samples, len(matches_A)), replace=False
        )

        for i in selected_indices:
            x_A = int(matches_A[i, 0])
            y_A = int(matches_A[i, 1])
            x_B = int(matches_B[i, 0]) + offset_x
            y_B = int(matches_B[i, 1])

            if (x_A < 0 or x_A >= im_A_cv.shape[1] or y_A < 0 or y_A >= im_A_cv.shape[0] or
                    x_B < offset_x or x_B >= offset_x + im_B_cv.shape[1] or y_B < 0 or y_B >= im_B_cv.shape[0]):
                continue

            cv2.circle(combined, (x_A, y_A), 5, (0, 255, 0), -1)
            cv2.circle(combined, (x_B, y_B), 5, (0, 0, 255), -1)
            cv2.line(combined, (x_A, y_A), (x_B, y_B), (255, 0, 0), 1)

    return combined


def save_image1_filelist(output_dir):
    """Save absolute paths of all files in the image1 directory as a .npy file.

    Called after all images have been processed. Saves to output/index/image1_files.npy.

    Args:
        output_dir: output root directory path
    """
    image1_dir = os.path.join(output_dir, "image1")
    image1_files = [str(p) for p in Path(image1_dir).iterdir() if p.is_file()]
    image1_files.sort()
    npy_path = os.path.join(output_dir, "index", "image1_files.npy")
    np.save(npy_path, np.array(image1_files))
    print(f"Saved {len(image1_files)} file paths to {npy_path}")


class NovelViewTransformation(Dataset):
    """Novel view synthesis dataset.

    Performs depth estimation, random view transformation, and image inpainting
    on input images to generate training data for novel views.
    """

    def __init__(
            self,
            image_path_list,
            width=512,
            height=512,
            device="cuda:3",
            trans_range={"x": 0.3, "y": 0.3, "z": 0.5, "a": 24, "b": 24, "c": 24},
    ):
        """Initialize the dataset.

        Args:
            image_path_list: list of input image paths
            width: output image width, default 512
            height: output image height, default 512
            device: compute device, default "cuda:3"
            trans_range: range for random camera extrinsics sampling
                x/y/z: translation range (in normalized coordinates)
                a/b/c: rotation angle denominator (angle = pi / value)
        """
        self.renderer = RGBDRenderer(device)
        self.width = width
        self.height = height
        self.device = device
        self.trans_range = trans_range
        self.image_path_list = image_path_list

        depth_device = (
            "cuda:3"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.MoGe_depth = MoGeModel.from_pretrained(
            "MoGe/checkpoint/model.pt"
        ).to(depth_device).eval()

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.image_path_list)

    def rand_tensor(self, r, l):
        """Generate a random translation/rotation tensor in range [-r/2, r/2].

        Supports negative values (returns zero tensor, meaning no randomization).

        Args:
            r: random range parameter
            l: batch size

        Returns:
            random tensor (l, 1, 1)
        """
        if r < 0:
            return torch.zeros((l, 1, 1))
        rand = torch.rand((l, 1, 1))
        sign = 2 * (torch.randn_like(rand) > 0).float() - 1
        return sign * (r / 2 + r / 2 * rand)

    def get_rand_ext(self, bs):
        """Generate random camera extrinsics matrices (rotation + translation).

        Randomly samples rotation angles and translation amounts within the
        specified trans_range, then constructs 3x4 extrinsics and inverse matrices.

        Args:
            bs: batch size

        Returns:
            cam_ext: extrinsics matrix (bs, 3, 4)
            cam_ext_inv: inverse extrinsics matrix (bs, 3, 4)
        """
        x, y, z = self.trans_range["x"], self.trans_range["y"], self.trans_range["z"]
        a, b, c = self.trans_range["a"], self.trans_range["b"], self.trans_range["c"]
        cix = self.rand_tensor(x, bs)
        ciy = self.rand_tensor(y, bs)
        ciz = self.rand_tensor(z, bs)
        aix = self.rand_tensor(math.pi / a, bs)
        aiy = self.rand_tensor(math.pi / b, bs)
        aiz = self.rand_tensor(math.pi / c, bs)

        axisangle = torch.cat([aix, aiy, aiz], dim=-1)
        translation = torch.cat([cix, ciy, ciz], dim=-1)
        cam_ext = transformation_from_parameters(axisangle, translation)
        cam_ext_inv = torch.inverse(cam_ext)

        return cam_ext[:, :-1], cam_ext_inv[:, :-1]

    def __getitem__(self, idx, data_path):
        """Get one sample: original image + random novel view synthesis result.

        Pipeline:
        1. Read image and estimate depth with MoGe
        2. Resize and center-crop to 512x512
        3. Apply random scaling and offset augmentation to depth map
        4. Randomly sample focal length and extrinsics
        5. Render novel view using RGBD renderer
        6. Return a dictionary with all data

        Args:
            idx: sample index
            data_path: output directory path (used to check if already processed)

        Returns:
            Dictionary with the following keys, or None if already processed:
            - rgb: original image (1, 3, 512, 512)
            - disp: disparity map (1, 1, 512, 512)
            - warp_mask: novel view hole mask (1, 1, 512, 512)
            - warp_rgb: novel view image (1, 3, 512, 512)
            - warp_disp: novel view disparity map (1, 1, 512, 512)
            - image_name: image file name (without extension)
            - cam_int: camera intrinsic matrix (3, 3)
            - cam_ext: camera extrinsic matrix (3, 4)
        """
        image_path = self.image_path_list[idx]
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        exists_file = data_path+f"{image_name}.jpg"
        if os.path.exists(exists_file):
            return None

        # Read image and estimate depth with MoGe
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        output = self.MoGe_depth.infer(image_tensor)
        disp = 1/output["depth"].cpu().numpy().astype(np.float32)

        # Resize image to 512 on the shortest side and center-crop to square
        H, W = image.shape[:2]
        if H > W:
            from imutils import resize as imresize
            image = imresize(image, width=512)
            disp = imresize(disp, width=512)
        else:
            from imutils import resize as imresize
            image = imresize(image, height=512)
            disp = imresize(disp, height=512)

        image, disp = resize_and_center_crop(image, disp)

        # Normalize depth and add random perturbation (scaling + offset)
        max_d, min_d = disp.max(), disp.min()
        disp = (disp - min_d) / (max_d - min_d)
        image = torch.tensor(image).permute(2, 0, 1) / 255
        disp = (
            torch.tensor(disp).unsqueeze(0) * (0.5 + np.random.random() * 1.5)
            + 0.001 + np.random.random() * 0.3
        )

        # Random focal length (covers common vehicle/phone camera FOV range)
        focal = 0.58 + np.random.random() * 0.3
        K = torch.tensor(
            [[focal, 0, 0.5], [0, focal, 0.5], [0, 0, 1]]
        ).to(self.device)

        # Build RGBD mesh and render novel view
        image = image.to(self.device).unsqueeze(0).float()
        disp = disp.to(self.device).unsqueeze(0).float()
        rgbd = torch.cat([image, disp], dim=1)
        b = image.shape[0]

        cam_int = K.repeat(b, 1, 1)

        mesh = self.renderer.construct_mesh(
            rgbd, cam_int, torch.ones_like(disp), normalize_depth=True
        )

        cam_ext, cam_ext_inv = self.get_rand_ext(b)
        cam_ext = cam_ext.to(self.device)

        warp_image, warp_disp, warp_mask, object_mask = self.renderer.render_mesh(
            mesh, cam_int, cam_ext
        )
        warp_mask = (warp_mask < 0.5).float()
        warp_image = torch.clip(warp_image, 0, 1)

        # Convert normalized intrinsics to pixel intrinsics for 512x512 images
        cam_int[0, :2, :] *= 512

        data = {
            "rgb": image,
            "disp": disp,
            "warp_mask": warp_mask,
            "warp_rgb": warp_image,
            "warp_disp": warp_disp,
            "image_name": image_name,
            "cam_int": cam_int[0],
            "cam_ext": cam_ext[0],
        }

        return data


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, default="./data/Single-View-Data/")
    # parser.add_argument("--data_path", type=str, default="/data/Zizhuo_li/MnYang/AnyMatch/data/Single-View-Data/")
    parser.add_argument("--output_path", type=str, default="AnyMatch_results/")
    parser.add_argument("--prompt", type=str,
                        default="a realistic photo, aligned with visible edges, perspective-corrected, occlusion-aware, seamless transition")
    parser.add_argument("--negative_prompt", type=str,
                        default="neg lowres, bad anatomy, unrealistic geometry, mismatched lighting, distorted perspective, artifacts, watermark")
    parser.add_argument("--seed", type=int, default=2026)
    opt, _ = parser.parse_known_args()

    # Fix random seed
    setup_seed(opt.seed)

    prompt = opt.prompt
    negative_prompt = opt.negative_prompt

    # Load Stable Diffusion Inpainting model
    model_path = "stable-diffusion-2-inpainting"
    # model_path = "/data/Zizhuo_li/MnYang/L2M/L2M/stable-diffusion-2-inpainting"
    inpainting_pipe = StableDiffusionInpaintPipeline.from_pretrained(model_path, torch_dtype=torch.float32)
    inpainting_pipe.set_progress_bar_config(disable=True)
    inpainting_pipe.to("cuda:3")

    # Create output directory structure
    output = opt.output_path
    os.makedirs(os.path.join(output, "image1"), exist_ok=True)
    os.makedirs(os.path.join(output, "image2"), exist_ok=True)
    os.makedirs(os.path.join(output, "image3"), exist_ok=True)
    os.makedirs(os.path.join(output, "depth1"), exist_ok=True)
    os.makedirs(os.path.join(output, "depth2"), exist_ok=True)
    os.makedirs(os.path.join(output, "cam_int"), exist_ok=True)
    os.makedirs(os.path.join(output, "cam_ext"), exist_ok=True)
    os.makedirs(os.path.join(output, "matching_images"), exist_ok=True)
    os.makedirs(os.path.join(output, "index"), exist_ok=True)
    os.makedirs(os.path.join(output, "mask_image"), exist_ok=True)

    # Get all input image paths
    data_folders = get_folder_paths(opt.data_path)

    # Initialize the dataset
    data = NovelViewTransformation(
        image_path_list=data_folders,
        trans_range={"x": 0.3, "y": 0.3, "z": 0.5, "a": 24, "b": 24, "c": 24},
    )

    # Process each image
    for idx in tqdm(range(len(data))):
        image_path = data.image_path_list[idx]
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # Get novel view synthesis data
        batch = data.__getitem__(idx, opt.output_path)

        if batch is None:
            continue

        image, disp = batch["rgb"], batch["disp"]
        w_image, w_disp = batch["warp_rgb"], batch["warp_disp"]
        warp_mask = batch["warp_mask"]
        w_disp = torch.clip(w_disp, 0.01, 100)

        # Convert tensors to PIL images
        input_image = torchvision.transforms.functional.to_pil_image(image[0])
        novel_image = torchvision.transforms.functional.to_pil_image(w_image[0])
        mask_image = torchvision.transforms.functional.to_pil_image(warp_mask[0])

        # Dilate the mask to smooth inpainting region edges
        for _ in range(2):
            mask_image_filter = mask_image.filter(ImageFilter.MaxFilter(3))

        # Inpaint hole regions with Stable Diffusion
        inpaint_image = inpainting_pipe(
            prompt=prompt, image=novel_image, mask_image=mask_image_filter,
            height=512, width=512, num_inference_steps=50,
            negative_prompt=negative_prompt, guidance_scale=7.5
        ).images[0]

        # Save various output files
        input_image.save(os.path.join(output, "image1", batch["image_name"] + ".jpg"))
        inpaint_image.save(os.path.join(output, "image2", batch["image_name"] + ".jpg"))
        novel_image.save(os.path.join(output, "image3", batch["image_name"] + ".jpg"))
        mask_image.save(os.path.join(output, "mask_image", batch["image_name"] + ".jpg"))

        # Save depth maps
        np.save(
            os.path.join(output, "depth1", batch["image_name"] + ".npy"),
            1 / disp.squeeze().cpu().numpy(),
        )
        np.save(
            os.path.join(output, "depth2", batch["image_name"] + ".npy"),
            1 / w_disp.squeeze().cpu().numpy(),
        )

        # Save camera parameters
        cam_int = batch["cam_int"].cpu().numpy()
        cam_ext = batch["cam_ext"].cpu().numpy()
        cam_ext = np.concatenate(
            [cam_ext, np.array([[0.0000, 0.0000, 0.0000, 1.0000]])], 0
        )

        np.savetxt(
            os.path.join(output, "cam_ext", batch["image_name"] + ".txt"),
            cam_ext, fmt="%f", delimiter=",",
        )
        np.savetxt(
            os.path.join(output, "cam_int", batch["image_name"] + ".txt"),
            cam_int, fmt="%f", delimiter=",",
        )

        # Compute matches and draw matching visualization
        im_A_depth = 1 / disp.squeeze()

        K1 = batch["cam_int"].float()
        K2 = batch["cam_int"].float()
        T_1to2 = torch.tensor(cam_ext).cuda(3).float()

        matches_A, matches_B = compute_correspondences(im_A_depth, K1, K2, T_1to2, step=10)

        im_combined = draw_matching_image(
            np.array(input_image),
            np.array(inpaint_image),
            np.array(novel_image),
            matches_A,
            matches_B,
            num_samples=100,
        )

        cv2.imwrite(
            os.path.join(output, "matching_images", batch["image_name"] + ".jpg"),
            cv2.cvtColor(im_combined, cv2.COLOR_RGB2BGR),
        )

    # After all images are processed, save the image1 file list
    save_image1_filelist(output)