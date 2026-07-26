"""
Evaluation script for relative homography estimation on RGB-normal image pairs.

This script evaluates feature matchers on DIODE normal map datasets by:
1. Loading RGB-normal image pairs with homography ground truth
2. Running feature matching (e.g., LoFTR, RoMa, EDM)
3. Computing homography via RANSAC
4. Evaluating matching accuracy and mean corner distance
"""

import argparse
import csv
import json
import logging
import os
import os.path as osp
import time
import warnings
from collections import defaultdict, OrderedDict
from pathlib import Path

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
from kornia.geometry.transform import warp_perspective
from math import pi
from tqdm import tqdm

from load_model import load_model, choose_method_arguments, add_method_arguments
from src.utils.metrics import error_auc
from src.utils.plotting import dynamic_alpha, make_matching_figure2
from src.utils.sample_h import sample_homography


# --------------------------------------------------------------------------- #
# Visualization utilities
# --------------------------------------------------------------------------- #

def save_matching_figure(path, img0, img1, mkpts0, mkpts1, mean_distance, correct_mask, svg=False):
    """Generate and save a matching visualization figure.

    Args:
        path: Output file path.
        img0: First image (H, W, 3), uint8.
        img1: Second image (H, W, 3), uint8.
        mkpts0: Matched keypoints in image 0 (N, 2).
        mkpts1: Matched keypoints in image 1 (N, 2).
        mean_distance: Mean corner distance after homography.
        correct_mask: Boolean mask indicating correct matches (N,).
        svg: Whether to save as SVG (default: False).
    """
    correct_mask = correct_mask.astype(float)
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = int(np.sum(correct_mask))
    n = mkpts0.shape[0]

    color = np.zeros((n, 3), dtype=np.uint8)
    color[correct_mask == 0] = (255, 0, 0)    # red for incorrect
    color[correct_mask == 1] = (0, 255, 0)    # green for correct

    text = [
        f"Mean Distance: {mean_distance:.2f} px",
        f'Precision(3px) ({100 * precision:.1f}%): {n_correct}/{n}',
    ]

    img1 = img1 / 255.0
    make_matching_figure2(img0, img1, mkpts0, mkpts1, color,
                          text=text, path=path, dpi=100, svg=svg)


def draw_homography_comparison(image1, image2, real_warped_corners, warped_corners,
                                mean_dist, file_name, save_path, method):
    """Draw a side-by-side comparison of ground-truth vs. estimated homography corners.

    Args:
        image1: First image (H, W, 3).
        image2: Second image (H, W, 3).
        real_warped_corners: Ground-truth warped corners (4, 2).
        warped_corners: Estimated warped corners (4, 2).
        mean_dist: Mean corner distance.
        file_name: Base name for the saved figure.
        save_path: Output directory.
        method: Matching method name (for title).
    """
    border_size = 100
    image2 = _expand_image(image2, border_size)
    image1 = _expand_image(image1, border_size)
    real_warped_corners = np.array(real_warped_corners + border_size, dtype=np.int32)
    warped_corners = np.array(warped_corners + border_size, dtype=np.int32)

    if image1.dtype != np.uint8:
        image1 = cv2.convertScaleAbs(image1)
    if image2.dtype != np.uint8:
        image2 = cv2.convertScaleAbs(image2)

    real_warped_corners = real_warped_corners.reshape((-1, 1, 2))
    warped_corners = warped_corners.reshape((-1, 1, 2))

    combined = np.hstack((image2, image1))
    combined = cv2.polylines(combined, [real_warped_corners], isClosed=True, color=(0, 255, 0), thickness=2)
    combined = cv2.polylines(combined, [warped_corners], isClosed=True, color=(0, 0, 255), thickness=2)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(combined.astype('uint8'), cv2.COLOR_BGR2RGB))
    plt.title(f'Homography Comparison ({method} Mean Distance: {mean_dist:.2f})')
    plt.axis('off')
    plt.savefig(os.path.join(save_path, f"{file_name}_homography_comparison.png"))
    plt.close()
    print(f"Saved homography comparison image to: {save_path}")


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def _expand_image(image, border_size):
    """Add a constant-value border around an image."""
    return cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size,
                              cv2.BORDER_CONSTANT, value=[0, 0, 0])


def _order_corners(corners):
    """Order four corner points in top-left, top-right, bottom-right, bottom-left order."""
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]
    return rect


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_vis_normal_pairs_npz(data_root, csv_path):
    """Load RGB-normal image pairs with synthetic homographies from a CSV index.

    Each row in the CSV contains a relative path to an RGB image.
    The corresponding normal map path is derived by replacing '/outdoor/' with '/outdoor_normal/',
    and the image path is redirected to the RGB-depth folder for the RGB source.

    Args:
        data_root: Root directory of the dataset.
        csv_path: Path to the CSV index file.

    Returns:
        scene_pairs: Dictionary with scene name as key and list of pair dicts as value.
                     Each pair dict contains 'im0', 'im1', and 'H' (homography).
    """
    config = {
        'perspective': True,
        'scaling': True,
        'rotation': True,
        'translation': True,
        'n_scales': 5,
        'n_angles': 25,
        'scaling_amplitude': 0.05,
        'perspective_amplitude_x': 0.05,
        'perspective_amplitude_y': 0.05,
        'patch_ratio': 0.8,
        'max_angle': 10 * (pi / 180),
        'allow_artifacts': False,
        'translation_overflow': 0.0,
    }

    image_shape = [480, 640]
    pairs = []

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            rel_path = row[0].replace('./', '')
            image_path = os.path.join(data_root, rel_path)

            # Derive normal map path
            normal_path = image_path.replace('/outdoor/', '/outdoor_normal/')
            # Redirect RGB path to the RGB-depth folder
            image_path = image_path.replace('/RGB-normal/', '/RGB-depth/')
            # Append _normal suffix to the normal filename
            normal_path = str(Path(normal_path).parent / (Path(normal_path).stem + '_normal.png'))

            H0 = sample_homography(image_shape, image_path, config).squeeze().numpy()
            H1 = sample_homography(image_shape, normal_path, config).squeeze().numpy()

            pairs.append({'im0': image_path, 'im1': normal_path, 'H': H0})
            pairs.append({'im0': normal_path, 'im1': image_path, 'H': H1})

    return {'DIO-normal': pairs}


# --------------------------------------------------------------------------- #
# Homography transformation
# --------------------------------------------------------------------------- #

def H_transform(img_tensor, homography):
    """Apply a homography warp to an image tensor.

    Args:
        img_tensor: Input image tensor (b, c, h, w).
        homography: Homography matrix (b, 3, 3).

    Returns:
        Warped image tensor with the same shape as input.
    """
    shape = img_tensor.shape[2:]
    return warp_perspective(img_tensor, homography, shape, align_corners=True)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def compute_mean_distance(real_H, pred_H, H, W, visualize=False,
                          save_path=None, file_name=None, image1=None, image2=None, method=None):
    """Compute the mean corner distance between ground-truth and predicted homography.

    Args:
        real_H: Ground-truth homography (3, 3).
        pred_H: Predicted homography (3, 3).
        H, W: Image height and width.
        visualize: Whether to save a comparison figure.
        save_path, file_name, image1, image2, method: Arguments for visualization.

    Returns:
        mean_dist: Mean Euclidean distance between the four warped corners.
    """
    corners = np.array([[0, 0, 1],
                        [W - 1, 0, 1],
                        [0, H - 1, 1],
                        [W - 1, H - 1, 1]])

    real_H_inv = np.linalg.inv(real_H)
    pred_H_inv = np.linalg.inv(pred_H)

    real_warped = np.dot(corners, np.transpose(real_H_inv))
    real_warped = real_warped[:, :2] / real_warped[:, 2:]

    pred_warped = np.dot(corners, np.transpose(pred_H_inv))
    pred_warped = pred_warped[:, :2] / pred_warped[:, 2:]

    real_warped = _order_corners(real_warped)
    pred_warped = _order_corners(pred_warped)

    mean_dist = np.mean(np.linalg.norm(real_warped - pred_warped, axis=1))

    if visualize:
        draw_homography_comparison(image2, image1, real_warped, pred_warped,
                                   mean_dist, file_name, save_path, method)
    return mean_dist


def compute_matching_accuracy(mkpts0, mkpts1, H):
    """Compute matching accuracy at thresholds 1px, 3px, 5px.

    Args:
        mkpts0: Keypoints in image 0 (N, 2).
        mkpts1: Keypoints in image 1 (N, 2).
        H: Ground-truth homography (3, 3).

    Returns:
        accuracies: List of accuracy values at [1px, 3px, 5px].
        n: Number of keypoints.
    """
    n = mkpts0.shape[0]
    if n == 0:
        return [0, 0, 0], 0

    mkpts0_h = np.hstack([mkpts0, np.ones((n, 1))])
    projected = (H @ mkpts0_h.T).T
    projected = projected[:, :2] / projected[:, 2:]

    distances = np.linalg.norm(projected - mkpts1, axis=1)
    thresholds = [1, 3, 5]
    accuracies = [np.mean(distances <= t) for t in thresholds]
    return accuracies, n


def compute_mask(real_H, mkpts0, mkpts1, threshold=3):
    """Compute a binary mask indicating correct matches (symmetric transfer error < threshold).

    Uses symmetric transfer error: average of forward and backward projection errors.

    Args:
        real_H: Ground-truth homography (3, 3).
        mkpts0: Keypoints in image 0 (N, 2).
        mkpts1: Keypoints in image 1 (N, 2).
        threshold: Error threshold in pixels (default: 3).

    Returns:
        mask: Boolean array (N,), True for correct matches.
    """
    mkpts0_h = np.hstack([mkpts0, np.ones((mkpts0.shape[0], 1))])
    mkpts1_h = np.hstack([mkpts1, np.ones((mkpts1.shape[0], 1))])

    proj_1 = (real_H @ mkpts0_h.T).T
    proj_1 = proj_1[:, :2] / proj_1[:, 2, np.newaxis]

    H_inv = np.linalg.inv(real_H)
    proj_0 = (H_inv @ mkpts1_h.T).T
    proj_0 = proj_0[:, :2] / proj_0[:, 2, np.newaxis]

    error = (np.linalg.norm(mkpts1 - proj_1, axis=1) + np.linalg.norm(mkpts0 - proj_0, axis=1)) / 2
    return error < threshold


# --------------------------------------------------------------------------- #
# Scene-level aggregation
# --------------------------------------------------------------------------- #

def _aggregate_scenes(scene_pose_auc, thresholds):
    """Average AUC results across sub-scenes sharing the same prefix.

    Args:
        scene_pose_auc: Dict mapping scene names to AUC arrays.
        thresholds: List of thresholds used.

    Returns:
        agg_pose_auc: Dict mapping aggregated scene names to mean AUC arrays.
    """
    temp = {}
    for name, auc in scene_pose_auc.items():
        key = name.split("_scene")[0]
        if key not in temp:
            temp[key] = [np.zeros(len(thresholds), dtype=np.float32), 0]
        temp[key][0] += auc
        temp[key][1] += 1
    return {k: v[0] / v[1] for k, v in temp.items()}


# --------------------------------------------------------------------------- #
# Main evaluation loop
# --------------------------------------------------------------------------- #

def eval_relapose(matcher, scene_pairs, save_figs, figures_dir=None,
                  method=None, print_out=False, debug=False, read_color=False):
    """Run homography estimation evaluation on all scene pairs.

    Args:
        matcher: Feature matcher object with __call__(img0, img1) -> dict.
        scene_pairs: Dict of scene_name -> list of pair dicts.
        save_figs: Whether to save visualization figures.
        figures_dir: Output directory for figures.
        method: Matching method name.
        print_out: Whether to print per-pair results.
        debug: If True, only evaluate first 10 pairs.
        read_color: Whether to read images in color (for RoMa).

    Returns:
        scene_pose_auc: Per-scene AUC results.
        agg_pose_auc: Aggregated AUC results.
    """
    scene_pose_auc = {}
    precs = {}
    precs_no_inlier = {}

    for scene_name, groups in scene_pairs.items():
        scene_dir = osp.join(figures_dir, scene_name.split(".")[0]) if save_figs else None
        if save_figs and not osp.exists(scene_dir):
            os.makedirs(scene_dir)

        statis = defaultdict(list)

        logging.info(f"Start evaluation on scene: {scene_name}")
        print(f"Start evaluation on scene: {scene_name}")

        for i, pair in tqdm(enumerate(groups), smoothing=.1, total=len(groups)):
            if debug and i > 10:
                break

            im0_path, im1_path = pair['im0'], pair['im1']
            real_H = pair['H']

            # Read, resize, and apply homography warp to the second image
            if read_color:
                im0 = cv2.imread(im0_path, cv2.IMREAD_COLOR)
                im1 = cv2.imread(im1_path, cv2.IMREAD_COLOR)
                im0 = cv2.resize(im0, (640, 480))
                im1 = cv2.resize(im1, (640, 480))
                im1_t = torch.tensor(im1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                H_t = torch.tensor(real_H, dtype=torch.float32).unsqueeze(0)
                im1_warped = H_transform(im1_t, H_t).squeeze().permute(1, 2, 0).cpu().numpy() * 255
                real_H = H_t.squeeze().cpu().numpy()
            else:
                im0 = cv2.imread(im0_path, cv2.IMREAD_GRAYSCALE)
                im1 = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE)
                im0 = cv2.resize(im0, (640, 480))
                im1 = cv2.resize(im1, (640, 480))
                im1_t = torch.tensor(im1, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
                H_t = torch.tensor(real_H, dtype=torch.float32).unsqueeze(0)
                im1_warped = H_transform(im1_t, H_t).squeeze().cpu().numpy() * 255
                real_H = H_t.squeeze().cpu().numpy()

            # Run feature matching
            match_res = matcher(im0, im1_warped)
            mkpts0 = match_res['mkpts0']
            mkpts1 = match_res['mkpts1']
            img0 = match_res['img0']
            img1 = match_res['img1']
            mconf = match_res['mconf']

            # Normalize confidence to [0, 1] for color mapping
            if len(mconf) > 0:
                conf_min, conf_max = mconf.min(), mconf.max()
                mconf = (mconf - conf_min) / (conf_max - conf_min + 1e-5)
            color = cm.jet(mconf)

            # Ensure images are 3-channel
            if len(img0.shape) == 2:
                H_img, W_img = img0.shape
                img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
                img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            else:
                H_img, W_img, _ = img0.shape

            # Construct file name
            img0_name = f"{'vis' if 'normal' in pair['im0'] else 'normal'}_{osp.basename(pair['im0']).split('.')[0]}"
            img1_name = f"{'vis' if 'normal' in pair['im1'] else 'normal'}_{osp.basename(pair['im1']).split('.')[0]}"
            file_name = f"{img0_name}_{img1_name}"

            # Estimate homography with RANSAC
            if len(mkpts0) >= 4:
                ret_H, _ = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)
                mean_dist = compute_mean_distance(real_H, ret_H, H_img, W_img,
                                                  visualize=save_figs, save_path=scene_dir,
                                                  file_name=file_name, image1=img0,
                                                  image2=img1, method=method) if ret_H is not None else np.inf
            else:
                ret_H = None
                mean_dist = np.inf

            # Save matching visualization
            if save_figs:
                mask0 = compute_mask(real_H, mkpts0, mkpts1, threshold=3)
                fig_path = osp.join(scene_dir, f"{img0_name}_{img1_name}_{method}.jpg")
                save_matching_figure(path=fig_path, img0=img0, img1=img1,
                                     mkpts0=mkpts0, mkpts1=mkpts1,
                                     mean_distance=mean_dist, correct_mask=mask0,
                                     svg=args.svg)

            # Accumulate statistics
            if ret_H is None:
                statis['mean_dist'].append(np.inf)
                statis['failed'].append(i)
                statis['matching_accuracy'].append((0, 0, 0))
                statis['n'].append(0)
            else:
                accuracies, n = compute_matching_accuracy(mkpts0, mkpts1, real_H)
                statis['mean_dist'].append(mean_dist)
                statis['matching_accuracy'].append(accuracies)
                statis['n'].append(n)

                if print_out:
                    msg = f"#M={len(n):5d} R={mean_dist:.3f}"
                    logging.info(msg)

        # ----- Scene-level summary -----
        n_total = len(groups)
        n_failed = len(statis['failed'])
        logging.info(f"Scene: {scene_name} Total: {n_total} Failed: {n_failed}")
        print(f"Scene: {scene_name} Total: {n_total} Failed: {n_failed}")

        # AUC
        mean_dist_all = np.array(statis['mean_dist'])
        thresholds_auc = [1, 3, 5, 7, 10, 15, 20]
        homography_auc = error_auc(mean_dist_all, thresholds_auc)
        scene_pose_auc[scene_name] = 100 * np.array([homography_auc[f'auc@{t}'] for t in thresholds_auc])

        # Average metrics
        total_match_nums = np.zeros(3)
        total_matches = 0
        num_images = 0
        sum_accuracies = np.zeros(3)

        for acc, n in zip(statis['matching_accuracy'], statis['n']):
            acc = np.array(acc)
            total_match_nums += acc * n
            total_matches += n
            num_images += 1
            sum_accuracies += acc

        avg_match_accuracy = total_match_nums / total_matches if total_matches > 0 else 0
        avg_match_nums = total_matches / num_images if num_images > 0 else 0
        avg_acc_array = sum_accuracies / num_images if num_images > 0 else 0
        avg_match_accuracy_nums = total_match_nums / num_images if num_images > 0 else 0

        filtered_dist = mean_dist_all[np.isfinite(mean_dist_all)]
        avg_mean_dist = np.mean(filtered_dist) if len(filtered_dist) > 0 else np.inf

        logging.info(f"Avg Mean Dist: {avg_mean_dist:.4f}")
        logging.info(f"Avg Matching Accuracy: {avg_match_accuracy}")
        logging.info(f"Avg Matching Nums: {avg_match_nums:.2f}")
        logging.info(f"Avg Accuracies Array: {avg_acc_array}")
        logging.info(f"Avg Matching Accuracy Nums: {avg_match_accuracy_nums}")

        print(f"Avg Mean Dist: {avg_mean_dist:.4f}")
        print(f"Avg Matching Accuracy: {avg_match_accuracy}")
        print(f"Avg Matching Nums: {avg_match_nums:.2f}")
        print(f"Avg Accuracies Array: {avg_acc_array}")
        print(f"Avg Matching Accuracy Nums: {avg_match_accuracy_nums}")
        print(f"{scene_name} {homography_auc}")

    # Aggregate across scenes
    agg_pose_auc = _aggregate_scenes(scene_pose_auc, thresholds_auc)

    return scene_pose_auc, agg_pose_auc, precs, precs_no_inlier, {}, {}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def test_relative_pose_visnormal(method="xoftr", exp_name="VisNormal",
                                 ransac_thres=1.5, print_out=False,
                                 save_dir=None, save_figs=False, debug=False, args=None):
    """Run the full evaluation pipeline for RGB-normal homography estimation.

    Args:
        method: Feature matching method name.
        exp_name: Experiment name.
        ransac_thres: RANSAC threshold (unused, kept for interface compatibility).
        print_out: Whether to print per-pair results.
        save_dir: Root directory for saving results.
        save_figs: Whether to save match figures.
        debug: Debug mode (only 10 pairs).
        args: Full argument namespace.
    """
    # Determine checkpoint name
    if method == "roma":
        save_ = "roma" if args.ckpt is None else args.ckpt.split("/")[-1].replace(".ckpt", "")
    else:
        save_ = args.ckpt.split("/")[-1].replace(".ckpt", "") if args.ckpt else "default"

    read_color = (method == 'roma')

    # Create experiment directory
    base_dir = osp.join(save_dir, method, save_)
    if debug and args.debug:
        base_dir = osp.join(base_dir, "debug")

    os.makedirs(base_dir, exist_ok=True)
    counter = 0
    exp_name_full = f"{exp_name}_thresh_{args.thr}" if hasattr(args, 'thr') else exp_name
    exp_dir = osp.join(base_dir, f"{exp_name_full}_{counter}")
    while osp.exists(exp_dir):
        counter += 1
        exp_dir = osp.join(base_dir, f"{exp_name_full}_{counter}")
    os.makedirs(exp_dir)

    # Setup logging
    results_file = osp.join(exp_dir, "results.json")
    logging.basicConfig(filename=results_file.replace('.json', '.log'),
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Args: {args}")

    # Figures directory
    figures_dir = osp.join(exp_dir, "match_figures")
    if save_figs:
        os.makedirs(figures_dir)

    # Data paths
    data_root = "./data/RealDataset/RGB-Normal/"
    csv_path = "./data/RealDataset/RGB-Normal/val_outdoor.csv"

    # Load pairs and matcher
    scene_pairs = load_vis_normal_pairs_npz(data_root, csv_path)
    matcher = load_model(method, args, use_path=False)

    # Run evaluation
    scene_auc, agg_auc, precs, precs_no_inlier, agg_precs, agg_precs_no_inlier = eval_relapose(
        matcher, scene_pairs, save_figs=save_figs, figures_dir=figures_dir,
        method=method, print_out=print_out, debug=debug, read_color=read_color,
    )

    # Build results dictionary
    thresholds = [5, 10, 20]
    results = OrderedDict(method=method, exp_name=exp_name, ransac_thres=ransac_thres,
                          auc_thresholds=thresholds)
    results.update({k: v for k, v in vars(args).items() if k not in results})
    results.update({k: v.tolist() for k, v in agg_auc.items()})
    results.update({k: v.tolist() for k, v in scene_auc.items()})

    logging.info(f"Results: {json.dumps(results, indent=4)}")
    print(f"Results: {json.dumps(results, indent=4)}")

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark Relative Pose on RGB-Normal')
    choose_method_arguments(parser)
    parser.add_argument('--exp_name', type=str, default="VisNormal")
    parser.add_argument('--save_dir', type=str, default="./results_relative_normal_homo/")
    parser.add_argument('--e_name', type=str, default=None)
    parser.add_argument('--ransac_thres', type=float, default=1.5)
    parser.add_argument('--print_out', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_figs', action='store_true')
    parser.add_argument('--svg', action='store_true')

    args, remaining_args = parser.parse_known_args()
    add_method_arguments(parser, args.method)
    args = parser.parse_args()

    save_dir = osp.join(args.save_dir, args.e_name) if args.e_name else args.save_dir

    tt = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_relative_pose_visnormal(
            args.method, args.exp_name, ransac_thres=args.ransac_thres,
            print_out=args.print_out, save_dir=save_dir,
            save_figs=args.save_figs, debug=args.debug, args=args,
        )
    print(f"Elapsed time: {time.time() - tt:.2f}s")