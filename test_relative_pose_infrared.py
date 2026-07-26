"""
Evaluation script for relative pose estimation on visible-infrared image pairs (VisTIR).

This script evaluates feature matchers on the METU VisTIR dataset by:
1. Loading visible-infrared image pairs with ground-truth camera poses
2. Running feature matching (e.g., LoFTR, RoMa, EDM)
3. Estimating relative pose (R, t) via 5-point algorithm + RANSAC
4. Evaluating rotation/translation errors and epipolar distance precision
"""

import argparse
import json
import logging
import os
import os.path as osp
import time
import warnings
from collections import defaultdict, OrderedDict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from load_model import load_model, choose_method_arguments, add_method_arguments
from src.utils.metrics import (
    estimate_pose,
    relative_pose_error,
    error_auc,
    symmetric_epipolar_distance_numpy,
    epidist_prec,
)
from src.utils.plotting import dynamic_alpha, error_colormap, make_matching_figure


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_vis_tir_pairs_npz(npz_root, npz_list):
    """Load visible-infrared image pairs with ground-truth camera parameters from npz files.

    Each npz file corresponds to a scene and contains:
    - image_paths: list of (visible_path, infrared_path) tuples
    - intrinsics: camera intrinsic matrices
    - distortion_coefs: lens distortion coefficients
    - poses: camera-to-world 4x4 transformation matrices

    Args:
        npz_root: Directory path containing npz scene files.
        npz_list: Text file listing the npz filenames to process.

    Returns:
        scene_pairs: Dict mapping scene names to lists of pair dicts.
                     Each pair dict contains 'im0', 'im1', 'K0', 'K1', 'T_0to1', etc.
    """
    with open(npz_list, 'r') as f:
        npz_names = [name.split()[0] for name in f.readlines()]
    print(f"Parsed {len(npz_names)} npz files from {npz_list}.")

    total_pairs = 0
    scene_pairs = {}

    for name in npz_names:
        print(f"Loading {name}")
        scene_info = np.load(f"{npz_root}/{name}", allow_pickle=True)
        pairs = []

        for pair_info in scene_info['pair_infos']:
            total_pairs += 1
            id0, id1 = pair_info

            im0 = scene_info['image_paths'][id0][0]
            im1 = scene_info['image_paths'][id1][1]
            K0 = scene_info['intrinsics'][id0][0].astype(np.float32)
            K1 = scene_info['intrinsics'][id1][1].astype(np.float32)
            dist0 = np.array(scene_info['distortion_coefs'][id0][0], dtype=float)
            dist1 = np.array(scene_info['distortion_coefs'][id1][1], dtype=float)
            T0 = scene_info['poses'][id0]
            T1 = scene_info['poses'][id1]

            # Compute relative transformation from camera 0 to camera 1
            T_0to1 = np.matmul(T1, np.linalg.inv(T0))

            pairs.append({
                'im0': im0, 'im1': im1,
                'K0': K0, 'K1': K1,
                'dist0': dist0, 'dist1': dist1,
                'T_0to1': T_0to1,
            })

        scene_pairs[name] = pairs

    print(f"Loaded {total_pairs} pairs.")
    return scene_pairs


# --------------------------------------------------------------------------- #
# Visualization utilities
# --------------------------------------------------------------------------- #

def save_matching_figure(path, img0, img1, mkpts0, mkpts1, inlier_mask, T_0to1,
                         K0, K1, t_err=None, R_err=None, conf_thr=5e-4, svg=False):
    """Save a matching visualization figure after RANSAC (only inliers shown).

    Colors matches by their symmetric epipolar distance.

    Args:
        path: Output file path.
        img0, img1: Input images (H, W, 3), uint8 RGB.
        mkpts0, mkpts1: All matched keypoints (N, 2).
        inlier_mask: Boolean mask from RANSAC (N,).
        T_0to1: Ground-truth 4x4 transformation matrix.
        K0, K1: Camera intrinsics (3, 3).
        t_err, R_err: Translation and rotation errors in degrees.
        conf_thr: Epipolar distance threshold for correct/incorrect coloring.
        svg: Whether to save as SVG.
    """
    if inlier_mask is None or len(inlier_mask) == 0:
        return

    mkpts0_inliers = mkpts0[inlier_mask]
    mkpts1_inliers = mkpts1[inlier_mask]

    # Compute essential matrix and epipolar errors
    Tx = np.cross(np.eye(3), T_0to1[:3, 3])
    E_mat = Tx @ T_0to1[:3, :3]

    epi_errs = symmetric_epipolar_distance_numpy(mkpts0_inliers, mkpts1_inliers, E_mat, K0, K1)
    correct_mask = epi_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = int(np.sum(correct_mask))

    color = error_colormap(epi_errs, conf_thr, alpha=dynamic_alpha(len(correct_mask)))

    text = []
    if t_err is not None and R_err is not None:
        text += [f"err_t: {t_err:.2f} °", f"err_R: {R_err:.2f} °"]
    text += [f'Pre.({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(mkpts0_inliers)}']

    make_matching_figure(img0, img1, mkpts0_inliers, mkpts1_inliers,
                         color, text=text, path=path, dpi=100, svg=svg)


def save_matching_figure_before_ransac(path, img0, img1, mkpts0, mkpts1, inlier_mask,
                                        T_0to1, K0, K1, t_err=None, R_err=None,
                                        conf_thr=5e-4, svg=False):
    """Save a matching visualization figure before RANSAC (all matches shown).

    Similar to save_matching_figure but uses all matches (not just inliers).

    Args:
        Same as save_matching_figure.
    """
    Tx = np.cross(np.eye(3), T_0to1[:3, 3])
    E_mat = Tx @ T_0to1[:3, :3]

    epi_errs = symmetric_epipolar_distance_numpy(mkpts0, mkpts1, E_mat, K0, K1)
    correct_mask = epi_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = int(np.sum(correct_mask))

    color = error_colormap(epi_errs, conf_thr, alpha=dynamic_alpha(len(correct_mask)))

    text = []
    if t_err is not None and R_err is not None:
        text += [f"err_t: {t_err:.2f} °", f"err_R: {R_err:.2f} °"]
    text += [f'Pre.({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(mkpts0)}']

    make_matching_figure(img0, img1, mkpts0, mkpts1,
                         color, text=text, path=path, dpi=100, svg=svg)


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #

def _compute_epi_errs(mkpts0, mkpts1, inlier_mask, T_0to1, K0, K1):
    """Compute symmetric epipolar distances for inlier matches only.

    Args:
        mkpts0: Keypoints in image 0 (N, 2).
        mkpts1: Keypoints in image 1 (N, 2).
        inlier_mask: RANSAC inlier mask (N,).
        T_0to1: Ground-truth 4x4 transformation.
        K0, K1: Camera intrinsics (3, 3).

    Returns:
        epi_errs: Epipolar distances for inlier matches.
    """
    Tx = np.cross(np.eye(3), T_0to1[:3, 3])
    E_mat = Tx @ T_0to1[:3, :3]

    if inlier_mask is not None and len(inlier_mask) > 0:
        mkpts0_inliers = mkpts0[inlier_mask]
        mkpts1_inliers = mkpts1[inlier_mask]
        epi_errs = symmetric_epipolar_distance_numpy(mkpts0_inliers, mkpts1_inliers, E_mat, K0, K1)
    else:
        epi_errs = np.array([])
    return epi_errs


def _compute_epi_errs_all(mkpts0, mkpts1, inlier_mask, T_0to1, K0, K1):
    """Compute symmetric epipolar distances for all matches (ignoring inlier mask).

    Args:
        Same as _compute_epi_errs (inlier_mask is ignored).

    Returns:
        epi_errs: Epipolar distances for all matches.
    """
    Tx = np.cross(np.eye(3), T_0to1[:3, 3])
    E_mat = Tx @ T_0to1[:3, :3]
    return symmetric_epipolar_distance_numpy(mkpts0, mkpts1, E_mat, K0, K1)


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


def _aggregate_precisions(precs, precs_no_inlier):
    """Aggregate precision values across sub-scenes sharing the same prefix.

    Args:
        precs: Per-scene precision dict.
        precs_no_inlier: Per-scene precision (no inlier filter) dict.

    Returns:
        agg_precs: Aggregated precision dict.
        agg_precs_no_inlier: Aggregated precision (no inlier) dict.
    """
    temp_precs = defaultdict(lambda: defaultdict(list))
    temp_precs_no_inlier = defaultdict(lambda: defaultdict(list))

    for scene_name, precision_dict in precs.items():
        main_scene = scene_name.split("_scene")[0]
        for threshold, precision in precision_dict.items():
            temp_precs[main_scene][threshold].append(precision)

    for scene_name, precision_dict in precs_no_inlier.items():
        main_scene = scene_name.split("_scene")[0]
        for threshold, precision in precision_dict.items():
            temp_precs_no_inlier[main_scene][threshold].append(precision)

    agg_precs = {
        scene: {thr: np.mean(vals) for thr, vals in thr_dict.items()}
        for scene, thr_dict in temp_precs.items()
    }
    agg_precs_no_inlier = {
        scene: {thr: np.mean(vals) for thr, vals in thr_dict.items()}
        for scene, thr_dict in temp_precs_no_inlier.items()
    }
    return agg_precs, agg_precs_no_inlier


# --------------------------------------------------------------------------- #
# Main evaluation loop
# --------------------------------------------------------------------------- #

def eval_relapose(matcher, data_root, scene_pairs, ransac_thres, thresholds,
                  save_figs, figures_dir=None, method=None, print_out=False, debug=False):
    """Run relative pose estimation evaluation on all scene pairs.

    Args:
        matcher: Feature matcher with __call__(img0, img1, K0, K1, dist0, dist1) -> dict.
        data_root: Root directory Path object for resolving image paths.
        scene_pairs: Dict of scene_name -> list of pair dicts.
        ransac_thres: RANSAC threshold for 5-point algorithm.
        thresholds: List of AUC thresholds (e.g., [5, 10, 20]).
        save_figs: Whether to save visualization figures.
        figures_dir: Output directory for figures.
        method: Matching method name.
        print_out: Whether to print per-pair results.
        debug: If True, only evaluate first 10 pairs.

    Returns:
        scene_pose_auc, agg_pose_auc, precs, precs_no_inlier, agg_precs, agg_precs_no_inlier.
    """
    scene_pose_auc = {}
    precs = {}
    precs_no_inlier = {}

    for scene_name, pairs in scene_pairs.items():
        scene_dir = figures_dir if args.svg else osp.join(figures_dir, scene_name.split(".")[0])
        if save_figs and not osp.exists(scene_dir):
            os.makedirs(scene_dir)

        statis = defaultdict(list)

        logging.info(f"Start evaluation on scene: {scene_name}")
        print(f"Start evaluation on scene: {scene_name}")

        for i, pair in tqdm(enumerate(pairs), smoothing=.1, total=len(pairs)):
            if debug and i > 10:
                break

            T_0to1 = pair['T_0to1']
            im0 = str(data_root / pair['im0'])
            im1 = str(data_root / pair['im1'])

            # Run feature matching (with known intrinsics and distortion coefficients)
            match_res = matcher(im0, im1, pair['K0'], pair['K1'], pair['dist0'], pair['dist1'])

            mkpts0 = match_res['mkpts0']
            mkpts1 = match_res['mkpts1']
            matches = match_res['matches']
            new_K0 = match_res['new_K0']
            new_K1 = match_res['new_K1']
            n = len(matches)

            # Estimate relative pose
            ret = estimate_pose(mkpts0, mkpts1, new_K0, new_K1, thresh=ransac_thres)

            if ret is None:
                R, t, inliers = None, None, None
                t_err, R_err = np.inf, np.inf
                epi_errs = np.array([]).astype(np.float32)
                epi_errs_no_inlier = np.array([]).astype(np.float32)
                statis['failed'].append(i)
            else:
                R, t, inliers = ret
                t_err, R_err = relative_pose_error(T_0to1, R, t)
                epi_errs = _compute_epi_errs(mkpts0, mkpts1, inliers, T_0to1, new_K0, new_K1)
                epi_errs_no_inlier = _compute_epi_errs_all(mkpts0, mkpts1, inliers, T_0to1, new_K0, new_K1)

                if print_out:
                    msg = f"#M={n:5d} R={R_err:.3f}, t={t_err:.3f}"
                    logging.info(msg)

            statis['R_errs'].append(R_err)
            statis['t_errs'].append(t_err)
            statis['epi_errs'].append(epi_errs)
            statis['epi_errs_no_inlier'].append(epi_errs_no_inlier)
            statis['inliers'].append(inliers.sum() / len(mkpts0) if inliers is not None else np.array([]))
            statis['match_nums'].append(n)

            # Save visualization
            if save_figs and ret is not None:
                img0_name = f"{'vis' if 'visible' in pair['im0'] else 'tir'}_{osp.basename(pair['im0']).split('.')[0]}"
                img1_name = f"{'vis' if 'visible' in pair['im1'] else 'tir'}_{osp.basename(pair['im1']).split('.')[0]}"

                img0 = cv2.cvtColor(cv2.imread(im0), cv2.COLOR_BGR2RGB)
                img1 = cv2.cvtColor(cv2.imread(im1), cv2.COLOR_BGR2RGB)

                # After RANSAC (inliers only)
                fig_path = osp.join(scene_dir, f"{img0_name}_{img1_name}_{method}_after_ransac.jpg")
                save_matching_figure(path=fig_path, img0=img0, img1=img1,
                                     mkpts0=mkpts0, mkpts1=mkpts1, inlier_mask=inliers,
                                     T_0to1=T_0to1, K0=new_K0, K1=new_K1,
                                     t_err=t_err, R_err=R_err, svg=args.svg)

                # Before RANSAC (all matches)
                fig_path = osp.join(scene_dir, f"{img0_name}_{img1_name}_{method}_before_ransac.jpg")
                save_matching_figure_before_ransac(path=fig_path, img0=img0, img1=img1,
                                                    mkpts0=mkpts0, mkpts1=mkpts1,
                                                    inlier_mask=inliers, T_0to1=T_0to1,
                                                    K0=new_K0, K1=new_K1,
                                                    t_err=t_err, R_err=R_err, svg=args.svg)

        # ----- Scene-level summary -----
        n_total = len(pairs)
        n_failed = len(statis['failed'])
        logging.info(f"Scene: {scene_name} Total: {n_total} Failed: {n_failed}")
        print(f"Scene: {scene_name} Total: {n_total} Failed: {n_failed}")

        # Pose AUC (max of rotation and translation error)
        pose_errors = np.max(np.stack([statis['R_errs'], statis['t_errs']]), axis=0)
        pose_auc = error_auc(pose_errors, thresholds)
        scene_pose_auc[scene_name] = 100 * np.array([pose_auc[f'auc@{t}'] for t in thresholds])

        # Epipolar distance precision
        epi_err_thr = 5e-4
        dist_thresholds = [epi_err_thr]
        precs[scene_name] = epidist_prec(np.array(statis['epi_errs'], dtype=object),
                                         dist_thresholds, True, True)
        precs_no_inlier[scene_name] = epidist_prec(np.array(statis['epi_errs_no_inlier'], dtype=object),
                                                   dist_thresholds, True, False)

        logging.info(f"{scene_name} {pose_auc}\n{precs}\n{precs_no_inlier}")
        print(f"{scene_name} {pose_auc}\n")

    # Aggregate across scenes
    agg_pose_auc = _aggregate_scenes(scene_pose_auc, thresholds)
    agg_precs, agg_precs_no_inlier = _aggregate_precisions(precs, precs_no_inlier)

    return scene_pose_auc, agg_pose_auc, precs, precs_no_inlier, agg_precs, agg_precs_no_inlier


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def test_relative_pose_vistir(data_root_dir, method="xoftr", exp_name="VisTIR",
                               ransac_thres=1.5, print_out=False, save_dir=None,
                               save_figs=False, debug=False, args=None):
    """Run the full evaluation pipeline for visible-infrared relative pose estimation.

    Args:
        data_root_dir: Root directory Path object for dataset paths.
        method: Feature matching method name.
        exp_name: Experiment name.
        ransac_thres: RANSAC threshold for 5-point algorithm.
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
    npz_root = data_root_dir / 'index/scene_info_test/'
    npz_list = data_root_dir / 'index/val_test_list/test_list.txt'

    # Load pairs and matcher
    scene_pairs = load_vis_tir_pairs_npz(npz_root, npz_list)
    matcher = load_model(method, args)

    thresholds = [5, 10, 20]

    # Run evaluation
    scene_auc, agg_auc, precs, precs_no_inlier, agg_precs, agg_precs_no_inlier = eval_relapose(
        matcher, data_root_dir, scene_pairs,
        ransac_thres=ransac_thres, thresholds=thresholds,
        save_figs=save_figs, figures_dir=figures_dir,
        method=method, print_out=print_out, debug=debug,
    )

    # Build results dictionary
    results = OrderedDict(method=method, exp_name=exp_name, ransac_thres=ransac_thres,
                          auc_thresholds=thresholds)
    results.update({k: v for k, v in vars(args).items() if k not in results})
    results.update({k: v.tolist() for k, v in agg_auc.items()})
    results.update({k: v.tolist() for k, v in scene_auc.items()})
    results.update({f"precs_{k}": v for k, v in precs.items()})
    results.update({f"precs_no_inlier_{k}": v for k, v in precs_no_inlier.items()})
    results.update({f"agg_precs_{k}": v for k, v in agg_precs.items()})
    results.update({f"agg_precs_no_inlier_{k}": v for k, v in agg_precs_no_inlier.items()})

    logging.info(f"Results: {json.dumps(results, indent=4)}")
    print(f"Results: {json.dumps(results, indent=4)}")

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark Relative Pose on Visible-Infrared')
    choose_method_arguments(parser)
    parser.add_argument('--exp_name', type=str, default="VisTIR")
    parser.add_argument('--data_root_dir', type=str,
                        default="./data/RealDataset/METU_VisTIR/")
    parser.add_argument('--save_dir', type=str, default="./results_relative_infrared_pose/")
    parser.add_argument('--ransac_thres', type=float, default=6)
    parser.add_argument('--e_name', type=str, default=None)
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
        test_relative_pose_vistir(
            Path(args.data_root_dir), args.method, args.exp_name,
            ransac_thres=args.ransac_thres, print_out=args.print_out,
            save_dir=save_dir, save_figs=args.save_figs,
            debug=args.debug, args=args,
        )
    print(f"Elapsed time: {time.time() - tt:.2f}s")