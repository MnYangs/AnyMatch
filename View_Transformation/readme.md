# View Transformation Module in AnyMatch

## Overview

The View Transformation module lifts a single 2D image into 3D space and renders novel views with strict 3D geometric consistency, without relying on SfM‑MVS pipelines. It is a core component of the AnyMatch framework for generating multi‑view training data.

---

## Workflow (Step‑by‑Step)

1. **Monocular Depth Estimation**  
   Input: a single‑view image `I_sin`.  
   A pretrained monocular relative depth estimation model `M_rd` predicts a dense relative depth map `D_r`.

2. **Depth Affine Perturbation**  
   Apply random affine transformation to `D_r` to increase robustness against depth scale uncertainty:  
   `D = 1 / ( alpha * (1/D_r) + beta + epsilon )`  
   where `alpha` is sampled from (0.5, 2.0) and `beta` from (0, 0.3).  
   `epsilon` is a small constant for numerical stability.

3. **3D Point Cloud Construction**  
   A random camera intrinsic generator initialises the intrinsic matrix `K`:
   - focal length `f` ∈ (0.58, 0.88)
   - principal point fixed at (0.5, 0.5) (normalised coordinates)  
   Thus `K = [[f, 0, 0.5], [0, f, 0.5], [0, 0, 1]]`.  

   For each pixel `(u, v)` with depth `D(u,v)`, the 3D point `P` is computed as:  
   `P = D(u,v) * K^{-1} * [u, v, 1]^T`  
   (where `^T` denotes transpose). This builds the point cloud `P`.

4. **Random Viewpoint Transformation**  
   A random camera extrinsic generator samples a new pose `[R | T]`:
   - rotation angles within `[-7.5°, +7.5°]`
   - translation: `t_x, t_y` ∈ [-0.3, 0.3], `t_z` ∈ [-0.5, 0.5]  
   The point cloud is transformed to the new camera coordinate system:  
   `P' = R * P + T`

5. **Differentiable Rendering and Inpainting**  
   - Differentiable rendering reprojects `P'` to generate:
     - a preliminary novel‑view image `I_novel`
     - an occlusion mask `M_occ`
     - the corresponding depth map `D'`
   - A diffusion‑based inpainting model `M_inpaint` fills occluded regions:  
     `I_inp = M_inpaint(I_novel, M_occ)`  
   `I_inp` is the final output novel‑view image.

6. **Geometric Verification (Optional)**  
   The generated image pair is validated using a dense matching model (e.g., RoMa) to ensure geometric consistency. Pairs with `PCK@τ ≥ η` are retained.

---

## Key Algorithms and Resources

| Algorithm / Model | Purpose | Official Links |
|-------------------|---------|----------------|
| **MoGe (Monocular Geometry)** | Dense relative depth and surface normal estimation | - GitHub: [microsoft/MoGe](https://github.com/microsoft/MoGe) <br> - Project Page (MoGe‑1): [https://wangrc.site/MoGePage](https://wangrc.site/MoGePage) <br> - Project Page (MoGe‑2): [https://wangrc.site/MoGe2Page](https://wangrc.site/MoGe2Page) <br> - Hugging Face: [`Ruicheng/moge-vitl`](https://huggingface.co/Ruicheng/moge-vitl) (MoGe‑1) <br> - Hugging Face: [`Ruicheng/moge-2-vitl-normal`](https://huggingface.co/Ruicheng/moge-2-vitl-normal) (MoGe‑2 with normals) |
| **Stable Diffusion 2 Inpainting** | Context‑aware completion of occluded regions | - Hugging Face: [`stable_diffusion_2`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_2) <br> - Community mirror: [`sd2-community/stable-diffusion-2-inpainting`](https://huggingface.co/sd2-community/stable-diffusion-2-inpainting) |
| **RoMa (Robust Dense Matching)** | Geometric verification and correspondence filtering | - GitHub: [parskatt/RoMa](https://github.com/parskatt/RoMa) (reference) |

---

## Outputs

- Final inpainted novel‑view image `I_inp`
- Camera intrinsic and extrinsic parameters `{K, [R|T]}`
- Depth maps `{D, D'}`
- Pixel‑level ground‑truth correspondences between the original and novel views

These outputs provide geometrically accurate supervision for training multi‑modal matching models.

The structure of the output file is:

```
AnyMatch_results/                          # Root output directory (--output_path)
│
├── image1/                                # Original input images (512x512, resized & center-cropped)
│   ├── {image_name}.jpg                   #   - RGB image from input source (original view)
│   └── ...
│
├── image2/                                # Inpainted novel view images (SD inpainted)
│   ├── {image_name}.jpg                   #   - RGB image after Stable Diffusion inpainting
│   └── ...                                #     (holes from viewpoint change are filled)
│
├── image3/                                # Un-inpainted novel view images (raw warp)
│   ├── {image_name}.jpg                   #   - RGB image directly warped to novel viewpoint
│   └── ...                                #     (may contain black holes in occluded regions)
│
├── image_depth/                           # The depth map corresponds to the images in image1
│   ├── {image_name}.jpg                   #  
│   └── ...                                #     (Obtained through deep estimation algorithm)
│
├── image_normal/                          # The normal map corresponds to the images in image1
│   ├── {image_name}.jpg                   #   
│   └── ...                                #     
│
├── image_ir/                              # The ir image corresponds to the images in image1
│   ├── {image_name}.jpg                   #   
│   └── ...                                #    
│
├── image_event/                           # The event image corresponds to the images in image1
│   ├── {image_name}.jpg                   #   
│   └── ...                                #  
│
├── depth1/                                # Depth maps of view 1 (original viewpoint)
│   ├── {image_name}.npy                   #   - Float32 numpy array, shape (512, 512)
│   └── ...                                #     Values = 1 / disparity (depth in world coords)
│
├── depth2/                                # Depth maps of view 2 (novel viewpoint)
│   ├── {image_name}.npy                   #   - Float32 numpy array, shape (512, 512)
│   └── ...                                #     Values = 1 / w_disp (depth in world coords)
│
├── cam_int/                               # Camera intrinsic matrices (K)
│   ├── {image_name}.txt                   #   - 3x3 matrix, comma-separated values
│   └── ...                                #     [fx, 0, cx; 0, fy, cy; 0, 0, 1]
│                                         #     (converted to pixel coordinates: cx=256, cy=256)
│
├── cam_ext/                               # Camera extrinsic matrices (R|t)
│   ├── {image_name}.txt                   #   - 4x4 homogeneous transformation matrix
│   └── ...                                #     [R(3x3), t(3x1); 0, 0, 0, 1]
│                                         #     (transforms from view 1 to view 2 coords)
│
├── mask_image/                            # Binary hole masks for inpainting
│   ├── {image_name}.jpg                   #   - White = hole regions (need inpainting)
│   └── ...                                #   - Black = valid regions (already rendered)
│
├── matching_images/                       # Match visualization images
│   ├── {image_name}.jpg                   #   - Side-by-side: [image1 | image2 | image3]
│   └── ...                                #     Green circles = keypoints in view 1
│                                         #     Red circles   = keypoints in view 2
│                                         #     Blue lines    = correspondences
│
└── index/                                 # Index / metadata files
    ├── image1_files.npy                   #   - Numpy array of all image1 file paths  (absolute paths, sorted alphabetically)
    └── images_names_pcks.npz              #   - The PCK values of all image pairs  
```
---

*We thank the authors of L2M (https://github.com/Sharpiless/L2M) for sharing their code, which inspired our view transformation design.*
Please cite:
```bibtex
@inproceedings{Liang2025L2M,
  author    = {Yingping Liang and Yutao Hu and Wenqi Shao and Ying Fu},
  title     = {Learning Dense Feature Matching via Lifting Single 2D Image to 3D Space},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025},
  pages     = {6621--6631}
}
```