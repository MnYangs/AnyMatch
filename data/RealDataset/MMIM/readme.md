# MMIM Dataset (Multimodal Image Matching Dataset)

## Overview

The MMIM (Multimodal Image Matching) dataset is a comprehensive benchmark for **multimodal image matching**, introduced as part of the survey paper **"A Review of Multimodal Image Matching: Methods and Applications"** published in **Information Fusion (2021)**. The dataset was collected and annotated by Jiang et al. to provide a standardized evaluation platform for multimodal image registration and feature matching methods.

- **Paper**: [A review of multimodal image matching: Methods and applications](https://doi.org/10.1016/j.inffus.2021.02.012) (Information Fusion, 2021)
- **Source Repository**: [https://github.com/StaRainJ/Multi-modality-image-matching-database-metrics-methods](https://github.com/StaRainJ/Multi-modality-image-matching-database-metrics-methods)
- **Authors**: Xingyu Jiang, Jiayi Ma, Guobao Xiao, Zhenfeng Shao, Xiaojie Guo

## Dataset Description

The dataset contains **18 common types of multimodal image pairs** with ground truth annotations, covering diverse application domains including medical imaging, remote sensing, and computer vision. For each image pair, **15 to 20 matched landmarks** (i.e., point correspondences) are manually labeled, which can be used to evaluate registration accuracy based on the distance between these matched landmarks.

For image pairs that intrinsically undergo linear geometric transformations, **affine transformation matrices** are also provided. These matrices enable accurate image registration without visible misalignment and serve as ground truth for evaluating feature matching performance — any matched point pair produced by a feature descriptor can be verified against the known transformation.

## Download Link

The dataset is available for download via Google Drive:

- [https://drive.google.com/file/d/12msTsEm-iRE9_I6GWMaU7lT04baFxFxf/view?usp=sharing](https://drive.google.com/file/d/12msTsEm-iRE9_I6GWMaU7lT04baFxFxf/view?usp=sharing)

## Data Format

The dataset is organized into folders corresponding to different modality types. Each image pair includes:

- **Fixed image** and **Moving image**: The source and target images from different modalities.
- **Manually labeled landmarks**: 15–20 point correspondences per pair, annotated by human experts.
- **Affine transformation matrix** (for applicable pairs): The ground-truth geometric transformation between the two images.

## Key Features

- **Multi-modality**: 18 different types of cross-modal image pairs (e.g., visible-infrared, CT-MRI, optical-SAR, etc.).
- **Manual annotations**: Each pair has 15–20 hand-labeled point correspondences for precise evaluation.
- **Ground-truth transformations**: Affine matrices provided for pairs with linear geometric relationships.
- **Multi-domain**: Covers medical imaging, remote sensing, and computer vision applications.
- **Standardized benchmark**: Enables fair comparison across hand-crafted and deep learning-based methods.

## Citation

If you use the MMIM dataset in your research, please cite the following paper:

```bibtex
@article{jiang2021review,
  title={A review of multimodal image matching: Methods and applications},
  author={Jiang, Xingyu and Ma, Jiayi and Xiao, Guobao and Shao, Zhenfeng and Guo, Xiaojie},
  journal={Information Fusion},
  volume={73},
  pages={22--71},
  year={2021},
  publisher={Elsevier}
}
```

## References

- Source Repository: [https://github.com/StaRainJ/Multi-modality-image-matching-database-metrics-methods](https://github.com/StaRainJ/Multi-modality-image-matching-database-metrics-methods)
- Paper: [https://doi.org/10.1016/j.inffus.2021.02.012](https://doi.org/10.1016/j.inffus.2021.02.012)