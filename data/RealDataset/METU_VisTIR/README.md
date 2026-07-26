# METU-VisTIR Dataset

## Overview

The METU-VisTIR dataset is a benchmark dataset for cross-modal local feature matching between thermal infrared (TIR) and visible images. It was introduced alongside **XoFTR: Cross-modal Feature Matching Transformer**, published at the CVPR 2024 Image Matching Workshop.

The dataset comprises thermal and visible images captured across **six diverse scenes** with ground-truth camera poses. Four of the scenes include images captured under both **cloudy** and **sunny** conditions, while the remaining two scenes exclusively feature cloudy conditions. Since the cameras are auto-focus, there may be slight imperfections in the ground-truth camera parameters.

> **Source Repository:** [XoFTR - GitHub](https://github.com/OnderT/XoFTR)

> **Paper:** [XoFTR: Cross-modal Feature Matching Transformer (arXiv)](https://arxiv.org/pdf/2404.09692)

## Download

- [METU-VisTIR Dataset (Google Drive)](https://drive.google.com/file/d/1Sj_vxj-GXvDQIMSg-ZUJR0vHBLIeDrLg/view)
- [Supplementary Files (Google Drive)](https://drive.google.com/file/d/1M5bKE56N--SoA554xEpb4tBzvZjZMVDi/view?usp=sharing)

## Data Format

The dataset is organized into folders by scenario. The directory structure is as follows:

```text
METU-VisTIR/
├── index/
│   ├── scene_info_test/
│   │   ├── cloudy_cloudy_scene_1.npz    # scene info with test pairs
│   │   └── ...
│   ├── scene_info_val/
│   │   ├── cloudy_cloudy_scene_1.npz    # scene info with val pairs
│   │   └── ...
│   └── val_test_list/
│       ├── test_list.txt                # test scenes list
│       └── val_list.txt                 # val scenes list
├── cloudy/                              # cloudy scenes
│   ├── scene_1/
│   │   ├── thermal/
│   │   │   └── images/                  # thermal images
│   │   └── visible/
│   │       └── images/                  # visible images
│   └── ...
└── sunny/                               # sunny scenes
    └── ...
```

The `cloudy_cloudy_scene_*.npz` and `cloudy_sunny_scene_*.npz` files contain ground-truth camera poses and image pairs.

## Scene Summary

| Scene   | Weather Conditions | Description                                          |
|---------|--------------------|------------------------------------------------------|
| 1 - 4   | Cloudy + Sunny     | Images captured under both cloudy and sunny conditions |
| 5 - 6   | Cloudy only        | Images captured exclusively under cloudy conditions   |

## License

The METU-VisTIR dataset is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).

## Citation

If you find this dataset useful for your research, please cite the following paper:

```bibtex
@inproceedings{tuzcuouglu2024xoftr,
  title     = {XoFTR: Cross-modal Feature Matching Transformer},
  author    = {Tuzcuo{\u{g}}lu, {\"O}nder and K{\"o}ksal, Aybora and Sofu, Bu{\u{g}}ra and Kalkan, Sinan and Alatan, A Aydin},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {4275--4286},
  year      = {2024}
}
```
