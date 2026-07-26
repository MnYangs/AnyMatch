# RGB-Normal Dataset (DIODE Surface Normal)

## Overview

The **RGB-Normal** dataset is the surface normal subset of **DIODE (Dense Indoor and Outdoor DEpth)**, a large-scale dataset that contains diverse high-resolution color images paired with accurate, dense surface normal maps. It is the **first public dataset to include RGB-Normal data of both indoor and outdoor scenes obtained with a single sensor suite**, introduced by researchers from TTI-Chicago, University of Chicago, and Beihang University.

- **Paper**: [DIODE: A Dense Indoor and Outdoor DEpth Dataset](https://arxiv.org/abs/1908.00463) (2019)
- **Official Website**: [https://diode-dataset.org/](https://diode-dataset.org/)
- **Development Toolkit**: [https://github.com/diode-dataset/diode-devkit](https://github.com/diode-dataset/diode-devkit)
- **License**: MIT License

## Dataset Description

The DIODE surface normal dataset provides **RGB images** and corresponding **surface normal maps** derived from high-precision depth measurements. The data was collected using a **FARO Focus laser scanner** with high precision and long range. Surface normals are computed from the dense depth maps and represent the orientation of each surface point in 3D space. Invalid normals are represented as (0, 0, 0).

The dataset covers a wide variety of indoor and outdoor scenes, captured across different times of day, seasons (summer, fall, winter), and multiple cities.

### Sensor Specifications

| Property | Value |
|---|---|
| Return Density | 99.6% (indoor) / 66.9% (outdoor) |
| Depth Precision | ±1 mm |
| Angular Resolution | 0.009° |
| Max Range | 350 m |
| Min Range | 0.6 m |

### Dataset Statistics

| Split | Scene Type | Scenes | Scans | Images |
|---|---|---|---|---|
| train | indoors | 7 | 80 | 8,574 |
| train | outdoor | 12 | 100 | 16,884 |
| val | indoors | 3 | 10 | 325 |
| val | outdoor | 3 | 10 | 446 |
| test | indoors | 2 | 20 | 753 |
| test | outdoor | 3 | 20 | 876 |

## Download Link

The RGB-Normal dataset (preprocessed split) is available via Google Drive:

- [https://drive.google.com/file/d/1kxI-SI5Wpl37YaYbKXpMnfilLYQbJYVt/view?usp=sharing](https://drive.google.com/file/d/1kxI-SI5Wpl37YaYbKXpMnfilLYQbJYVt/view?usp=sharing)

Official downloads are also available on the [DIODE website](https://diode-dataset.org/):
- DIODE Normal (train): 126 GB
- DIODE Normal (validation): 4.6 GB

## Data Format

The dataset is organized hierarchically by scene and scan:

```
DIODE/
├── indoors/
│   ├── scene_00001/
│   │   ├── scan_00001/
│   │   │   ├── 00001_00001_indoors_150_000.png        # RGB image (1024×768)
│   │   │   └── 00001_00001_indoors_150_000_normal.npy  # surface normal map
│   │   └── ...
│   └── ...
└── outdoor/
    ├── scene_00011/
    │   ├── scan_00001/
    │   │   └── ...
    │   └── ...
    └── ...
```

### File Types

- **RGB images** (`*.png`): Color images with a resolution of 1024 × 768.
- **Surface normal maps** (`*_normal.npy`): Surface normal vector ground truth with the same resolution as the images. Each pixel contains a 3D unit vector (nx, ny, nz) representing the surface orientation. Invalid normals are represented as (0, 0, 0).

## Key Features

- **Indoor + Outdoor**: The first public dataset spanning both environments with a single sensor suite.
- **High precision**: Surface normals derived from depth measurements with ±1 mm accuracy.
- **Diverse conditions**: Daytime and nighttime captures across multiple seasons and cities.
- **Dense annotations**: Full-resolution surface normal maps for every RGB image.
- **Large scale**: Over 25,000 training images across 19 scenes.

## Citation

If you use the RGB-Normal dataset in your research, please cite the following paper:

```bibtex
@article{diode_dataset,
  title={{DIODE}: {A} {D}ense {I}ndoor and {O}utdoor {DE}pth {D}ataset},
  author={Igor Vasiljevic and Nick Kolkin and Shanyi Zhang and Ruotian Luo and
  Haochen Wang and Falcon Z. Dai and Andrea F. Daniele and Mohammadreza Mostajabi and
  Steven Basart and Matthew R. Walter and Gregory Shakhnarovich},
  journal={CoRR},
  volume={abs/1908.00463},
  year={2019},
  url={http://arxiv.org/abs/1908.00463}
}
```

## References

- Official Website: [https://diode-dataset.org/](https://diode-dataset.org/)
- Paper on arXiv: [https://arxiv.org/abs/1908.00463](https://arxiv.org/abs/1908.00463)
- Devkit Repository: [https://github.com/diode-dataset/diode-devkit](https://github.com/diode-dataset/diode-devkit)