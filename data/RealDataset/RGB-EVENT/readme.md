# DSEC Dataset (Stereo Event Camera Dataset for Driving Scenarios)

## Overview

**DSEC (Stereo Event Camera Dataset)** is a large-scale stereo dataset for driving scenarios that contains synchronized data from **event cameras**, **global shutter RGB cameras**, **Lidar**, and **RTK GPS**. It is the **first high-resolution, large-scale stereo dataset with event cameras**, introduced by the Robotics and Perception Group (RPG) at the University of Zurich. DSEC is widely used for benchmarking event-based stereo depth estimation, optical flow, and semantic segmentation in autonomous driving contexts.

- **Paper**: [DSEC: A Stereo Event Camera Dataset for Driving Scenarios](http://rpg.ifi.uzh.ch/docs/RAL21_DSEC.pdf) (IEEE RA-L, 2021)
- **Official Website**: [https://dsec.ifi.uzh.ch/](https://dsec.ifi.uzh.ch/)
- **Source Repository**: [https://github.com/uzh-rpg/DSEC](https://github.com/uzh-rpg/DSEC)
- **License**: [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/legalcode)

## Dataset Description

DSEC captures driving scenes in both favorable and challenging illumination conditions (e.g., low-light, high dynamic range). The sensor suite includes:

- **Two monochrome event cameras** (Prophesee Gen3, VGA resolution: 640 × 480)
- **Two global shutter color cameras** (RGB, 1440 × 1080, frame rate up to 20 Hz)
- **Lidar** (Velodyne VLP-16, 360° coverage, ~300k points/sec)
- **RTK GPS** (for high-precision geolocalization)
- **IMU** (inertial measurement unit)

All sensors are **hardware-synchronized**. The dataset contains **53 sequences** collected by driving in various environments (urban, suburban, highway) and lighting conditions.

### Key Specifications

| Property | Value |
|---|---|
| Event Camera Resolution | 640 × 480 (VGA) |
| RGB Camera Resolution | 1440 × 1080 |
| RGB Camera Type | Global shutter, color |
| Lidar | Velodyne VLP-16 |
| Total Sequences | 53 |
| Ground Truth | Disparity, optical flow, semantic labels, object detection |
| Illumination | Day, night, twilight, high dynamic range |

## Download Link

The dataset (preprocessed split) is available via Google Drive:

- [https://drive.google.com/file/d/1KRV29uyuNfOGhSf9E-19yHbNpJbfMIV-/view?usp=sharing](https://drive.google.com/file/d/1KRV29uyuNfOGhSf9E-19yHbNpJbfMIV-/view?usp=sharing)

Official full dataset downloads are also available on the [DSEC download page](https://dsec.ifi.uzh.ch/dsec-datasets/download/).

## Data Format

The dataset is organized into sequences, each containing synchronized data from all sensors:

```
DSEC/
├── train/
│   ├── zurich_city_00_a/
│   │   ├── events/
│   │   │   ├── left/
│   │   │   │   ├── events.h5          # event stream (left)
│   │   │   │   └── rectify_map.h5     # rectification map
│   │   │   └── right/
│   │   │       ├── events.h5          # event stream (right)
│   │   │       └── rectify_map.h5
│   │   ├── images/
│   │   │   ├── left/
│   │   │   │   └── *.png              # RGB images (left)
│   │   │   └── right/
│   │   │       └── *.png              # RGB images (right)
│   │   ├── disparity/                 # ground truth disparity
│   │   ├── lidar/                     # Lidar point clouds
│   │   ├── gps/                       # RTK GPS data
│   │   └── timestamps/               # synchronization timestamps
│   └── ...
└── test/
    └── ...
```

### Data Types

- **Event streams** (`events.h5`): High-temporal-resolution event data in HDF5 format, including timestamps, pixel coordinates, and polarity.
- **RGB images** (`*.png`): Global shutter color images at 1440 × 1080 resolution.
- **Disparity maps** (`*.png`): Ground truth stereo disparity for evaluation.
- **Lidar point clouds**: 3D point clouds from Velodyne VLP-16.
- **GPS/IMU**: Geolocation and inertial measurements.

## Key Features

- **First high-resolution event camera stereo dataset**: VGA resolution (640 × 480) event cameras.
- **Multi-modal**: Synchronized event cameras, RGB cameras, Lidar, and GPS/IMU.
- **Diverse illumination**: Day, night, twilight, and high-dynamic-range scenes.
- **Driving scenarios**: Urban, suburban, and highway environments.
- **Rich benchmarks**: Provides ground truth for stereo disparity, optical flow, semantic segmentation, and object detection.
- **Competitions**: Hosted CVPR 2021 competition on event-based stereo depth estimation.

## Citation

If you use the DSEC dataset in your research, please cite the following paper:

```bibtex
@Article{Gehrig21ral,
  author = {Mathias Gehrig and Willem Aarents and Daniel Gehrig and Davide Scaramuzza},
  title = {DSEC: A Stereo Event Camera Dataset for Driving Scenarios},
  journal = {IEEE Robotics and Automation Letters},
  year = {2021},
  doi = {10.1109/LRA.2021.3068942}
}
```

For optical flow work, also cite:

```bibtex
@InProceedings{Gehrig3dv2021,
  author = {Mathias Gehrig and Mario Millh\"ausler and Daniel Gehrig and Davide Scaramuzza},
  title = {E-RAFT: Dense Optical Flow from Event Cameras},
  booktitle = {International Conference on 3D Vision (3DV)},
  year = {2021}
}
```

## References

- Official Website: [https://dsec.ifi.uzh.ch/](https://dsec.ifi.uzh.ch/)
- GitHub Repository: [https://github.com/uzh-rpg/DSEC](https://github.com/uzh-rpg/DSEC)
- Paper: [DSEC: A Stereo Event Camera Dataset for Driving Scenarios](http://rpg.ifi.uzh.ch/docs/RAL21_DSEC.pdf)