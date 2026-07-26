# GLDv2 Dataset (Google Landmarks Dataset v2)

## Overview

**GLDv2 (Google Landmarks Dataset v2)** is a large-scale benchmark for **instance-level landmark recognition and image retrieval**, introduced in a **CVPR 2020** paper. It contains approximately **5 million images** spanning over **200,000 distinct landmarks** (both human-made and natural). The dataset was created by Google Research and is hosted by the CVDF (Common Visual Data Foundation).

- **Paper**: [Google Landmarks Dataset v2 - A Large-Scale Benchmark for Instance-Level Recognition and Retrieval](https://arxiv.org/abs/2004.01804) (CVPR 2020)
- **Source Repository**: [https://github.com/cvdfoundation/google-landmark](https://github.com/cvdfoundation/google-landmark)
- **Visual Explorer**: [https://storage.googleapis.com/gld-v2/web/index.html](https://storage.googleapis.com/gld-v2/web/index.html)
- **License**: Annotations under CC BY 4.0; images have CC-BY, CC-0, or Public Domain licenses (varies by split)
- **Current Version**: 2.1 (May 2023, with hierarchical labels)

## Dataset Description

GLDv2 provides a massive collection of landmark images downloaded from the web, annotated with landmark labels. The dataset is split into three sets enabling different experimental setups:

| Split | Images | Description |
|---|---|---|
| `train` | 4,132,914 | Training images with landmark labels; all CC-BY licensed |
| `index` | 761,757 | Index images for retrieval evaluation; all CC-0 or Public Domain |
| `test` | 117,577 | Test images with ground truth for recognition and retrieval; CC-0 or Public Domain |

The dataset was associated with two **Kaggle challenges** (landmark recognition and landmark retrieval, 2019), with results discussed at the **CVPR 2019 Workshop on Landmark Recognition**.

### Version 2.1 Additions

The latest version (2.1, May 2023) adds **hierarchical labels** for landmarks, including:
- `category`: Wikimedia URL referring to the class definition
- `supercategory`: Type of landmark (e.g., building, bridge, mountain)
- `hierarchical_label`: Hierarchical label string
- `natural_or_human_made`: Whether the landmark is natural or human-made

## Download Links

All dataset files are available for download from the official source repository. The repository provides a `download-dataset.sh` script for automated downloading. The images are hosted on AWS S3.

### Key Download URLs

- **Metadata files**: [https://s3.amazonaws.com/google-landmark/metadata/](https://s3.amazonaws.com/google-landmark/metadata/)
- **Train images** (500 TAR files, ~1GB each): [https://s3.amazonaws.com/google-landmark/train/](https://s3.amazonaws.com/google-landmark/train/)
- **Index images** (100 TAR files): [https://s3.amazonaws.com/google-landmark/index/](https://s3.amazonaws.com/google-landmark/index/)
- **Test images** (20 TAR files): [https://s3.amazonaws.com/google-landmark/test/](https://s3.amazonaws.com/google-landmark/test/)
- **Ground truth**: [https://s3.amazonaws.com/google-landmark/ground_truth/](https://s3.amazonaws.com/google-landmark/ground_truth/)

### Automated Download

```bash
# Clone the repository
git clone https://github.com/cvdfoundation/google-landmark.git
cd google-landmark

# Download train set (500 files)
mkdir train && cd train
bash ../download-dataset.sh train 499

# Download index set (100 files)
mkdir index && cd index
bash ../download-dataset.sh index 99

# Download test set (20 files)
mkdir test && cd test
bash ../download-dataset.sh test 19
```

## Data Format

### Metadata Files

- `train.csv`: `id`, `url`, `landmark_id` — mapping from image IDs to landmark labels
- `train_clean.csv`: `landmark_id`, `images` — cleaned training data with verified labels
- `train_attribution.csv`: `id`, `url`, `author`, `license`, `title` — image attribution information
- `train_label_to_category.csv`: `landmark_id`, `category` — landmark-to-category mapping
- `train_label_to_hierarchical.csv`: `landmark_id`, `category`, `supercategory`, `hierarchical_label`, `natural_or_human_made`

### Image Storage Structure

Images are stored in a sharded directory structure based on the first three characters of the image ID:

```
train/
├── 0/
│   ├── 1/
│   │   ├── 2/
│   │   │   └── 0123456789abcdef.jpg
│   │   └── ...
│   └── ...
└── ...
```

Each image is a JPG file named by its 16-character hex ID.

### Ground Truth Files

- `recognition_solution_v2.1.csv`: `id`, `landmarks`, `Usage` — recognition ground truth
- `retrieval_solution_v2.1.csv`: `id`, `images`, `Usage` — retrieval ground truth

## Key Features

- **Massive scale**: ~5 million images across 200,000+ landmark classes.
- **Instance-level recognition**: Fine-grained landmark identification, not just category classification.
- **Dual tasks**: Supports both landmark recognition (classification) and landmark retrieval (image search).
- **Rich metadata**: Hierarchical labels, supercategories, natural/human-made classification, and image attribution.
- **Image licenses**: All images have permissive licenses (CC-BY, CC-0, or Public Domain).
- **Kaggle benchmarks**: Established benchmarks with publicly available leaderboard results.
- **Baseline models**: ResNet101-ArcFace baseline provided via TensorFlow models repository.

## Citation

If you use the GLDv2 dataset in your research, please cite the following paper:

```bibtex
@inproceedings{weyand2020GLDv2,
  author = {Weyand, T. and Araujo, A. and Cao, B. and Sim, J.},
  title = {{Google Landmarks Dataset v2 - A Large-Scale Benchmark for Instance-Level Recognition and Retrieval}},
  year = {2020},
  booktitle = {Proc. CVPR},
}
```

For the hierarchical labels extension:

```bibtex
@inproceedings{ramzi2023optimization,
  author = {Ramzi, E. and Audebert, N. and Rambour, C. and Araujo, A. and Bitot, X. and Thome, N.},
  title = {{Optimization of Rank Losses for Image Retrieval}},
  year = {2023},
  booktitle = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
}
```

## References

- Source Repository: [https://github.com/cvdfoundation/google-landmark](https://github.com/cvdfoundation/google-landmark)
- Paper: [https://arxiv.org/abs/2004.01804](https://arxiv.org/abs/2004.01804)
- Dataset Visual Explorer: [https://storage.googleapis.com/gld-v2/web/index.html](https://storage.googleapis.com/gld-v2/web/index.html)
- Kaggle Challenges: [Landmark Recognition 2019](https://kaggle.com/c/landmark-recognition-2019) | [Landmark Retrieval 2019](https://www.kaggle.com/c/landmark-retrieval-2019)
- Baseline Models: [TensorFlow DELF](https://github.com/tensorflow/models/tree/master/research/delf/delf/python/datasets/google_landmarks_dataset)