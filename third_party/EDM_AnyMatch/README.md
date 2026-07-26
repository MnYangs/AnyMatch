# EDM: Efficient Deep Feature Matching

[![arXiv](https://img.shields.io/badge/arXiv-2503.05122-b31b1b.svg?style=flat)](https://arxiv.org/abs/2503.05122)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dNtUw1DAdVUcbvbfi6IXSBlLh_WGeqlG?usp=drive_link)
[![Open in Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/xi-li/EDM)


https://github.com/user-attachments/assets/f7e0e026-6b57-4577-a26c-185f82be725e

## Installation
```shell
conda env create -f environment.yaml
conda activate edm
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt 
```
We provide our pretrained model and a fixed onnx model in [google drive](https://drive.google.com/drive/folders/1PkYNihwgnNwqQeeewBz4OUDrvY8xFdH0?usp=sharing) and [Baidu Netdisk](https://pan.baidu.com/s/1AZloQae-cNkCkITHDsgdJA?pwd=na4r). Please place ckpt in folder weights/ and onnx model in folder deploy/.

## Demo
See demo_single_pair.ipynb

## Deployment
See subdirectory [deploy](/deploy)
### ONNX Model
Exporting onnx model first:
```shell
cd deploy
pip install -r requirements_deploy.txt 
python export_onnx.py
```
Run demo on ONNX Runtime using TensorRT backend:
```shell
python run_onnx.py
```
### C++ Inference demo
Refer to [edm_onnx_cpp](/deploy/edm_onnx_cpp/)


## Testing
Setup the testing subsets of ScanNet and MegaDepth first. 

The test and training can be downloaded by [download link](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf?usp=sharing) provided by [LoFTR](https://github.com/zju3dv/LoFTR).

Create symlinks from the previously downloaded datasets to `data/{{dataset}}/test`.

```shell
# set up symlinks
ln -s /path/to/scannet-1500-testset/* data/scannet/test
ln -s /path/to/megadepth-1500-testset/* data/megadepth/test
```

### MegaDepth dataset
```shell
bash scripts/reproduce_test/outdoor.sh
```

### ScanNet dataset
```shell
bash scripts/reproduce_test/indoor.sh
```

## Training
Prepare training data according to the settings of [LoFTR](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md).
```shell
bash scripts/reproduce_train/outdoor.sh
```

## Acknowledgement
Part of the code is based on [EfficientLoFTR](https://github.com/zju3dv/efficientloftr) and [RLE](https://github.com/Jeff-sjtu/res-loglikelihood-regression). We thank the authors for their useful source code.

## Citation
If you find this project useful, please cite:
```bibtex
@InProceedings{Li_2025_ICCV,
    author    = {Li, Xi and Rao, Tong and Pan, Cihui},
    title     = {EDM: Efficient Deep Feature Matching},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {26198-26208}
}
```
