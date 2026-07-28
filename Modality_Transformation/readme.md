# Modality Transformation Module in AnyMatch

## Overview

The Modality Transformation module converts the original single‑view visible (RGB) image into multiple target modalities—**Infrared (IR)**, **Depth**, **Normal**, and **Event**—using dedicated cross‑modal translation models. This simulates real‑world sensor differences and provides diverse multi‑modal training pairs.

---

## Workflow (Step‑by‑Step)

1. **Infrared (IR) Modality**  
   - Uses a pre‑trained RGB‑to‑IR diffusion model (**DiffV2IR**).  
   - The model is first fine‑tuned with Low‑Rank Adaptation (LoRA) on a large‑scale RGB‑IR paired dataset.  
   - During inference, the input RGB image serves as the conditional input to synthesize the corresponding infrared image.
   ```bash
   python ./Modality_Transformation/RGB2IR/rgb2ir.py
   ```
2. **Depth Modality**  
   - Directly applies a pre‑trained monocular relative depth estimation model (**MoGe**) to predict a dense depth map from the input image.
   ```bash
   python ./Modality_Transformation/RGB2depth/rgb2depth.py
   ```
3. **Normal Modality**  
   - Directly applies a pre‑trained monocular surface normal estimation model (**MoGe**) to predict a surface normal map from the input image.
   ```bash
   python ./Modality_Transformation/RGB2normal/rgb2normal.py
   ```
4. **Event Modality**  
   - Employs a motion‑simulation method to synthesize event streams from static visible image pairs.  
   - The simulation mimics the asynchronous per‑pixel brightness changes captured by event cameras.
   ```bash
   python ./Modality_Transformation/RGB2normal/rgb2event.py
   ```
5. **Other Modality**  
   - If you want to transform to other modality, please make sure to set the resolution to 512*512. 
   - The input image is from "./View_Transformation/AnyMatch_results/image1" and is output to "./View_Transformation/AnyMatch_results/image_other"
   ```bash
   python ./Modality_Transformation/RGB2other/....py
   ```
---

## Key Algorithms and Resources

| Modality | Algorithm / Model | Purpose | Official Links |
|----------|-------------------|---------|----------------|
| **Infrared** | DiffV2IR | Visible‑to‑infrared image translation via a diffusion model | - arXiv: [2503.19012](https://arxiv.org/abs/2503.19012) <br> - GitHub: [LidongWang-26/DiffV2IR](https://github.com/LidongWang-26/DiffV2IR) |
| **Depth & Normal** | MoGe (Monocular Geometry) | Dense relative depth and surface normal estimation from a single RGB image | - GitHub: [microsoft/MoGe](https://github.com/microsoft/MoGe) <br> - Project Page (MoGe‑1): [https://wangrc.site/MoGePage](https://wangrc.site/MoGePage) <br> - Project Page (MoGe‑2): [https://wangrc.site/MoGe2Page](https://wangrc.site/MoGe2Page) <br> - Hugging Face (Depth): [`Ruicheng/moge-vitl`](https://huggingface.co/Ruicheng/moge-vitl) <br> - Hugging Face (Normal): [`Ruicheng/moge-2-vitl-normal`](https://huggingface.co/Ruicheng/moge-2-vitl-normal) |
| **Event** | ESIM (Event Simulator) | Open‑source event camera simulator for generating realistic event streams from intensity images | - GitHub: [Arieswu0324/rpg_esim](https://github.com/Arieswu0324/rpg_esim) <br> - Project Page: [https://rpg.ifi.uzh.ch/esim.html](https://rpg.ifi.uzh.ch/esim.html) |

---

## Inputs

The input images are from `./View_Transformation/AnyMatch_results/image1`.

## Outputs

- Synthesized multi‑modal images: **Infrared**, **Depth**, **Normal**, and **Event** representations of the original scene.
- These outputs are paired with the geometric annotations (camera parameters, depth maps, and pixel‑wise correspondences) inherited from the View Transformation module.

Together with the View Transformation outputs, the Modality Transformation results form a complete multi‑view, multi‑modal training sample with strict 3D geometric supervision.

---

## Acknowledgments
*We thank the authors of MINIMA (https://github.com/LSXI7/MINIMA) for sharing their code, which inspired our view transformation design.*
Please cite:
```bibtex
@inproceedings{ren2025minima,
  title={MINIMA: Modality Invariant Image Matching},
  author={Ren, Jiangwei and Jiang, Xingyu and Li, Zizhuo and Liang, Dingkang and Zhou, Xin and Bai, Xiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
