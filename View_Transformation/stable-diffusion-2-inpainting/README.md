# Stable Diffusion 2 Inpainting

## Overview

**Stable Diffusion 2 Inpainting** is a diffusion-based text-to-image inpainting model developed by **Robin Rombach, Patrick Esser** and the team at **Stability AI**. It is part of the **Stable Diffusion 2** family, which builds upon the original [Latent Diffusion Model](https://arxiv.org/abs/2112.10752) (CVPR 2022) and uses a new **OpenCLIP-ViT/H** text encoder for improved image quality. The inpainting variant is specifically designed to fill in masked regions of an image guided by a text prompt.

- **Model Type**: Diffusion-based text-to-image inpainting generation model
- **Developers**: Robin Rombach, Patrick Esser (Stability AI)
- **Source Repository**: [Stability AI / stablediffusion](https://github.com/Stability-AI/stablediffusion)
- **Diffusers Documentation**: [Stable Diffusion 2](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_2)
- **Download**: [https://huggingface.co/sd2-community/stable-diffusion-2-inpainting](https://huggingface.co/sd2-community/stable-diffusion-2-inpainting)
- **License**: [CreativeML Open RAIL++-M License](https://huggingface.co/sd2-community/stable-diffusion-2/blob/main/LICENSE-MODEL)

## Model Description

Stable Diffusion 2 Inpainting is a **Latent Diffusion Model (LDM)** that operates in the compressed latent space of a pretrained autoencoder. It uses a fixed, pretrained **OpenCLIP-ViT/H** text encoder for conditioning on natural language prompts. The inpainting model is **resumed from `stable-diffusion-2-base`** (`512-base-ema.ckpt`) and fine-tuned for an additional **200k steps** on the LAION-5B dataset.

The model follows the mask-generation strategy presented in [LAMA](https://github.com/saic-mdal/lama), where the masked image's latent VAE representations are combined with the mask as additional conditioning. The extra input channels of the U-Net that process this information were **zero-initialized** before fine-tuning.

### Key Architecture Details

| Component | Specification |
|---|---|
| Backbone | Latent Diffusion Model (LDM) |
| Text Encoder | OpenCLIP-ViT/H (frozen) |
| Autoencoder Downsampling | 8× (H×W×3 → H/8×W/8×4) |
| Base Checkpoint | `512-base-ema.ckpt` |
| Inpainting Fine-tuning | 200k additional steps |
| Mask Strategy | LAMA-style mask generation |
| Training Resolution | 512×512 |
| Training Data | LAION-5B (aesthetic-filtered, NSFW-filtered) |
| Hardware | 32 × 8 × A100 GPUs |
| Optimizer | AdamW |
| Batch Size | 2048 |
| Learning Rate | 1e-4 (warmup 10k steps) |

## Download

The model checkpoint is available for download from Hugging Face:

- **Model Hub**: [https://huggingface.co/sd2-community/stable-diffusion-2-inpainting](https://huggingface.co/sd2-community/stable-diffusion-2-inpainting)
- **Original checkpoint**: [`512-inpainting-ema.ckpt`](https://huggingface.co/sd2-community/stable-diffusion-2-inpainting/resolve/main/512-inpainting-ema.ckpt)

### Installation

```bash
pip install diffusers transformers accelerate scipy safetensors
```

Optionally, install `xformers` for memory-efficient attention:

```bash
pip install xformers
```

## Usage

### Using the Diffusers Library

```python
import torch
from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe.to("cuda")

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
# image and mask_image should be PIL images
# The mask structure: white for inpainting, black for keeping as is
image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
image.save("./yellow_cat_on_park_bench.png")
```

### Using the DPMSolverMultistepScheduler (recommended)

```python
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import load_image, make_image_grid

init_image = load_image("path/to/image.png").resize((512, 512))
mask_image = load_image("path/to/mask.png").resize((512, 512))

repo_id = "stabilityai/stable-diffusion-2-inpainting"
pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image, num_inference_steps=25).images[0]
```

### Memory Optimization

If you have limited GPU VRAM, use attention slicing:

```python
pipe.enable_attention_slicing()
```

## How It Works

The inpainting process takes three inputs:

1. **Image**: The original image to be edited (PIL image).
2. **Mask**: A binary mask where **white pixels** indicate regions to be inpainted (replaced) and **black pixels** indicate regions to keep unchanged.
3. **Prompt**: A text description of the desired content for the masked region.

The masked image is encoded into the latent space via the VAE encoder, combined with the mask as additional conditioning channels, and processed by the U-Net backbone guided by the text prompt via cross-attention. The model only modifies the masked regions while preserving the unmasked areas.

| Input | Mask | Output |
|---|---|---|
| Original image | White = inpaint, Black = keep | Inpainted result |

## Key Features

- **Text-guided inpainting**: Fill masked regions with content described by natural language prompts.
- **OpenCLIP-ViT/H text encoder**: Improved text understanding compared to Stable Diffusion v1.
- **512×512 resolution**: Generates high-quality inpainted images at 512×512.
- **LAMA-style mask strategy**: Robust mask conditioning for realistic inpainting results.
- **Memory efficient**: Supports `xformers` and attention slicing for low-VRAM environments.
- **Diffusers integration**: Simple API via the Hugging Face Diffusers library.

## Limitations

- Does not achieve perfect photorealism in all cases.
- Cannot render legible text reliably.
- Struggles with compositional tasks (e.g., "a red cube on top of a blue sphere").
- Faces and people may not be generated properly.
- Primarily trained on English captions; performance degrades with other languages.
- The autoencoder is lossy, which may introduce artifacts.
- Training data (LAION-5B) may contain biases; NSFW content was filtered using LAION's NSFW detector.

## Citation

If you use Stable Diffusion 2 Inpainting in your research, please cite the original Latent Diffusion Model paper:

```bibtex
@InProceedings{Rombach_2022_CVPR,
  author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn},
  title     = {High-Resolution Image Synthesis With Latent Diffusion Models},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2022},
  pages     = {10684-10695}
}
```

## References

- Model Download: [https://huggingface.co/sd2-community/stable-diffusion-2-inpainting](https://huggingface.co/sd2-community/stable-diffusion-2-inpainting)
- Diffusers Documentation: [https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_2](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_2)
- Stable Diffusion 2 Announcement: [https://stability.ai/blog/stable-diffusion-v2-release](https://stability.ai/blog/stable-diffusion-v2-release)
- LDM Paper: [https://arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)
- GitHub Repository: [https://github.com/Stability-AI/stablediffusion](https://github.com/Stability-AI/stablediffusion)
- LAMA (Mask Strategy): [https://github.com/saic-mdal/lama](https://github.com/saic-mdal/lama)