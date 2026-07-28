# BLIP: Bootstrapping Language-Image Pre-training

## Overview

**BLIP (Bootstrapping Language-Image Pre-training)** is a unified vision-language pre-training (VLP) framework developed by **Salesforce Research**, published at **ICML 2022**. BLIP is designed for both **vision-language understanding** (image-text retrieval, visual question answering, visual reasoning) and **vision-language generation** (image captioning). It introduces a novel **Captioning and Filtering (CapFilt)** mechanism that bootstraps high-quality synthetic captions from noisy web data, significantly improving downstream performance.

- **Paper**: [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086) (ICML 2022)
- **Source Repository**: [https://github.com/salesforce/BLIP](https://github.com/salesforce/BLIP)
- **Successor**: [LAVIS](https://github.com/salesforce/LAVIS) (BLIP is now officially integrated into LAVIS)
- **Blog Post**: [https://blog.salesforceairesearch.com/blip-bootstrapping-language-image-pretraining/](https://blog.salesforceairesearch.com/blip-bootstrapping-language-image-pretraining/)
- **License**: BSD-3-Clause
- **Status**: Deprecated (superseded by LAVIS library)

## Model Description

BLIP introduces a novel **Multimodal Mixture of Encoder-Decoder (MED)** architecture with three functionalities:

1. **Unimodal Encoder**: Separately encodes images (ViT) and text (BERT), used for image-text contrastive learning.
2. **Image-grounded Text Encoder**: Uses cross-attention to inject visual information into the text encoder, used for image-text matching.
3. **Image-grounded Text Decoder**: Replaces the bidirectional self-attention with causal self-attention, enabling text generation conditioned on images.

The key innovation of BLIP is the **CapFilt (Captioning and Filtering)** mechanism:
- A **Captioner** generates synthetic captions for web images.
- A **Filter** removes noisy captions from both the original web text and the synthetic captions.
- The bootstrapped dataset is then used to pre-train a new BLIP model, achieving state-of-the-art results.

### Model Variants

| Variant | Image Encoder | Pre-training Data | Description |
|---|---|---|---|
| BLIP w/ ViT-B (14M) | ViT-Base | 14M images | Lightweight pre-training |
| BLIP w/ ViT-B (129M) | ViT-Base | 129M images | Base model with web captions |
| BLIP w/ ViT-B + CapFilt-L | ViT-Base | 129M images + bootstrapped captions | Base model with CapFilt using ViT-L captioner |
| BLIP w/ ViT-L | ViT-Large | 129M images | Large model variant |

## Download

### Pre-trained Checkpoints

| Checkpoint | Download |
|---|---|
| BLIP w/ ViT-B (14M) | [Download](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth) |
| BLIP w/ ViT-B (129M) | [Download](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth) |
| BLIP w/ ViT-B + CapFilt-L | [Download](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth) |
| BLIP w/ ViT-L | [Download](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth) |

### Finetuned Checkpoints

| Task | BLIP w/ ViT-B | BLIP w/ ViT-B + CapFilt-L | BLIP w/ ViT-L |
|---|---|---|---|
| Image Captioning (COCO) | — | [Download](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth) | [Download](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth) |
| Image-Text Retrieval (COCO) | [Download](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth) | — | [Download](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth) |
| Image-Text Retrieval (Flickr30k) | [Download](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_flickr.pth) | — | [Download](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_flickr.pth) |
| VQA | [Download](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_vqa.pth) | [Download](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth) | — |
| NLVR2 | [Download](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_nlvr.pth) | — | — |

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Image Captioning

```python
import torch
from PIL import Image
from torchvision import transforms
from models.blip import blip_decoupled

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth'
model = blip_decoupled(pretrained=model_url, image_size=384, vit='base')
model.eval()
model = model.to(device)

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])
image = transform(image).unsqueeze(0).to(device)

# Generate caption
with torch.no_grad():
    caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
```

### Visual Question Answering

```python
from models.blip_vqa import blip_vqa

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'
model = blip_vqa(pretrained=model_url, image_size=480, vit='base')
model.eval()
model = model.to(device)

# Generate answer
with torch.no_grad():
    answer = model(image, question, train=False, inference='generate')
```

### Using Hugging Face Transformers (Recommended)

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
```

## Supported Tasks

| Task | Type | Description |
|---|---|---|
| Image Captioning | Generation | Generate natural language descriptions of images |
| Image-Text Retrieval | Understanding | Retrieve relevant images given text, or vice versa |
| Visual Question Answering (VQA) | Understanding | Answer natural language questions about images |
| Visual Reasoning (NLVR2) | Understanding | Determine if a text statement is true for a pair of images |
| Multimodal Feature Extraction | Understanding | Extract joint image-text embeddings |
| Image-Text Matching | Understanding | Score the relevance between an image and a text |
| Zero-shot Video-Text Retrieval | Understanding | Retrieve videos without video-specific training |

## Key Features

- **Unified framework**: Handles both understanding and generation tasks in a single model.
- **CapFilt mechanism**: Bootstraps high-quality captions from noisy web data, reducing the impact of noisy annotations.
- **Multimodal Mixture of Encoder-Decoder (MED)**: Flexible architecture with three functionalities for different objectives.
- **Strong performance**: Achieves state-of-the-art results on image-text retrieval, image captioning, VQA, and NLVR2.
- **Multiple variants**: ViT-Base and ViT-Large backbones for different computational budgets.
- **Zero-shot transfer**: Performs zero-shot video-text retrieval without any video-specific training.
- **Hugging Face integration**: Available via the Transformers library for easy use.

## Citation

If you use BLIP in your research, please cite the following paper:

```bibtex
@inproceedings{li2022blip,
  title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
  author={Junnan Li and Dongxu Li and Caiming Xiong and Steven Hoi},
  year={2022},
  booktitle={ICML},
}
```

## References

- Source Repository: [https://github.com/salesforce/BLIP](https://github.com/salesforce/BLIP)
- Paper: [https://arxiv.org/abs/2201.12086](https://arxiv.org/abs/2201.12086)
- LAVIS Library (Successor): [https://github.com/salesforce/LAVIS](https://github.com/salesforce/LAVIS)
- Hugging Face Demo: [https://huggingface.co/spaces/Salesforce/BLIP](https://huggingface.co/spaces/Salesforce/BLIP)
- Hugging Face Model: [https://huggingface.co/Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)