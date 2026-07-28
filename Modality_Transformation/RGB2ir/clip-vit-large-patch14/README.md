# CLIP ViT-L/14 (clip-vit-large-patch14)

## Overview

**CLIP (Contrastive Language-Image Pre-training)** is a multimodal vision-language model developed by **OpenAI**, published in January 2021. The **ViT-L/14** variant uses a **Vision Transformer (ViT-Large)** with **patch size 14** as the image encoder and a masked self-attention Transformer as the text encoder. It is trained to maximize the similarity between (image, text) pairs via a contrastive loss, enabling **zero-shot image classification** across arbitrary categories without task-specific fine-tuning.

- **Paper**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (ICML 2021)
- **Model Card**: [https://github.com/openai/CLIP/blob/main/model-card.md](https://github.com/openai/CLIP/blob/main/model-card.md)
- **Source Repository**: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)
- **Download**: [https://huggingface.co/openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
- **Release Date**: January 2022
- **License**: MIT License

## Model Description

CLIP learns a joint embedding space for images and text by training on **400 million** (image, text) pairs collected from publicly available internet sources. The ViT-L/14 variant uses:

- **Image Encoder**: Vision Transformer (ViT-Large) with 14×14 patch size, processing images at **224×224** resolution.
- **Text Encoder**: A masked self-attention Transformer (similar to GPT-2 architecture).
- **Training Objective**: Contrastive loss that maximizes cosine similarity between matched image-text pairs while minimizing it for unmatched pairs.

CLIP is not trained on any specific labeled dataset; instead, it learns to associate images with their natural language descriptions. This enables it to perform zero-shot transfer to a wide range of downstream classification and retrieval tasks.

### Model Variants

| Variant | Release Date | Image Encoder | Resolution |
|---|---|---|---|
| ViT-B/32 | Jan 2021 | ViT-Base, patch 32 | 224×224 |
| RN50 | Jan 2021 | ResNet-50 | 224×224 |
| RN101 | Mar 2021 | ResNet-101 | 224×224 |
| RN50×4 | Mar 2021 | EfficientNet-scaled ResNet-50 | 288×288 |
| ViT-B/16 | Jul 2021 | ViT-Base, patch 16 | 224×224 |
| RN50×16 | Jul 2021 | EfficientNet-scaled ResNet-50 | 384×384 |
| **ViT-L/14** | **Jan 2022** | **ViT-Large, patch 14** | **224×224** |
| RN50×64 | Jan 2022 | EfficientNet-scaled ResNet-50 | 448×448 |
| ViT-L/14@336px | Apr 2022 | ViT-Large, patch 14 | 336×336 |

## Download

The model is available for download via Hugging Face:

- **Model Hub**: [https://huggingface.co/openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
- Files include `config.json`, `model.safetensors`, `preprocessor_config.json`, `tokenizer.json`, `vocab.json`, etc.

### Installation

```bash
pip install transformers torch pillow
```

## Usage

### Using Hugging Face Transformers

```python
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"],
    images=image,
    return_tensors="pt",
    padding=True
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # image-text similarity scores
probs = logits_per_image.softmax(dim=1)       # label probabilities
print(probs)
```

### Zero-shot Image Classification

```python
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

image = Image.open("path/to/image.jpg")
class_labels = ["cat", "dog", "car", "tree", "house"]

inputs = processor(
    text=[f"a photo of a {label}" for label in class_labels],
    images=image,
    return_tensors="pt",
    padding=True
)

outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)
predicted_class = class_labels[probs.argmax()]
print(f"Predicted: {predicted_class}")
```

### Image-Text Similarity

```python
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Get image and text embeddings
inputs = processor(text=["a description"], images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

image_embeds = outputs.image_embeds   # normalized image embeddings
text_embeds = outputs.text_embeds     # normalized text embeddings

# Cosine similarity
similarity = (image_embeds @ text_embeds.T).item()
```

## Key Features

- **Zero-shot classification**: Classify images into arbitrary categories without any task-specific training data.
- **Multimodal understanding**: Joint embedding space for images and text, enabling cross-modal retrieval.
- **Vision Transformer backbone**: ViT-Large architecture with 14×14 patch size for high-quality visual representations.
- **Contrastive pre-training**: Trained on 400M image-text pairs from the web.
- **Broad benchmark coverage**: Evaluated on 30+ diverse datasets including ImageNet, CIFAR, Food101, UCF101, etc.
- **Strong generalization**: Matches or exceeds the performance of task-specific supervised models on many benchmarks.

## Limitations

- **Fine-grained classification**: Struggles with tasks requiring fine-grained distinctions (e.g., car models, bird species).
- **Counting and spatial reasoning**: Performs poorly on tasks involving object counting or spatial relationships.
- **English-only**: Only trained on English captions; performance degrades significantly for other languages.
- **Bias and fairness**: Exhibits demographic biases; performance and fairness depend on class taxonomy design.
- **Not for deployment**: Intended for research purposes only; not recommended for untested commercial deployment.
- **Text rendering**: Cannot read or generate legible text within images.

## Citation

If you use CLIP in your research, please cite the following paper:

```bibtex
@inproceedings{radford2021clip,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  booktitle={Proceedings of the 38th International Conference on Machine Learning},
  year={2021},
  organization={PMLR}
}
```

## References

- Model Download: [https://huggingface.co/openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
- Model Card: [https://github.com/openai/CLIP/blob/main/model-card.md](https://github.com/openai/CLIP/blob/main/model-card.md)
- Paper: [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
- Blog Post: [https://openai.com/blog/clip/](https://openai.com/blog/clip/)
- GitHub Repository: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)