# BERT Base Model (Uncased)

## Overview

**BERT (Bidirectional Encoder Representations from Transformers)** is a transformer-based language model developed by **Google Research**, introduced in the seminal paper by **Jacob Devlin et al.** in 2018 and published at **NAACL 2019**. The **bert-base-uncased** variant is the most widely used version, featuring **12 layers**, **768 hidden dimensions**, **12 attention heads**, and **110 million parameters**. It is trained on English text in a self-supervised manner using masked language modeling (MLM) and next sentence prediction (NSP).

- **Paper**: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (NAACL 2019)
- **Source Repository**: [https://github.com/google-research/bert](https://github.com/google-research/bert)
- **Download**: [https://huggingface.co/google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
- **License**: Apache 2.0

## Model Description

BERT is a **bidirectional transformer encoder** pre-trained on a large corpus of English text. Unlike traditional recurrent neural networks (RNNs) that process text sequentially, or autoregressive models like GPT that only attend to preceding tokens, BERT uses a bidirectional attention mechanism that allows each token to attend to both left and right context simultaneously. This enables the model to learn deeply contextualized word representations.

The model is **uncased**, meaning it does not distinguish between uppercase and lowercase letters (e.g., "English" and "english" are treated identically). Accent markers are also stripped.

### Architecture Specifications

| Property | Value |
|---|---|
| Layers (Transformer Blocks) | 12 |
| Hidden Size | 768 |
| Attention Heads | 12 |
| Parameters | 110M |
| Vocabulary Size | 30,522 |
| Max Sequence Length | 512 |
| Tokenizer | WordPiece (uncased) |
| Activation Function | GELU |
| Position Embeddings | Learned (absolute) |

### Pre-training Objectives

BERT is pre-trained with two unsupervised objectives:

1. **Masked Language Modeling (MLM)**: 15% of input tokens are randomly masked, and the model must predict the original tokens. The masking procedure is:
   - 80% of the time: replaced with `[MASK]`
   - 10% of the time: replaced with a random token
   - 10% of the time: left unchanged

2. **Next Sentence Prediction (NSP)**: The model receives two sentences and predicts whether the second sentence follows the first in the original corpus (50% consecutive, 50% random).

## Model Variants

BERT was originally released in multiple variants:

| Model | Layers | Hidden | Heads | Parameters | Language |
|---|---|---|---|---|---|
| **bert-base-uncased** | **12** | **768** | **12** | **110M** | **English** |
| bert-base-cased | 12 | 768 | 12 | 110M | English |
| bert-large-uncased | 24 | 1024 | 16 | 340M | English |
| bert-large-cased | 24 | 1024 | 16 | 340M | English |
| bert-base-chinese | 12 | 768 | 12 | 110M | Chinese |
| bert-base-multilingual-cased | 12 | 768 | 12 | 110M | 104 languages |

## Download

The model is available via Hugging Face:

- **Model Hub**: [https://huggingface.co/google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)

### Installation

```bash
pip install transformers torch
```

## Usage

### Masked Language Modeling (Fill-Mask)

```python
from transformers import pipeline

unmasker = pipeline('fill-mask', model='bert-base-uncased')
result = unmasker("Hello I'm a [MASK] model.")
print(result)
```

### Feature Extraction

```python
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

# output.last_hidden_state contains the contextualized embeddings
# Shape: (batch_size, sequence_length, 768)
```

### Sequence Classification (Fine-tuning)

```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize your dataset and fine-tune with Trainer API
```

### Token Classification (NER)

```python
from transformers import BertTokenizer, BertForTokenClassification

model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=9)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
```

### Question Answering

```python
from transformers import BertTokenizer, BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
```

## Training Details

### Training Data

| Corpus | Size |
|---|---|
| BookCorpus | 11,038 unpublished books (800M words) |
| English Wikipedia | 2,500M words (excluding lists, tables, headers) |

### Training Procedure

| Parameter | Value |
|---|---|
| Hardware | 4 Cloud TPUs (16 TPU chips total) |
| Training Steps | 1,000,000 |
| Batch Size | 256 |
| Sequence Length | 128 (90% of steps) / 512 (10% of steps) |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Learning Rate Warmup | 10,000 steps |
| Learning Rate Decay | Linear |
| Weight Decay | 0.01 |
| β₁, β₂ | 0.9, 0.999 |
| Dropout | 0.1 |

## GLUE Benchmark Results

When fine-tuned on downstream tasks, BERT base achieves:

| Task | Score |
|---|---|
| MNLI (m/mm) | 84.6 / 83.4 |
| QQP | 71.2 |
| QNLI | 90.5 |
| SST-2 | 93.5 |
| CoLA | 52.1 |
| STS-B | 85.8 |
| MRPC | 88.9 |
| RTE | 66.4 |
| **Average** | **79.6** |

## Key Features

- **Bidirectional context**: Unlike GPT, BERT attends to both left and right context simultaneously.
- **Transfer learning paradigm**: Pre-train once, fine-tune for many downstream tasks with minimal task-specific architecture changes.
- **State-of-the-art (at release)**: Set new records on 11 NLP benchmarks including GLUE, SQuAD, and SWAG.
- **Versatile**: Supports sequence classification, token classification, question answering, and feature extraction.
- **Uncased**: Lowercased text, making it robust to casing variations.
- **Hugging Face integration**: Fully supported in the Transformers library with extensive documentation.

## Limitations

- **Not for text generation**: BERT is an encoder-only model and cannot autoregressively generate text (use GPT for generation).
- **Training data bias**: May exhibit gender and racial biases (e.g., associating "nurse" with women and "carpenter" with men).
- **Fixed vocabulary**: Uses WordPiece tokenization with a fixed 30K vocabulary; out-of-vocabulary words are split into subwords.
- **512 token limit**: Cannot process documents longer than 512 tokens without truncation or chunking strategies.
- **English-only**: Only trained on English text; multilingual variants are available separately.

## Citation

If you use BERT in your research, please cite the following paper:

```bibtex
@article{devlin2018bert,
  author    = {Jacob Devlin and
               Ming{-}Wei Chang and
               Kenton Lee and
               Kristina Toutanova},
  title     = {{BERT:} Pre-training of Deep Bidirectional Transformers for Language
               Understanding},
  journal   = {CoRR},
  volume    = {abs/1810.04805},
  year      = {2018},
  url       = {http://arxiv.org/abs/1810.04805},
  archivePrefix = {arXiv},
  eprint    = {1810.04805},
}
```

## References

- Model Download: [https://huggingface.co/google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
- Paper: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
- GitHub Repository: [https://github.com/google-research/bert](https://github.com/google-research/bert)
- Hugging Face Documentation: [https://huggingface.co/docs/transformers/model_doc/bert](https://huggingface.co/docs/transformers/model_doc/bert)