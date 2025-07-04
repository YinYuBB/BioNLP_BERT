# BioNLP_BERT

A repository for BioNLP (Biomedical Natural Language Processing) classification using BERT models from Hugging Face.

## Overview

This repository provides scripts for training, evaluating, and performing inference with BERT models specifically tailored for biomedical Natural Language Processing (BioNLP) classification tasks.

## Dataset

- Abstract Dataset: Contains 53,093 paper abstracts. [Download here](https://drive.google.com/file/d/1suFksEP28eD7bWxMO4nsZdWCKwj2-M-D/view?usp=sharing)
- Manually Annotated Training Dataset: Contains 605 manually labeled papers. [Download here](https://drive.google.com/file/d/1Ws8aIu_C5VDhuxN6x_gjAu6FqHX3KMBa/view?usp=sharing)

## Usage

### Training

Train a BERT model on the BioNLP classification task:

```bash
python train.py \
  --model_name microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
  --train_data train.jsonl \
  --eval_data test.jsonl \
  --output_dir pubmedbert_bionlp_classification
```

To train with cross-validation:
```bash
python Bert_train_crossValidation.py \
  --model_name microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
  --jsonl_file test.jsonl \
  --eval_data test.jsonl \
  --output_dir cross_validation/
  --n_splits 5
```

### Evaluation

To evaluate a trained model:

```bash
python eval.py \
  --model_path ./pubmedbert_bionlp_classification/checkpoint-2440 \
  --jsonl_file test.jsonl
```

For evaluation with cross-validation:
```bash
python Bert_eval_crossValidation.py \
  --model_path ./pubmedbert_bionlp_classification/checkpoint-2440 \
  --jsonl_file test.jsonl
  --output_dir results/
  --n_splits 5
```

### Inference

Run inference using a trained model:

```bash
python inference.py \
  --model_path ./pubmedbert_bionlp_classification/checkpoint-2440 \
  --input_jsonl paper_data.jsonl \
  --output_jsonl predictions.jsonl
```

## Supported Models

This repository works with biomedical BERT models available on Hugging Face, including:
- microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
- microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
- michiyasunaga/BioLinkBERT-base
- michiyasunaga/BioLinkBERT-large
- Other compatible BERT-based models
