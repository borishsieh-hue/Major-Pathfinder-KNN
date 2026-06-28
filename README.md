# Cross-Modal Alignment of Chest X-ray Images and Clinical Text for Interpretable Diagnosis

Author - 
**Cheng-Chin Hsieh, Jenny Yang, Shirley Chiu, Chitse Chiang**

## Overview

This project builds a multimodal framework that integrates chest X-ray images and radiology reports to predict **Pneumonia** and **Pneumothorax**, using [CheXbert](https://arxiv.org/abs/2004.09167)-derived labels from the MIMIC-CXR dataset. We compare three fusion strategies — **early fusion, late fusion, and cross-attention** — and use Grad-CAM and attention-map alignment analysis to evaluate whether the model's predictions are grounded in genuine cross-modal reasoning or driven primarily by one modality.

## Architecture

- **Image encoder**: DenseNet-121, pretrained on MIMIC-CXR via `torchxrayvision`
- **Text encoder**: BioClinicalBERT (pretrained on MIMIC-III clinical notes), first 8 layers frozen
- **Fusion strategies compared**:
  - *Early fusion* — broadcast-add of text [CLS] embedding into image feature map
  - *Late fusion* — concatenation of pooled image + text features → MLP classifier
  - *Cross-attention* — text tokens as Query, image patches as Key/Value

## Dataset

- Source: [MIMIC-CXR](https://physionet.org/content/mimic-cxr/) chest X-rays + radiology reports, labeled via CheXbert
- After filtering out uncertain (-1) labels: **6,611 records** (5,708 train / 446 val / 457 test), focused on Pneumonia and Pneumothorax
- Hosted on Hugging Face: [cchitse/mimic-cxr-with-chexbert-labels](https://huggingface.co/datasets/cchitse/mimic-cxr-with-chexbert-labels)

## Results

| Strategy | Mean AUROC (val) |
|---|---|
| Late Fusion | 91.8% |
| Early Fusion | 91.3% |
| Cross-Attention | 91.2% |

95% bootstrap confidence intervals (1,000 resamples) showed overlapping ranges across all three strategies — the architectural differences were within natural variation, not statistically distinct.

**Key finding**: the text-only branch alone reached ~91% AUROC, nearly matching full fusion, while image-only reached only ~69–70%. Grad-CAM vs. attention-map alignment (Pearson r) varied widely (-0.6 to 0.6) across patients, indicating the model's cross-modal grounding was inconsistent — a core limitation discussed in the full report below.

## Repository Structure

```
MultiModal-Project/
├── Cross-Modal_Alignment_MIMIC-CXR.ipynb       # Main notebook: full pipeline
├── huggingface data/
│   └── mimic_cxr_with_chexbert_labels.ipynb    # CheXbert labeling of MIMIC-CXR
└── models/
    ├── image_only_baseline_HF.ipynb             # Image-only baseline
    └── text_only_baseline_HF.ipynb              # Text-only baseline
```

## How to Run

All notebooks run on **Google Colab** with no local setup. Start with `Cross-Modal_Alignment_MIMIC-CXR.ipynb`, which covers preprocessing, training, evaluation, and Grad-CAM visualization. Baseline notebooks under `models/` can be run independently.

## Full Report

See [Final_Project_Report.pdf](./Final_Project_Report.pdf) for the complete write-up, including methodology, statistical significance testing, qualitative Grad-CAM analysis, limitations, and proposed future work.

## Dependencies

`torch`, `torchvision`, `transformers`, `datasets`, `scikit-learn`, `matplotlib`, `numpy`, `opencv-python`
