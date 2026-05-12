# 🇵🇰 Pakistani Politician Image Classifier

A deep learning project that classifies images of 16 prominent Pakistani politicians using CNN-based transfer learning (ResNet-50 and EfficientNet-B2), with an interactive Gradio web interface for real-time inference.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Classes](#classes)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Evaluation](#evaluation)
- [Gradio App](#gradio-app)
- [Results](#results)

---

## Overview

This project tackles a 16-class face recognition problem using pretrained CNNs fine-tuned on a custom Pakistani politician image dataset. Two backbone architectures are trained and compared:

- **ResNet-50** — pretrained on ImageNet (V2 weights), custom classification head
- **EfficientNet-B2** — pretrained on ImageNet, custom classification head

Both models are evaluated on a held-out test set and exposed through an interactive Gradio app that shows top-5 predictions with confidence scores.

---

## Classes

The classifier recognises 16 Pakistani politicians:

| | | | |
|---|---|---|---|
| Ahsan Iqbal | Asif Ali Zardari | Benazir Bhutto | Bilawal Bhutto Zardari |
| Hamza Shehbaz | Imran Khan | Ishaq Dar | Khawaja Asif |
| Maryam Nawaz | Mohsin Naqvi | Murad Ali Shah | Nawaz Sharif |
| Pervez Musharraf | Rana Sanaullah | Shehbaz Sharif | Yousef Raza Gillani |

---

## Dataset

**Source:** [Pakistani Politicians Images Dataset on Kaggle](https://www.kaggle.com/datasets/abdullahsaood/pakistani-politicians-images-dataset)

The dataset is automatically split into three subsets:

| Split | Ratio |
|-------|-------|
| Train | 75%   |
| Validation | 15% |
| Test  | 10%   |

Images are filtered for corruption before training begins. Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`.

---

## Project Structure

```
├── pakistani-politician-image-classification-using-cn.ipynb
├── README.md
└── /kaggle/working/
    ├── dataset_split/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── results/
        ├── resnet50/
        │   ├── best_model.pt
        │   ├── history.json
        │   ├── training_curves.png
        │   ├── confusion_matrix.png
        │   ├── top5_misclassified.png
        │   └── evaluation_summary.json
        ├── efficientnet_b2/
        │   └── (same structure as above)
        └── model_comparison.png
```

---

## Installation

```bash
pip install torch torchvision gradio scikit-learn seaborn matplotlib Pillow split-folders
```

---

## Usage

The notebook is designed to run on **Kaggle** with the dataset attached. Steps are sequential:

1. **Install dependencies** — cell 1
2. **Import libraries and set config** — cells 1–2
3. **Split the dataset** — cell 3
4. **Check class distribution** — cell 4
5. **Build transforms and dataloaders** — cell 5
6. **Define model builders** — cell 6
7. **Train ResNet-50** — cell 8
8. **Train EfficientNet-B2** — cell 9
9. **Evaluate both models** — cells 11–12
10. **Compare models** — cell 13
11. **Launch Gradio app** — cell 14

---

## Model Architecture

Both models use a shared custom classification head replacing the original output layer:

```
Dropout(0.4) → Linear(in_features, 512) → ReLU → Dropout(0.3) → Linear(512, 16)
```

| Model | Backbone In-Features |
|-------|----------------------|
| ResNet-50 | 2048 |
| EfficientNet-B2 | 1408 |

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Epochs | 30 |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Loss Function | CrossEntropyLoss (label smoothing = 0.1) |
| Gradient Clipping | max norm = 1.0 |

**Training augmentations:** random crop, horizontal flip, rotation (±15°), color jitter, random resized crop.

**Validation/Test transforms:** resize to 256×256, center crop to 224×224, ImageNet normalisation.

---

## Evaluation

Each model is evaluated with:

- Overall accuracy
- Macro precision, recall, and F1
- Per-class classification report
- Normalised confusion matrix
- Top-5 most confident misclassifications

Results are saved as `evaluation_summary.json` alongside visualisation plots in the respective model's results directory.

A side-by-side bar chart comparing all metrics across both models is saved to `results/model_comparison.png`.

---

## Gradio App

After training, a Gradio interface lets you classify politician images interactively:

- Upload any politician image
- Choose between **ResNet-50** and **EfficientNet-B2**
- View the predicted politician, confidence score, top-5 predictions, and a confidence bar chart

```python
demo.launch(share=True)  # Generates a public URL on Kaggle
```

---

## Results

The target accuracy is **≥ 90%** on the test set. Training curves (loss and accuracy) are plotted per epoch and saved for both models, enabling direct comparison of convergence behaviour.
