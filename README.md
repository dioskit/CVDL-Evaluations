# CVDL-Evaluations

> **Computer Vision & Deep Learning — Semester 6 Evaluation Projects**
> A collection of evaluation notebooks and presentations for the CVDL course, covering object detection model comparisons and plant disease detection using state-of-the-art deep learning architectures.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Projects](#projects)
  - [1. Object Detection Comparison: Faster R-CNN vs YOLOv5](#1-object-detection-comparison-faster-r-cnn-vs-yolov5)
  - [2. GAF-Net: Potato Leaf Disease Detection (YOLOv8-based)](#2-gaf-net-potato-leaf-disease-detection-yolov8-based)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Results Summary](#results-summary)
- [Author](#author)

---

## Overview

This repository contains the evaluation work for the **Computer Vision and Deep Learning (CVDL)** course. It includes two major projects that explore different facets of modern object detection and classification:

1. A **performance comparison** between two popular object detection architectures — **Faster R-CNN** and **YOLOv5** — on the Pascal VOC 2012 dataset.
2. An implementation of **GAF-Net** (a YOLOv8-based architecture enhanced with GSConv, FASFF, and a Small Target Detection Head) for **potato leaf disease classification**.

Both projects are implemented as Kaggle notebooks and are accompanied by presentation slides.

---

## Projects

### 1. Object Detection Comparison: Faster R-CNN vs YOLOv5

**Notebook:** `g5-bytsewang (1).ipynb`
**Presentation:** `CVDL-ODC-FasterRCNNvsYOLOv5.pptx`

#### Objective
Perform a rigorous performance comparison between **Faster R-CNN (ResNet50-FPN)** and **YOLOv5s** on a subsampled Pascal VOC 2012 dataset to analyze the speed–accuracy trade-off between two-stage and single-stage detectors.

#### Dataset
- **Pascal VOC 2012** — 20 object classes
- Subsampled to **3,000 images** for a feasible training pipeline
- Split: **70% Train / 20% Validation / 10% Test** (2100 / 600 / 300 images)

#### Methodology
| Aspect | Faster R-CNN | YOLOv5s |
|---|---|---|
| **Backbone** | ResNet50-FPN (pretrained) | YOLOv5s (pretrained) |
| **Input Size** | 512×512 | 640×640 |
| **Optimizer** | SGD (lr=0.005, momentum=0.9) | SGD (default YOLOv5 hyps) |
| **Epochs** | 25 | 25 (patience=7) |
| **Batch Size** | 4 | 16 |
| **Framework** | PyTorch / Torchvision | Ultralytics YOLOv5 |

#### Evaluation Metrics
- **mAP@0.5** — Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95** — Mean Average Precision across IoU thresholds 0.5 to 0.95
- **Inference Speed** — milliseconds per image & FPS

#### Key Highlights
- End-to-end reproducible pipeline on **Kaggle (Tesla P100 GPU)**
- CUDA/P100 compatibility fix (PyTorch 2.5.1+cu118)
- Unified evaluation using `torchmetrics.detection.MeanAveragePrecision`
- Publication-quality visualizations: training curves, per-class AP, confusion matrices, and side-by-side detection comparisons

---

### 2. GAF-Net: Potato Leaf Disease Detection (YOLOv8-based)

**Notebook:** `potato-gaf (2).ipynb`
**Presentation:** `GAF-Net_Potato_Disease.pptx`

#### Objective
Implement and evaluate an enhanced **GAF-Net** model based on YOLOv8 for classifying potato leaf diseases, leveraging components like **GSConv**, **FASFF** (Feature-Aligned Spatial Semantic Fusion), and a **Small Target Detection Head**.

#### Dataset
- **Potato Disease Leaf Dataset (PLD)** — 3 classes:
  - 🟠 Early Blight (1,303 train / 163 val / 162 test)
  - 🔴 Late Blight (1,132 train / 151 val / 141 test)
  - 🟢 Healthy (816 train / 102 val / 102 test)
- **Total:** 3,251 training / 416 validation / 405 test images
- Images are 256×256 pixels

#### Architecture
The GAF-Net model enhances the standard YOLOv8 backbone with:
- **GSConv (Ghost Shuffle Convolution)** — Lightweight convolution for efficient feature extraction
- **FASFF (Feature-Aligned Spatial Semantic Fusion)** — Adaptive multi-scale feature fusion
- **Small Target Detection Head** — Improved detection of small/subtle disease patterns

#### Key Highlights
- Built and trained on **Kaggle (NVIDIA Tesla T4 GPU)**
- Full training pipeline: dataset download, preprocessing, model building, training, and evaluation
- Comprehensive metrics: accuracy, precision, recall, F1-score, confusion matrix
- Visualization of sample predictions and class distributions

---

## Repository Structure

```
CVDL-Evaluations/
│
├── README.md                                   # This file
│
├── g5-bytsewang (1).ipynb                      # Faster R-CNN vs YOLOv5 comparison notebook
├── CVDL-ODC-FasterRCNNvsYOLOv5.pptx            # Presentation for Object Detection Comparison
│
├── potato-gaf (2).ipynb                        # GAF-Net Potato Disease Detection notebook
└── GAF-Net_Potato_Disease.pptx                 # Presentation for GAF-Net project
```

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Languages** | Python 3.12 |
| **Deep Learning** | PyTorch, Torchvision, Ultralytics YOLOv5/v8 |
| **Evaluation** | torchmetrics, scikit-learn |
| **Data Processing** | NumPy, Pandas, OpenCV, Pillow |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Kaggle Notebooks (P100 / T4 GPUs) |

---

## Getting Started

### Prerequisites
- Python 3.10+
- PyTorch 2.x with CUDA support
- Kaggle account (for GPU access and datasets)

### Running the Notebooks

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dioskit/CVDL-Evaluations.git
   cd CVDL-Evaluations
   ```

2. **Object Detection Comparison (Faster R-CNN vs YOLOv5):**
   - Upload `g5-bytsewang (1).ipynb` to [Kaggle](https://www.kaggle.com/)
   - Add the dataset: `gopalbhattrai/pascal-voc-2012-dataset`
   - Enable **GPU accelerator** (P100 recommended)
   - Run all cells

3. **GAF-Net Potato Disease Detection:**
   - Upload `potato-gaf (2).ipynb` to [Kaggle](https://www.kaggle.com/)
   - The notebook auto-downloads the dataset via `kagglehub`
   - Enable **GPU accelerator** (T4 or better)
   - Run all cells

---

## Results Summary

### Object Detection Comparison

| Model | Parameters | Training Time | mAP@0.5 | Inference Speed |
|---|---|---|---|---|
| **Faster R-CNN (ResNet50-FPN)** | 41.4M | ~2 hrs | Higher accuracy | Slower |
| **YOLOv5s** | 7.1M | ~17 min | Competitive | Faster |

> *Faster R-CNN provides higher accuracy while YOLOv5 offers significantly faster training and inference — demonstrating the classic two-stage vs single-stage detector trade-off.*

### GAF-Net Potato Disease Detection

| Class | Training Samples | Notes |
|---|---|---|
| Early Blight | 1,303 | Most represented class |
| Late Blight | 1,132 | Moderate representation |
| Healthy | 816 | Least represented class |

> *The GAF-Net architecture, enhanced with GSConv and FASFF modules, achieves strong classification performance on the 3-class potato disease dataset.*

---

## Author

**Group 5** — CVDL Course, Semester 6

- GitHub: [@dioskit](https://github.com/dioskit)

---

## License

This project is for academic/educational purposes as part of university coursework.
