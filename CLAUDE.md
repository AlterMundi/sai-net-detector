# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**IMPORTANT: This repository is specifically for developing the DETECTOR stage only.** The verifier (SmokeyNet-like architecture) is developed in a separate repository.

SAI-Net Detector is the first stage of a two-stage wildfire detection system:

1. **Detector (THIS REPO)**: YOLOv8 object detection model for smoke/fire bounding box localization
2. **Verifier (SEPARATE REPO)**: SmokeyNet-like architecture (CNN + LSTM + ViT) for temporal verification and false positive reduction

This repository focuses solely on training and developing the YOLOv8-based detector using PyroSDIS and FASDD datasets. The detector's role is to rapidly identify potential smoke/fire regions with high recall, which are then passed to the verifier stage for temporal analysis and false positive reduction.

## Architecture

### Data Pipeline (Detector-Specific)
- **PyroSDIS**: 33,637 images with smoke bounding boxes in YOLO format
- **FASDD**: 95,314 images (fire/smoke) converted from COCO to YOLO format, mapped to single "smoke" class
- **Combined**: ~129k images for smoke detection training

### Model Architecture (Detector Only)
- **YOLOv8-s/m**: Anchor-free architecture with C2f blocks for rapid smoke/fire localization
- **Single-class detection**: Optimized for "smoke" class detection with high recall
- **Input resolution**: 1440×808 pixels for high-resolution detection
- **Output**: Bounding boxes with confidence scores for smoke/fire regions

### Training Infrastructure
- Distributed training on 2×A100 GPUs using DDP (Distributed Data Parallel)
- Mixed precision training (BF16/FP16) with gradient accumulation
- Ultralytics YOLO framework for streamlined training and evaluation

## Directory Structure

```
sai-net-detector/
├── configs/yolo/         # YOLO configuration files
├── data/
│   ├── raw/              # Raw datasets
│   │   ├── pyro-sdis/    # PyroSDIS dataset in YOLO format (~30k images)
│   │   │   ├── data.yaml # YOLO dataset configuration
│   │   │   ├── images/   # Training/validation images
│   │   │   └── labels/   # YOLO format labels (.txt files)
│   │   └── fasdd/        # FASDD dataset extracted (~100k images)
│   └── yolo/             # Processed datasets for training
├── src/detector/         # Detector source code
├── scripts/              # Training and data processing scripts
├── outputs/yolo/ckpts/   # Model checkpoints and outputs
├── logs/                 # Training logs
├── docs/                 # Documentation and guides
└── LICENSE               # GNU GPL v3 license
```

## Common Commands

### Dataset Preparation
```bash
# Convert FASDD to YOLO format (PyroSDIS already in YOLO format)
python scripts/convert_fasdd_to_yolo.py \
  --src data/raw \
  --dst data/yolo \
  --map-classes smoke
```

### Training

#### YOLOv8 Detector Training
```bash
# Optimal configuration for 2×A100 GPUs with 500GB RAM limit
python scripts/train_detector.py --config optimal

# Equivalent CLI command:
yolo detect train \
  data=configs/yolo/pyro_fasdd.yaml \
  model=yolov8s.pt \
  imgsz=1440 \
  epochs=150 \
  batch=120 \
  device=0,1 \
  workers=79 \
  amp=bf16 \
  cos_lr=True \
  cache=ram \
  name=sai_yolov8s_optimal_1440x808

# Conservative fallback configuration  
python scripts/train_detector.py --config conservative
```

**See `docs/training-config-optimal.md` for detailed hardware optimization.**

#### Model Evaluation and Export
```bash
# Evaluate trained detector
python scripts/evaluate_detector.py --mode best

# Export for deployment
python scripts/export_detector.py --mode best --formats onnx torchscript

# Direct CLI commands:
yolo detect val model=runs/detect/sai_yolov8s_optimal_1440x808/weights/best.pt data=configs/yolo/pyro_fasdd.yaml
yolo export model=runs/detect/sai_yolov8s_optimal_1440x808/weights/best.pt format=onnx
```

## Key Configuration Files

### YOLO Dataset Configuration (`configs/yolo/pyro_fasdd.yaml`)
- Combined PyroSDIS + FASDD datasets (~129k images total)
- Single class detection: `nc: 1, names: ['smoke']`  
- Paths to training/validation images and labels
- Compatible with Ultralytics YOLO training pipeline

### Model Hyperparameters
- **YOLOv8s**: `imgsz=1440`, `batch=120` (VRAM optimized), BF16 mixed precision
- **Hardware optimized**: 2×A100 GPUs (36.1GB VRAM per GPU), 79 workers
- **Training time**: ~35-40 hours for high-resolution training

## Data Formats

- **YOLO Labels**: Normalized coordinates `<class> <x_center> <y_center> <width> <height>`
- **COCO Annotations**: JSON format with bounding boxes and category mappings for FASDD dataset
- **Single-class format**: All smoke/fire annotations mapped to class 0 ("smoke")

## Performance Targets (Detector)

- **Primary Metric**: mAP@0.5 optimization with emphasis on high recall
- **Focus**: Detecting small smoke objects with minimal false negatives  
- **Inference Speed**: Real-time processing capability for camera feeds
- **Confidence Threshold**: Low threshold (e.g., 0.25) to maximize recall for downstream verifier

## Hardware Requirements

- **Training**: 2×A100 GPUs, distributed training with DDP
- **Inference**: GPU acceleration recommended for real-time processing
- **Memory**: Mixed precision training to handle large batch sizes (batch=64)

## Licensing

This project is licensed under GNU GPL v3. Dataset usage follows respective licenses:
- PyroSDIS: Hugging Face dataset license
- FASDD: Creative Commons Attribution 4.0 (CC BY 4.0)