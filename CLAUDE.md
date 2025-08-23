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
- **PyroSDIS**: Primary training dataset from Hugging Face (`pyronear/pyro-sdis`) with 33,600+ images containing smoke bounding boxes in YOLO format
- **FASDD**: Additional dataset with 100k+ images of fire/smoke from various sources (terrestrial cameras, drones)

### Model Architecture (Detector Only)
- **YOLOv8-s/m**: Anchor-free architecture with C2f blocks for rapid smoke/fire localization
- **Single-class detection**: Optimized for "smoke" class detection with high recall
- **Input resolution**: 960x960 pixels for small object detection
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
# Download PyroSDIS from Hugging Face
python scripts/export_pyrosdis_to_yolo.py \
  --hf_repo pyronear/pyro-sdis \
  --out data/yolo \
  --split train val \
  --single-cls 1

# Convert FASDD to YOLO format
python scripts/convert_fasdd_to_yolo.py \
  --src data/raw/fasdd \
  --dst data/yolo \
  --split-ratios 0.9 0.1 \
  --map-classes smoke

# Extract FASDD from archive
unzip fasdd-cv-coco.zip -d data/raw/
```

### Training

#### YOLOv8 Detector Training
```bash
# Train smoke/fire detector on PyroSDIS + FASDD
yolo detect train \
  data=configs/yolo/pyro_fasdd.yaml \
  model=yolov8s.pt \
  imgsz=960 \
  epochs=150 \
  batch=64 \
  device=0,1 \
  workers=16 \
  amp=bf16 \
  cos_lr=True \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  degrees=5 translate=0.1 scale=0.5 shear=2.0 \
  mosaic=1.0 mixup=0.1 copy_paste=0.0 \
  name=sai_yolov8s_pyrofasdd
```

#### Model Evaluation and Export
```bash
# Evaluate trained detector
yolo detect val model=runs/detect/sai_yolov8s_pyrofasdd/weights/best.pt data=configs/yolo/pyro_fasdd.yaml

# Export to ONNX for deployment
yolo export model=runs/detect/sai_yolov8s_pyrofasdd/weights/best.pt format=onnx
```

## Key Configuration Files

### YOLO Dataset Configuration (`pyro-sdis/data.yaml`)
- Single class detection: `nc: 1, names: ['smoke']`
- Paths to training/validation images and labels
- Compatible with Ultralytics YOLO training pipeline

### Model Hyperparameters
- **YOLOv8**: `imgsz=960`, `batch=64`, `lr0=0.01`, mosaic+mixup augmentation
- **Augmentation**: HSV jittering, rotation, translation, scaling, shearing
- **Optimization**: Cosine LR scheduling, mixed precision training (BF16)

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