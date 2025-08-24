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

#### Final Production YOLOv8 Training
```bash
# Production configuration with ForcedDDP (recommended)
python -m src.detector.train_forceddp

# Test 1-epoch configuration (for validation) - VERIFIED WORKING
python scripts/test_forceddp.py --epochs 1 --batch 60 --device 0,1 --name forceddp_verification_test

# Alternative: Direct Python module training
from src.detector.train_forceddp import train_optimal_forced_ddp
results = train_optimal_forced_ddp()
```

#### Validated Test Results (August 2025)
- **Test Configuration**: 1 epoch, batch=60, device=0,1, 1440×1440 resolution
- **Hardware**: 2×A100-40GB, ForcedDDP mode, 341GB RAM cache
- **Results**: mAP@0.5: **47.8%**, Precision: 50.1%, Recall: 48.7%
- **Performance**: 2.5ms inference, 1.85 it/s training speed
- **Status**: ✅ ForcedDDP working correctly, save_dir error patched
- **Training Time**: 16.7 minutes (1 epoch), estimated 42 hours for 150 epochs

#### Training Configuration (Production-Ready)

**Training Parameters:**
- **Epochs**: 150 (full training cycle)
- **Learning Rate**: 0.01 initial with cosine decay
- **Weight Decay**: 0.0005 
- **Loss Weights**: box=7.5, cls=0.5, dfl=1.5 (optimized for smoke detection)
- **Augmentation**: HSV (h=0.015, s=0.7, v=0.4), geometric transforms, mosaic/mixup

**Hardware Parameters (SAI-Net optimized for 2×A100, 500GB RAM):**
- **Resolution**: 1440×1440 (high-resolution for small smoke detection)
- **Batch Size**: 60 (proven stable in successful test, ~18GB per GPU)
- **Workers**: 8 (proven stable, prevents spawn explosion)
- **DDP Mode**: ForcedDDP with interactive fallback
- **Cache**: RAM (500GB limit), Mixed precision AMP
- **Scheduler**: Cosine LR with 5-epoch warmup

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
- **YOLOv8s**: `imgsz=1440`, `batch=60` (proven stable), BF16 mixed precision
- **Hardware optimized**: 2×A100-40GB GPUs (32.2GB VRAM per GPU), 8 workers
- **Training time**: ~42 hours for full 150-epoch training (tested: 16.7 min/epoch)

## Data Formats

- **YOLO Labels**: Normalized coordinates `<class> <x_center> <y_center> <width> <height>`
- **COCO Annotations**: JSON format with bounding boxes and category mappings for FASDD dataset
- **Single-class format**: All smoke/fire annotations mapped to class 0 ("smoke")

## Performance Targets (Detector)

- **Primary Metric**: mAP@0.5 optimization with emphasis on high recall
- **Current Performance**: mAP@0.5: 47.8% (verified in test), targeting >50% with full training
- **Focus**: Detecting small smoke objects with minimal false negatives  
- **Inference Speed**: 2.5ms per image (real-time capable for camera feeds)
- **Confidence Threshold**: Low threshold (e.g., 0.25) to maximize recall for downstream verifier

## Hardware Requirements

- **Training**: 2×A100-40GB GPUs, distributed training with ForcedDDP
- **Memory**: 500GB RAM limit (341GB used for cache), ~32GB VRAM per GPU
- **Inference**: GPU acceleration recommended for real-time processing (2.5ms/image)
- **Workers**: 8 workers optimal (prevents spawn explosion issues)
- **Mixed Precision**: AMP/BF16 enabled for memory efficiency

## Licensing

This project is licensed under GNU GPL v3. Dataset usage follows respective licenses:
- PyroSDIS: Hugging Face dataset license
- FASDD: Creative Commons Attribution 4.0 (CC BY 4.0)