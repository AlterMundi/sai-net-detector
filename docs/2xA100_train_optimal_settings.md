# 2×A100 Optimal Training Configuration

## Hardware Specifications
- **GPUs**: 2× NVIDIA A100-40GB (80GB total VRAM)
- **RAM**: 500GB system memory
- **CPU**: High-performance multi-core processor
- **Storage**: SSD with sufficient space for datasets

## Training Configuration

### Model Settings
```yaml
model: yolov8s.pt        # YOLOv8 small (11M params) for efficiency
device: 0,1              # Both GPUs for DDP training
imgsz: [1440, 808]       # Fixed resolution for SAI-Net detector
```

### Batch and Memory Optimization
```yaml
batch: 60                # Total batch size (30 per GPU)
workers: 8               # Optimal for preventing spawn explosion
cache: ram               # Utilize 341GB RAM cache (tested)
pin_memory: true         # Faster data transfer to GPU
persistent_workers: true # Keep workers alive between epochs
```

### Training Schedule
```yaml
# Stage 1: FASDD Pre-training
epochs: 110              # 100-110 epochs for FASDD
patience: 10             # Early stopping patience
single_cls: false        # Multi-class (fire + smoke)

# Stage 2: PyroSDIS Fine-tuning  
epochs: 60               # 40-60 epochs for PyroSDIS
patience: 10             # Early stopping patience
single_cls: true         # Single-class (smoke only)
lr0: 0.001              # 10× reduced learning rate
```

### Optimizer Configuration
```yaml
optimizer: SGD           # Memory efficient
lr0: 0.01               # Initial learning rate (Stage 1)
lrf: 0.01               # Final learning rate
momentum: 0.937         # SGD momentum
weight_decay: 0.0005    # L2 regularization
warmup_epochs: 5        # Gradual warmup
cos_lr: true            # Cosine learning rate decay
```

### Data Augmentation
```yaml
# Stage 1 (FASDD) - Aggressive augmentation
mosaic: 1.0             # Full mosaic augmentation
mixup: 0.15             # 15% mixup probability
copy_paste: 0.1         # 10% copy-paste augmentation
hsv_h: 0.015            # Hue variation
hsv_s: 0.7              # Saturation variation
hsv_v: 0.4              # Value variation

# Stage 2 (PyroSDIS) - Moderate augmentation
mosaic: 0.5             # Reduced mosaic
mixup: 0.1              # Reduced mixup
copy_paste: 0.0         # No copy-paste for fine-tuning
```

### Loss Weights
```yaml
box: 7.5                # Bounding box loss weight
cls: 0.5                # Classification loss weight
dfl: 1.5                # Distribution focal loss weight
```

### Performance Settings
```yaml
amp: true               # Mixed precision training (FP16)
ddp: true               # Distributed Data Parallel
sync_bn: true           # Synchronized BatchNorm
noval: false            # Run validation
save_period: 5          # Save checkpoint every 5 epochs
```

## Memory Usage Profile

| Component | Memory Usage | Notes |
|-----------|-------------|-------|
| **Per GPU VRAM** | ~32.2GB | Verified in testing |
| **Total VRAM** | ~64.4GB | Both GPUs combined |
| **RAM Cache** | 341GB | Full dataset cached |
| **Workers Memory** | ~50GB | 8 workers with prefetch |
| **Peak Usage** | ~400GB | During cache building |

## Performance Metrics

### Expected Training Time
- **Stage 1 (FASDD)**: ~30-35 hours (110 epochs)
- **Stage 2 (PyroSDIS)**: ~8-12 hours (60 epochs)
- **Total Time**: ~40-47 hours

### Verified Performance (Test Run)
- **1 Epoch Time**: 16.7 minutes
- **Training Speed**: 1.85 it/s
- **Inference Speed**: 2.5ms per image
- **mAP@0.5**: 47.8% (1 epoch baseline)

## Training Commands

### Stage 1: FASDD Pre-training
```bash
python scripts/train_two_stage.py --stage 1 --epochs 110
```

### Stage 2: PyroSDIS Fine-tuning
```bash
python scripts/train_two_stage.py --stage 2 --epochs 60
```

### Full Two-Stage Workflow
```bash
python scripts/train_two_stage.py --full-workflow
```

## Critical Settings for Stability

1. **Workers = 8**: Higher values cause spawn explosion
2. **Batch = 60**: Optimal for 40GB VRAM per GPU
3. **ForcedDDP Mode**: Ensures proper multi-GPU training
4. **RAM Cache**: Critical for performance (341GB tested)
5. **Resolution**: [1440, 808] rectangular (not square)

## Troubleshooting

### Common Issues
- **OOM Errors**: Reduce batch size to 48-56
- **Spawn Explosion**: Keep workers at 8 or less
- **Cache Building**: Allow 30-45 minutes initial time
- **DDP Errors**: Use ForcedDDP mode in training script

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Notes

- This configuration is validated on 2×A100-40GB with 500GB RAM
- Achieves optimal balance between speed and stability
- Two-stage training improves domain adaptation
- Early stopping prevents overfitting
- Mixed precision (AMP) provides 30% speedup