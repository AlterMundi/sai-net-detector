# 1×H200 (258GB RAM) Optimal Training Configuration

## Hardware Specifications
- **GPU**: 1× NVIDIA H200 (140GB VRAM, 700W TDP)
- **RAM**: 258GB system memory (hard limit)
- **CPU**: 2× Intel Xeon Platinum 8468V (192 threads total)
- **Storage**: 200GB available space
- **CUDA**: 12.8 with PyTorch 2.7.1

## Training Configuration

### Model Settings
```yaml
model: yolov8s.pt        # YOLOv8 small for efficiency
device: 0                # Single GPU (no DDP needed)
imgsz: [1440, 808]       # Fixed resolution for SAI-Net detector
```

### Batch and Memory Optimization (CRITICAL)
```yaml
batch: 128               # Maximum throughput (10GB VRAM)
workers: 8               # REDUCED from 32 to prevent OOM
cache: disk              # CRITICAL: NO RAM cache (would need 419GB)
pin_memory: false        # Avoid additional memory allocation
persistent_workers: true # Keep workers alive for efficiency
prefetch_factor: 2       # REDUCED from 4 to control memory
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
optimizer: SGD           # Memory efficient vs AdamW
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
half: true              # Use FP16 inference
ddp: false              # Single GPU, no DDP
noval: false            # Run validation
save_period: 5          # Save checkpoint every 5 epochs
```

## Memory Usage Analysis

### ⚠️ Critical Memory Constraints
| Component | Memory Required | Status |
|-----------|----------------|--------|
| **RAW Cache (All Images)** | 419.2 GB | ❌ IMPOSSIBLE |
| **Compressed Cache** | 21.0 GB | ⚠️ RISKY with workers |
| **No Cache (Disk I/O)** | 0 GB | ✅ SAFE |

### Safe Memory Profile
| Component | Memory Usage | Notes |
|-----------|-------------|-------|
| **OS + Base Processes** | 50 GB | System overhead |
| **DataLoader (8 workers)** | 13.3 GB | Prefetch buffers |
| **Training Batch (GPU)** | 2.5 GB | In VRAM, not RAM |
| **Validation Peaks** | +6.7 GB | During eval phase |
| **No RAM Cache** | 0 GB | Using disk I/O |
| **TOTAL PEAK** | **72.5 GB** | Safe margin: 60% |

### Memory Safety Limits
- **Total RAM**: 258 GB
- **Available RAM**: 158 GB (after OS)
- **Safe Limit (85%)**: 119.3 GB
- **Peak Usage**: 72.5 GB ✅
- **Safety Margin**: 46.8 GB (40%)

## Performance Metrics

### GPU Utilization
- **Batch 128**: 10.0 GB VRAM (7.1% of 140GB)
- **Training Speed**: ~2x faster than 2×A100
- **Inference Speed**: <2ms per image

### Expected Training Time
- **Stage 1 (FASDD)**: 12-18 hours
- **Stage 2 (PyroSDIS)**: 4-8 hours
- **Total Time**: ~20-25 hours
- **Note**: +50% time vs RAM cache due to disk I/O

## Training Commands

### Stage 1: FASDD Pre-training
```bash
python scripts/train_two_stage.py --stage 1 --epochs 110 --batch 128 --workers 8 --cache disk
```

### Stage 2: PyroSDIS Fine-tuning
```bash
python scripts/train_two_stage.py --stage 2 --epochs 60 --batch 128 --workers 8 --cache disk
```

### Full Two-Stage Workflow
```bash
python scripts/train_two_stage.py --full-workflow --batch 128 --workers 8 --cache disk
```

## Critical Configuration Differences

### ⚠️ MUST FOLLOW - Memory Safety Rules
1. **NO RAM CACHE**: Would require 419GB (impossible)
2. **Workers = 8 MAX**: Higher causes memory overflow
3. **pin_memory = false**: Saves memory
4. **prefetch_factor = 2**: Reduced from default 4
5. **cache = disk**: Essential for 258GB RAM limit

### Why These Settings?
- **32 workers → 8 workers**: Reduces DataLoader memory from 53.3GB to 13.3GB
- **RAM cache → Disk cache**: Saves 419GB RAM requirement
- **Prefetch 4 → 2**: Reduces worker queue memory by 50%
- **Pin memory off**: Avoids additional RAM allocation

## Optimization Tips

### For Faster Training (if memory allows)
```yaml
# Try these incrementally if RAM usage stays <100GB
workers: 12             # +6.7GB RAM
prefetch_factor: 3      # +6.7GB RAM  
cache: ram              # +21GB RAM (compressed only)
```

### For Maximum Stability
```yaml
# Ultra-safe configuration
batch: 96               # Reduce if needed
workers: 6              # Minimal workers
prefetch_factor: 1      # Minimal prefetch
```

## Monitoring Commands

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor RAM usage
watch -n 1 free -h

# Monitor training process memory
htop -p $(pgrep python)
```

## Troubleshooting

### Common Issues
- **OOM on RAM**: Reduce workers to 6 or 4
- **Slow I/O**: Ensure SSD, not HDD for datasets
- **GPU underutilized**: Increase batch if VRAM allows
- **Validation OOM**: Reduce validation batch size

### Emergency Settings (if OOM occurs)
```yaml
batch: 64               # Half batch size
workers: 4              # Minimal workers
cache: false            # No cache at all
pin_memory: false       # No pinned memory
prefetch_factor: 1      # Minimal prefetch
```

## Notes

- H200 provides 4.4× more VRAM than A100-40GB
- Single GPU eliminates DDP complexity
- Disk I/O adds ~50% training time vs RAM cache
- 192 CPU threads help offset I/O bottleneck
- Configuration prioritizes stability over speed