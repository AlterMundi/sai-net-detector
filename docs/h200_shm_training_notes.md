# H200 + /dev/shm Training Configuration Notes

## Problem Resolution Log (August 2024)

### Issue: Slow Training Performance
**Problem**: Initial training attempts with 110 epochs were extremely slow (31.41s/batch vs expected 1.07s/batch)

**Root Cause**: Cache configuration conflict
- YOLO `cache=True` was creating additional RAM cache
- Images were already cached in `/dev/shm` partition
- Double caching caused memory conflicts and extreme slowdowns

**Solution**: Disable YOLO cache, use /dev/shm directly
```python
# WRONG (double cache)
'cache': True,           # YOLO creates additional RAM cache

# CORRECT (single cache)  
'cache': False,          # No YOLO cache - use /dev/shm directly
```

### Validated Configuration

**Hardware:**
- 1× NVIDIA H200 (140GB VRAM, 700W TDP)
- 258GB RAM system limit
- 125GB /dev/shm tmpfs partition

**Software Configuration:**
```python
config = {
    'batch': 128,           # Optimal for H200 (validated in 1-epoch test)
    'workers': 12,          # Optimized for RAM I/O
    'cache': False,         # CRITICAL: No YOLO cache (use /dev/shm directly)
    'imgsz': [1440, 808],   # High resolution for small object detection
    'project': '/workspace/sai-net-detector/runs',  # Outputs to repo
    'amp': True,            # Mixed precision
    'patience': 10,         # Early stopping
}
```

**Performance Metrics:**
- **Training Speed**: 1.07s/batch (stable after ~30 batches warmup)  
- **VRAM Usage**: 135GB stable (no memory leaks)
- **Dataset**: 37,413 clean images in /dev/shm (34GB used)
- **1-epoch test**: mAP@0.5=46.7% in 6.06 minutes
- **Full training ETA**: ~9.5 hours for 110 epochs

### Setup Commands

```bash
# 1. Setup /dev/shm cache (run once)
scripts/setup_shm_training.sh

# 2. Start training with validated configuration
python scripts/train_h200_shm.py --stage 1 --epochs 110

# 3. Monitor with background process
# Training saves to: /workspace/sai-net-detector/runs/h200_shm_stage1_fasddX/
```

### Key Learnings

1. **Don't double cache**: If images are in /dev/shm, set YOLO `cache=False`
2. **Warmup period**: First 20-30 batches are slow (initialization), then stable
3. **VRAM pattern**: 25G → 35G → 135G stable (normal progression)
4. **Dataset cleaning critical**: 0 background images after cleanup
5. **Configuration isolation**: /dev/shm script must disable YOLO cache independently

### Files Modified
- `scripts/train_h200_shm.py`: cache=False fix
- `scripts/train_h200.py`: batch=128 restored  
- `CLAUDE.md`: Updated with working configuration
- `docs/h200_shm_training_notes.md`: This troubleshooting guide

## Current Status (August 24, 2024)
- ✅ **110-epoch Stage 1 training running** 
- ✅ **Performance validated**: 1.07s/batch stable
- ✅ **Memory stable**: 135GB VRAM, no leaks
- ✅ **Configuration documented** for future use