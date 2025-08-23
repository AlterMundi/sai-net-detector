# SAI-Net Detector

YOLOv8-based smoke detection for wildfire early warning systems.

## Quick Start

```bash
# Convert FASDD dataset
python scripts/convert_fasdd_to_yolo.py --src data/raw --dst data/yolo --map-classes smoke

# Train detector (optimal config for 2×A100)
yolo detect train data=configs/yolo/pyro_fasdd.yaml model=yolov8s.pt batch=152 workers=80 cache=ram

# Evaluate model
yolo detect val model=runs/detect/train/weights/best.pt data=configs/yolo/pyro_fasdd.yaml
```

## Dataset

- **PyroSDIS**: 33,637 smoke images (YOLO format)
- **FASDD**: 95,314 fire/smoke images (converted from COCO)
- **Total**: ~129k training images

## Hardware Requirements

- **Minimum**: 1×GPU with 8GB+ VRAM
- **Optimal**: 2×A100 GPUs, 500GB RAM
- **Training time**: 35-45 hours (optimal config)

## Documentation

- `CLAUDE.md` - Complete development guide
- `docs/training-config-optimal.md` - Hardware-optimized training
- `configs/yolo/pyro_fasdd.yaml` - Dataset configuration

## License

GNU GPL v3