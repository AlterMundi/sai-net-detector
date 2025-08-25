# SAI-Net Detector Test Suite

Comprehensive testing framework for evaluating the 4 key SAI-Net detector models against multiple datasets.

## 🎯 Test Structure

```
tests/
├── benchmark/           # Performance benchmarking
│   ├── benchmark_runner.py    # Comprehensive benchmark suite
│   ├── quick_benchmark.py     # Fast model testing
│   ├── models/         # Model-specific tests
│   └── results/        # Benchmark outputs
├── data/               # Test data configurations
│   └── configs/
│       └── test_datasets.yaml # Dataset and model configs
├── unit/               # Unit tests
├── integration/        # Integration tests
└── README.md          # This file
```

## 🏆 Models Under Test

1. **Production Final** (`runs/h200_stage2_pyrosdis3/weights/best.pt`)
   - 76.0% mAP@0.5, 22MB, Production-ready

2. **Stage 1 Baseline** (`runs/h200_stage1_fasdd7/weights/best.pt`) 
   - 90.6% mAP@0.5, 43MB, Multi-class baseline

3. **Pre-trained YOLOv8s** (`yolov8s.pt`)
   - 22MB, Ultralytics baseline reference

4. **YOLOv11n** (`yolo11n.pt`)
   - 5.4MB, Latest architecture efficiency test

## 📊 Test Datasets

**Location**: `/root/sai-benchmark.old/RNA/data/raw/`

- **FASDD**: 95k+ images, fire/smoke detection (COCO format)
- **PyroNear**: Real wildfire smoke dataset (YOLO format)  
- **D-Fire**: Fire detection dataset
- **NEMO**: Fire detection dataset
- **FigLib**: Fire detection dataset

## 🚀 Quick Start

### Fast Benchmark (100 images per model)
```bash
# Quick performance test on all 4 models
python tests/benchmark/quick_benchmark.py

# Expected output: FPS, memory usage, detection counts
```

### Comprehensive Benchmark
```bash
# Full benchmark on all models and datasets
python tests/benchmark/benchmark_runner.py

# Specific models only
python tests/benchmark/benchmark_runner.py --models production_final stage1_baseline

# Specific datasets only  
python tests/benchmark/benchmark_runner.py --datasets fasdd pyronear

# Custom configuration
python tests/benchmark/benchmark_runner.py --config tests/data/configs/test_datasets.yaml
```

## 📈 Expected Performance

Based on training results:

| Model | mAP@0.5 | Size | Expected FPS* | Memory |
|-------|---------|------|---------------|---------|
| Production Final | 76.0% | 22MB | ~230 FPS | ~150MB |
| Stage 1 Baseline | 90.6% | 43MB | ~180 FPS | ~200MB |
| Pre-trained YOLOv8s | Unknown | 22MB | ~230 FPS | ~150MB |
| YOLOv11n | Unknown | 5.4MB | ~300 FPS | ~80MB |

*FPS estimates based on 4.35ms inference time at 896×896 resolution

## 🔧 Configuration

Edit `tests/data/configs/test_datasets.yaml` to modify:

- Dataset paths and formats
- Model configurations  
- Test parameters (batch size, thresholds, etc.)
- Output formats and locations

## 📊 Results Format

### JSON Output
```json
{
  "timestamp": "20250825_143022",
  "results": [
    {
      "model_name": "Production Final",
      "dataset_name": "fasdd", 
      "total_images": 95314,
      "inference_time": 415.2,
      "fps": 229.6,
      "memory_usage": 157.4,
      "detections_count": 45823
    }
  ]
}
```

### CSV Summary
```csv
model_name,dataset_name,total_images,inference_time,fps,memory_mb,detections,errors
Production Final,fasdd,95314,415.200,229.60,157.40,45823,0
Stage 1 Baseline,fasdd,95314,530.100,179.90,198.20,52341,0
```

## 🎯 Benchmark Goals

1. **Performance Validation**: Confirm 76.0% mAP@0.5 production target
2. **Speed Analysis**: Measure real-world FPS on various datasets
3. **Memory Profiling**: Track GPU/RAM usage patterns
4. **Robustness Testing**: Error handling across diverse datasets
5. **Comparative Analysis**: Rank models by speed/accuracy trade-offs

## 🛠️ Dependencies

- PyTorch >= 1.9
- Ultralytics YOLO >= 8.0
- OpenCV >= 4.5
- NumPy, tqdm, PyYAML

## 📝 Notes

- Tests use 896×896 SACRED resolution (training resolution)
- Batch processing for memory efficiency
- GPU acceleration when available
- Comprehensive error logging and recovery
- Results timestamped for reproducibility