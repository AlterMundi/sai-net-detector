# SAI-Net Detector Benchmark Report

## Executive Summary
- **Final Grade**: B (54.2/100)
- **Status**: Acceptable - Close to SAI-Net target
- **Primary Metric**: 75.9% mAP@0.5
- **Inference Speed**: 4.41ms (226.8 FPS)

## Model Specifications
- **Architecture**: YOLOv8s
- **Input Resolution**: 1440x808
- **Model Size**: 21.5 MB
- **Framework**: Ultralytics YOLO

## Performance Metrics

### Detection Performance
| Metric | Value | Score |
|--------|--------|--------|
| mAP@0.5 | 75.9% | 65.7/100 |
| mAP@0.5:0.95 | 50.2% | 37.7/100 |
| Precision | 70.7% | 41.4/100 |
| Recall | 74.4% | 48.7/100 |
| F1-Score | 72.5% | - |

### Speed Performance
| Metric | Value |
|--------|--------|
| Average Inference | 4.41 ms |
| Min Inference | 4.38 ms |
| Max Inference | 4.46 ms |
| Frames Per Second | 226.8 FPS |
| Speed Score | 100.0/100 |

## Grade Breakdown
The final grade is calculated using weighted components:

- **mAP@0.5** (40%): 65.7/100
- **mAP@0.5:0.95** (25%): 37.7/100  
- **Precision** (15%): 41.4/100
- **Recall** (15%): 48.7/100
- **Speed** (5%): 100.0/100

**Final Grade: 54.2/100 (B)**

## Deployment Readiness
- **Production Ready**: ❌ Needs improvement
- **Real-time Capable**: ✅ Yes
- **Memory Efficient**: ✅ Yes

## Training Summary
- **Method**: Two-stage transfer learning (FASDD → PyroSDIS)  
- **Stage 1**: FASDD multi-class (90.6% mAP@0.5)
- **Stage 2**: PyroSDIS single-class (75.9% mAP@0.5)
- **Total Training Time**: ~10.5 hours
- **Hardware**: NVIDIA H200 GPU

Generated on: 2025-08-24 21:27:26
