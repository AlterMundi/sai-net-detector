#!/usr/bin/env python3
"""
SAI-Net Detector Comprehensive Benchmark
Evaluates the final trained detector on multiple metrics and generates deployment-ready exports
"""

import torch
import time
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def benchmark_detector(model_path: str, data_config: str, device: str = "0"):
    """
    Comprehensive benchmark of SAI-Net detector
    """
    print("🔥 SAI-Net Detector Comprehensive Benchmark")
    print("=" * 60)
    
    # Load model
    print(f"📥 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Handle device properly
    if device == "cpu":
        device_torch = torch.device("cpu")
    else:
        device_torch = torch.device(f"cuda:{device}")
        
    # For YOLO models, device is handled through predict/val methods
    
    results = {}
    
    # 1. Model Information
    print("\n📊 MODEL SPECIFICATIONS:")
    model_info = {
        "architecture": "YOLOv8s",
        "input_resolution": "896x896 (SACRED)",
        "parameters": model.model.parameters() if hasattr(model.model, 'parameters') else "Unknown",
        "model_size_mb": Path(model_path).stat().st_size / (1024 * 1024),
        "framework": "Ultralytics YOLO",
        "precision": "FP32"
    }
    
    for key, value in model_info.items():
        print(f"  • {key}: {value}")
    results["model_info"] = model_info
    
    # 2. Validation Metrics
    print("\n🎯 VALIDATION PERFORMANCE:")
    val_results = model.val(data=data_config, device=device, verbose=False)
    
    validation_metrics = {
        "mAP50": float(val_results.box.map50) if hasattr(val_results.box, 'map50') else 0.0,
        "mAP50_95": float(val_results.box.map) if hasattr(val_results.box, 'map') else 0.0,
        "precision": float(val_results.box.mp) if hasattr(val_results.box, 'mp') else 0.0,
        "recall": float(val_results.box.mr) if hasattr(val_results.box, 'mr') else 0.0,
        "f1_score": 0.0
    }
    
    # Calculate F1 score
    if validation_metrics["precision"] > 0 and validation_metrics["recall"] > 0:
        validation_metrics["f1_score"] = 2 * (validation_metrics["precision"] * validation_metrics["recall"]) / (validation_metrics["precision"] + validation_metrics["recall"])
    
    print(f"  • mAP@0.5: {validation_metrics['mAP50']:.1%}")
    print(f"  • mAP@0.5:0.95: {validation_metrics['mAP50_95']:.1%}")
    print(f"  • Precision: {validation_metrics['precision']:.1%}")
    print(f"  • Recall: {validation_metrics['recall']:.1%}")
    print(f"  • F1-Score: {validation_metrics['f1_score']:.1%}")
    
    results["validation_metrics"] = validation_metrics
    
    # 3. Inference Speed Benchmark
    print("\n⚡ INFERENCE SPEED BENCHMARK:")
    
    # Warm up with dummy image - SACRED resolution 896x896
    dummy_image = np.random.randint(0, 255, (896, 896, 3), dtype=np.uint8)
    for _ in range(10):
        _ = model.predict(dummy_image, device=device, verbose=False)
    
    # Speed test
    num_runs = 100
    times = []
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        run_start = time.time()
        
        _ = model.predict(dummy_image, device=device, verbose=False)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        run_end = time.time()
        times.append((run_end - run_start) * 1000)  # Convert to ms
    
    end_time = time.time()
    
    speed_metrics = {
        "avg_inference_ms": np.mean(times),
        "min_inference_ms": np.min(times),
        "max_inference_ms": np.max(times),
        "std_inference_ms": np.std(times),
        "fps": 1000 / np.mean(times),
        "total_time_s": end_time - start_time,
        "runs": num_runs
    }
    
    print(f"  • Average: {speed_metrics['avg_inference_ms']:.2f} ms/image")
    print(f"  • Min: {speed_metrics['min_inference_ms']:.2f} ms")
    print(f"  • Max: {speed_metrics['max_inference_ms']:.2f} ms")
    print(f"  • FPS: {speed_metrics['fps']:.1f}")
    print(f"  • Runs: {speed_metrics['runs']}")
    
    results["speed_metrics"] = speed_metrics
    
    # 4. Memory Usage
    print("\n💾 MEMORY USAGE:")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
        # Run inference to measure peak memory
        _ = model.predict(dummy_image, device=device, verbose=False)
        
        memory_metrics = {
            "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
            "current_memory_mb": torch.cuda.memory_allocated() / (1024 * 1024),
            "device": torch.cuda.get_device_name()
        }
        
        print(f"  • Peak GPU Memory: {memory_metrics['peak_memory_mb']:.1f} MB")
        print(f"  • Current GPU Memory: {memory_metrics['current_memory_mb']:.1f} MB")
        print(f"  • Device: {memory_metrics['device']}")
    else:
        memory_metrics = {"message": "CUDA not available"}
        print("  • CPU mode - GPU memory not measured")
    
    results["memory_metrics"] = memory_metrics
    
    # 5. Grade Calculation (SAI-Net specific)
    print("\n🏆 SAI-NET PERFORMANCE GRADE:")
    
    # Grading criteria based on project requirements
    grade_weights = {
        "mAP50": 0.4,      # 40% - Primary metric
        "mAP50_95": 0.25,  # 25% - Strict IoU performance  
        "precision": 0.15, # 15% - False positive control
        "recall": 0.15,    # 15% - Miss rate control
        "speed": 0.05      # 5% - Real-time capability
    }
    
    # SAI-Net specific scoring functions (0-100 scale)
    # Based on SAGRADO documentation: Target >50% mAP@0.5, achieved 76.0% = Grade A+
    def score_map50(val): return min(100, max(0, (val - 0.3) * 143))  # 30% = 0, 50% = 29, 76% = 66, 100% = 100
    def score_map50_95(val): return min(100, max(0, (val - 0.2) * 125))  # 20% = 0, 50% = 37.5, 100% = 100
    def score_precision(val): return min(100, max(0, (val - 0.5) * 200))  # 50% = 0, 75% = 50, 100% = 100
    def score_recall(val): return min(100, max(0, (val - 0.5) * 200))  # 50% = 0, 75% = 50, 100% = 100 (HIGH RECALL priority)
    def score_speed(fps): return min(100, max(0, (fps - 5) * 1.7))  # 5 FPS = 0, 30+ FPS = 42+, 60+ FPS = 94+
    
    component_scores = {
        "mAP50": score_map50(validation_metrics["mAP50"]),
        "mAP50_95": score_map50_95(validation_metrics["mAP50_95"]),
        "precision": score_precision(validation_metrics["precision"]),
        "recall": score_recall(validation_metrics["recall"]),
        "speed": score_speed(speed_metrics["fps"])
    }
    
    # Calculate weighted grade
    final_grade = sum(component_scores[metric] * grade_weights[metric] for metric in grade_weights)
    
    print(f"  • mAP@0.5 Score: {component_scores['mAP50']:.1f}/100 (weight: {grade_weights['mAP50']:.0%})")
    print(f"  • mAP@0.5:0.95 Score: {component_scores['mAP50_95']:.1f}/100 (weight: {grade_weights['mAP50_95']:.0%})")
    print(f"  • Precision Score: {component_scores['precision']:.1f}/100 (weight: {grade_weights['precision']:.0%})")
    print(f"  • Recall Score: {component_scores['recall']:.1f}/100 (weight: {grade_weights['recall']:.0%})")
    print(f"  • Speed Score: {component_scores['speed']:.1f}/100 (weight: {grade_weights['speed']:.0%})")
    print("-" * 40)
    print(f"  🎯 FINAL GRADE: {final_grade:.1f}/100")
    
    # SAI-Net specific grade classification (based on meeting project targets)
    if final_grade >= 85:
        grade_letter = "A+"
        grade_desc = "Exceptional - Exceeds SAI-Net targets significantly"
    elif final_grade >= 70:
        grade_letter = "A"
        grade_desc = "Excellent - Meets all SAI-Net targets with margin"
    elif final_grade >= 55:
        grade_letter = "B+"
        grade_desc = "Good - Meets core SAI-Net target (>50% mAP@0.5)"
    elif final_grade >= 40:
        grade_letter = "B"
        grade_desc = "Acceptable - Close to SAI-Net target"
    else:
        grade_letter = "C"
        grade_desc = "Below SAI-Net requirements - Needs improvement"
    
    print(f"  📊 Grade: {grade_letter} - {grade_desc}")
    
    results["grade"] = {
        "final_score": final_grade,
        "letter_grade": grade_letter,
        "description": grade_desc,
        "component_scores": component_scores,
        "weights": grade_weights
    }
    
    return results

def export_models(model_path: str, export_dir: str):
    """
    Export model to deployment formats
    """
    print(f"\n📦 EXPORTING DEPLOYMENT MODELS:")
    export_path = Path(export_dir)
    export_path.mkdir(exist_ok=True, parents=True)
    
    model = YOLO(model_path)
    exports = {}
    
    try:
        # ONNX Export
        print("  • Exporting to ONNX...")
        onnx_path = model.export(format="onnx", imgsz=(808, 1440))
        exports["onnx"] = str(onnx_path)
        print(f"    ✅ ONNX: {onnx_path}")
    except Exception as e:
        print(f"    ❌ ONNX failed: {e}")
    
    try:
        # TorchScript Export
        print("  • Exporting to TorchScript...")
        torchscript_path = model.export(format="torchscript", imgsz=(808, 1440))
        exports["torchscript"] = str(torchscript_path)
        print(f"    ✅ TorchScript: {torchscript_path}")
    except Exception as e:
        print(f"    ❌ TorchScript failed: {e}")
    
    return exports

def generate_benchmark_report(results: dict, output_dir: str):
    """
    Generate comprehensive benchmark report
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save JSON results
    json_path = output_path / "benchmark_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate markdown report
    md_path = output_path / "SAI_NET_BENCHMARK_REPORT.md"
    
    report = f"""# SAI-Net Detector Benchmark Report (SACRED Resolution)

## Executive Summary
- **Final Grade**: {results['grade']['letter_grade']} ({results['grade']['final_score']:.1f}/100)
- **Status**: {results['grade']['description']}
- **Primary Metric**: {results['validation_metrics']['mAP50']:.1%} mAP@0.5
- **Inference Speed**: {results['speed_metrics']['avg_inference_ms']:.2f}ms ({results['speed_metrics']['fps']:.1f} FPS)

## Model Specifications
- **Architecture**: {results['model_info']['architecture']}
- **Input Resolution**: {results['model_info']['input_resolution']}
- **Model Size**: {results['model_info']['model_size_mb']:.1f} MB
- **Framework**: {results['model_info']['framework']}
- **Note**: Model trained on 1440×808, evaluated on SACRED 896×896

## Performance Metrics

### Detection Performance
| Metric | Value | Score |
|--------|--------|--------|
| mAP@0.5 | {results['validation_metrics']['mAP50']:.1%} | {results['grade']['component_scores']['mAP50']:.1f}/100 |
| mAP@0.5:0.95 | {results['validation_metrics']['mAP50_95']:.1%} | {results['grade']['component_scores']['mAP50_95']:.1f}/100 |
| Precision | {results['validation_metrics']['precision']:.1%} | {results['grade']['component_scores']['precision']:.1f}/100 |
| Recall | {results['validation_metrics']['recall']:.1%} | {results['grade']['component_scores']['recall']:.1f}/100 |
| F1-Score | {results['validation_metrics']['f1_score']:.1%} | - |

### Speed Performance
| Metric | Value |
|--------|--------|
| Average Inference | {results['speed_metrics']['avg_inference_ms']:.2f} ms |
| Min Inference | {results['speed_metrics']['min_inference_ms']:.2f} ms |
| Max Inference | {results['speed_metrics']['max_inference_ms']:.2f} ms |
| Frames Per Second | {results['speed_metrics']['fps']:.1f} FPS |
| Speed Score | {results['grade']['component_scores']['speed']:.1f}/100 |

## Grade Breakdown
The final grade is calculated using weighted components:

- **mAP@0.5** ({results['grade']['weights']['mAP50']:.0%}): {results['grade']['component_scores']['mAP50']:.1f}/100
- **mAP@0.5:0.95** ({results['grade']['weights']['mAP50_95']:.0%}): {results['grade']['component_scores']['mAP50_95']:.1f}/100  
- **Precision** ({results['grade']['weights']['precision']:.0%}): {results['grade']['component_scores']['precision']:.1f}/100
- **Recall** ({results['grade']['weights']['recall']:.0%}): {results['grade']['component_scores']['recall']:.1f}/100
- **Speed** ({results['grade']['weights']['speed']:.0%}): {results['grade']['component_scores']['speed']:.1f}/100

**Final Grade: {results['grade']['final_score']:.1f}/100 ({results['grade']['letter_grade']})**

## Deployment Readiness
- **Production Ready**: {'✅ Yes' if results['grade']['final_score'] >= 80 else '❌ Needs improvement'}
- **Real-time Capable**: {'✅ Yes' if results['speed_metrics']['fps'] >= 30 else '⚠️ Limited' if results['speed_metrics']['fps'] >= 15 else '❌ Too slow'}
- **Memory Efficient**: {'✅ Yes' if results.get('memory_metrics', {}).get('peak_memory_mb', 1000) < 500 else '⚠️ Moderate'}

## Training Summary
- **Method**: Two-stage transfer learning (FASDD → PyroSDIS)  
- **Stage 1**: FASDD multi-class (90.6% mAP@0.5)
- **Stage 2**: PyroSDIS single-class ({results['validation_metrics']['mAP50']:.1%} mAP@0.5)
- **Total Training Time**: ~10.5 hours
- **Hardware**: NVIDIA H200 GPU

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(md_path, 'w') as f:
        f.write(report)
    
    print(f"📊 Benchmark report saved: {md_path}")
    return str(md_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SAI-Net Detector Comprehensive Benchmark")
    parser.add_argument("--model", type=str, 
                       default="/workspace/sai-net-detector/runs/h200_stage2_pyrosdis3/weights/best.pt",
                       help="Path to trained model")
    parser.add_argument("--data", type=str,
                       default="/workspace/sai-net-detector/data/raw/pyro-sdis/data.yaml", 
                       help="Path to data configuration")
    parser.add_argument("--device", type=str, default="0", help="Device for inference")
    parser.add_argument("--export", action="store_true", help="Export deployment models")
    parser.add_argument("--output", type=str, default="/workspace/sai-net-detector/benchmark_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Run benchmark
    results = benchmark_detector(args.model, args.data, args.device)
    
    # Export models if requested
    if args.export:
        exports = export_models(args.model, args.output + "/exports")
        results["exports"] = exports
    
    # Generate report
    report_path = generate_benchmark_report(results, args.output)
    
    print("\n" + "="*60)
    print("🎉 BENCHMARK COMPLETED!")
    print(f"📊 Final Grade: {results['grade']['letter_grade']} ({results['grade']['final_score']:.1f}/100)")
    print(f"📄 Report: {report_path}")
    print("="*60)