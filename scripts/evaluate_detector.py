#!/usr/bin/env python3
"""
SAI-Net Detector Evaluation Script
Model validation and performance analysis for wildfire smoke detection
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detector.evaluate import evaluate_detector, evaluate_best_model, benchmark_detector

def main():
    parser = argparse.ArgumentParser(description="Evaluate SAI-Net Detector")
    
    # Evaluation mode
    parser.add_argument(
        "--mode", 
        choices=["model", "best", "benchmark"], 
        default="best",
        help="Evaluation mode"
    )
    
    # Model and data
    parser.add_argument("--model", help="Path to model weights (.pt file)")
    parser.add_argument("--data", default="configs/yolo/pyro_fasdd.yaml", help="Dataset YAML")
    parser.add_argument("--experiment", default="sai_yolov8s_optimal_1440x808", help="Experiment name (for best mode)")
    
    # Evaluation parameters
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--imgsz", type=int, default=1440, help="Image size")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default="0", help="GPU device")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for NMS")
    
    # Output options
    parser.add_argument("--plots", action="store_true", default=True, help="Generate plots")
    parser.add_argument("--save-json", action="store_true", default=True, help="Save COCO JSON results")
    parser.add_argument("--name", help="Evaluation experiment name")
    
    # Benchmark options
    parser.add_argument("--conf-thresholds", nargs="+", type=float, 
                       default=[0.1, 0.25, 0.5, 0.75, 0.9], 
                       help="Confidence thresholds for benchmark")
    
    args = parser.parse_args()
    
    if args.mode == "model":
        if not args.model:
            print("Error: --model required for model evaluation mode")
            sys.exit(1)
            
        print(f"Evaluating model: {args.model}")
        results = evaluate_detector(
            model_path=args.model,
            data_yaml=args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            plots=args.plots,
            save_json=args.save_json,
            name=args.name
        )
        
    elif args.mode == "best":
        print(f"Evaluating best model from experiment: {args.experiment}")
        results = evaluate_best_model(
            experiment_name=args.experiment
        )
        
    elif args.mode == "benchmark":
        model_path = args.model
        if not model_path:
            # Try to find best model from experiment
            runs_dir = Path("/dev/shm/rrn/sai-net-detector/runs/detect")
            experiment_dir = runs_dir / args.experiment
            model_path = experiment_dir / "weights" / "best.pt"
            
            if not model_path.exists():
                print(f"Error: Model not found. Provide --model or ensure {args.experiment} exists")
                sys.exit(1)
        
        print(f"Benchmarking model: {model_path}")
        print(f"Confidence thresholds: {args.conf_thresholds}")
        
        results = benchmark_detector(
            model_path=model_path,
            data_yaml=args.data,
            conf_thresholds=args.conf_thresholds
        )
    
    # Print results
    if args.mode == "benchmark":
        print(f"\n=== Benchmark Results ===")
        for conf, metrics in results['benchmark_results'].items():
            if metrics:
                print(f"Confidence {conf.replace('conf_', '')}: "
                      f"mAP@0.5={metrics['mAP50']:.4f}, "
                      f"Precision={metrics['precision']:.4f}, "
                      f"Recall={metrics['recall']:.4f}, "
                      f"F1={metrics['f1']:.4f}")
            else:
                print(f"Confidence {conf.replace('conf_', '')}: Failed")
    
    else:
        if results['success']:
            print(f"\n✓ Evaluation completed successfully!")
            if results.get('save_dir'):
                print(f"Results saved to: {results['save_dir']}")
            
            metrics = results.get('metrics', {})
            if metrics:
                print(f"\n=== Performance Metrics ===")
                print(f"mAP@0.5: {metrics.get('mAP50', 0):.4f}")
                print(f"mAP@0.5:0.95: {metrics.get('mAP50_95', 0):.4f}")
                print(f"Precision: {metrics.get('precision', 0):.4f}")
                print(f"Recall: {metrics.get('recall', 0):.4f}")
                print(f"F1: {metrics.get('f1', 0):.4f}")
        else:
            print(f"\n✗ Evaluation failed: {results['error']}")
            sys.exit(1)

if __name__ == "__main__":
    main()