#!/usr/bin/env python3
"""
SAI-Net Detector Export Script
Model export utilities for deployment (ONNX, TensorRT, etc.)
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detector.export import export_detector, export_for_deployment, export_best_model, validate_exported_model

def main():
    parser = argparse.ArgumentParser(description="Export SAI-Net Detector")
    
    # Export mode
    parser.add_argument(
        "--mode", 
        choices=["single", "deployment", "best"], 
        default="best",
        help="Export mode"
    )
    
    # Model and format
    parser.add_argument("--model", help="Path to model weights (.pt file)")
    parser.add_argument("--experiment", default="sai_yolov8s_optimal_1440x808", help="Experiment name (for best mode)")
    parser.add_argument("--format", default="onnx", 
                       choices=["torchscript", "onnx", "openvino", "engine", "coreml", 
                               "tflite", "edgetpu", "tfjs", "paddle", "ncnn"],
                       help="Export format")
    parser.add_argument("--formats", nargs="+", default=["onnx", "torchscript"],
                       help="Multiple formats for deployment mode")
    
    # Export parameters
    parser.add_argument("--imgsz", type=str, default="1440,808", help="Image size (width,height)")
    parser.add_argument("--device", default="cpu", help="Export device")
    parser.add_argument("--half", action="store_true", help="Use FP16 precision")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic input shapes (ONNX)")
    parser.add_argument("--simplify", action="store_true", default=True, help="Simplify ONNX model")
    parser.add_argument("--opset", type=int, help="ONNX opset version")
    parser.add_argument("--workspace", type=float, default=4.0, help="TensorRT workspace (GB)")
    parser.add_argument("--nms", action="store_true", help="Add NMS module")
    
    # Validation
    parser.add_argument("--validate", action="store_true", help="Validate exported model")
    parser.add_argument("--test-image", help="Test image for validation")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold for validation")
    
    # Output
    parser.add_argument("--output-dir", help="Output directory for exported models")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        if not args.model:
            print("Error: --model required for single export mode")
            sys.exit(1)
            
        print(f"Exporting model: {args.model} to {args.format.upper()}")
        
        # Parse imgsz string to list
        imgsz = [int(x) for x in args.imgsz.split(',')] if ',' in args.imgsz else int(args.imgsz)
        
        results = export_detector(
            model_path=args.model,
            format=args.format,
            imgsz=imgsz,
            device=args.device,
            half=args.half,
            dynamic=args.dynamic,
            simplify=args.simplify,
            opset=args.opset,
            workspace=args.workspace,
            nms=args.nms
        )
        
        if results['success']:
            print(f"\n✓ Export completed successfully!")
            print(f"Exported model: {results['exported_path']}")
            print(f"File size: {results['file_size_mb']} MB")
            print(f"Format: {results['format']}")
            
            # Validation
            if args.validate:
                print(f"\nValidating exported model...")
                val_results = validate_exported_model(
                    exported_path=results['exported_path'],
                    test_image=args.test_image,
                    confidence=args.confidence
                )
                
                if val_results['success']:
                    print(f"✓ Model validation successful")
                    if val_results.get('detections') is not None:
                        print(f"Test detections: {val_results['detections']}")
                else:
                    print(f"✗ Model validation failed: {val_results['error']}")
                    
        else:
            print(f"\n✗ Export failed: {results['error']}")
            sys.exit(1)
    
    elif args.mode == "deployment":
        model_path = args.model
        if not model_path:
            # Try to find best model from experiment
            runs_dir = Path("runs/detect")
            experiment_dir = runs_dir / args.experiment
            model_path = experiment_dir / "weights" / "best.pt"
            
            if not model_path.exists():
                print(f"Error: Model not found. Provide --model or ensure {args.experiment} exists")
                sys.exit(1)
        
        print(f"Exporting model for deployment: {model_path}")
        print(f"Output formats: {', '.join(args.formats).upper()}")
        
        results = export_for_deployment(
            model_path=model_path,
            output_dir=args.output_dir
        )
        
        print(f"\n=== Deployment Export Results ===")
        for config_name, result in results.items():
            if result['success']:
                print(f"✓ {config_name}: {result['exported_path']} ({result['file_size_mb']} MB)")
            else:
                print(f"✗ {config_name}: {result['error']}")
    
    elif args.mode == "best":
        print(f"Exporting best model from experiment: {args.experiment}")
        print(f"Formats: {', '.join(args.formats).upper()}")
        
        results = export_best_model(
            experiment_name=args.experiment,
            formats=args.formats
        )
        
        print(f"\n=== Export Results ===")
        for fmt, result in results.items():
            if result['success']:
                print(f"✓ {fmt.upper()}: {result['exported_path']} ({result['file_size_mb']} MB)")
                
                # Validation
                if args.validate:
                    print(f"  Validating {fmt.upper()}...")
                    val_results = validate_exported_model(
                        exported_path=result['exported_path'],
                        test_image=args.test_image,
                        confidence=args.confidence
                    )
                    
                    if val_results['success']:
                        print(f"  ✓ Validation successful")
                        if val_results.get('detections') is not None:
                            print(f"  Test detections: {val_results['detections']}")
                    else:
                        print(f"  ✗ Validation failed: {val_results['error']}")
            else:
                print(f"✗ {fmt.upper()}: {result['error']}")

if __name__ == "__main__":
    main()