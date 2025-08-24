#!/usr/bin/env python3
"""
SAI-Net Detector Training Script
Optimized YOLOv8 training for wildfire smoke detection
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detector.train import train_detector, train_optimal, train_conservative

def main():
    parser = argparse.ArgumentParser(description="Train SAI-Net Detector")
    
    # Configuration presets
    parser.add_argument(
        "--config", 
        choices=["optimal", "conservative", "custom"], 
        default="optimal",
        help="Training configuration preset"
    )
    
    # Custom configuration options
    parser.add_argument("--data", default="configs/yolo/pyro_fasdd.yaml", help="Dataset YAML")
    parser.add_argument("--model", default="yolov8s.pt", help="Model architecture")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=120, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=1440, help="Image size")
    parser.add_argument("--device", default="0,1", help="GPU devices")
    parser.add_argument("--workers", type=int, default=79, help="Data loading workers")
    parser.add_argument("--name", help="Experiment name")
    parser.add_argument("--cache", default="ram", help="Cache mode (ram/disk)")
    
    # Learning rate and optimization
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor")
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay")
    
    # Mixed precision and hardware optimization
    parser.add_argument("--amp", action="store_true", default=True, help="Mixed precision training")
    parser.add_argument("--cos-lr", action="store_true", default=True, help="Cosine LR scheduler")
    parser.add_argument("--single-cls", action="store_true", default=True, help="Single class mode")
    
    args = parser.parse_args()
    
    if args.config == "optimal":
        print("Starting optimal training configuration...")
        results = train_optimal()
        
    elif args.config == "conservative":
        print("Starting conservative training configuration...")
        results = train_conservative()
        
    elif args.config == "custom":
        print("Starting custom training configuration...")
        results = train_detector(
            data_yaml=args.data,
            model=args.model,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            name=args.name,
            cache=args.cache,
            lr0=args.lr0,
            lrf=args.lrf,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            amp=args.amp,
            cos_lr=args.cos_lr,
            single_cls=args.single_cls
        )
    
    # Print results
    if results['success']:
        print(f"\n✓ Training completed successfully!")
        print(f"Results saved to: {results['save_dir']}")
        if results.get('best_fitness'):
            print(f"Best fitness: {results['best_fitness']:.4f}")
    else:
        print(f"\n✗ Training failed: {results['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()