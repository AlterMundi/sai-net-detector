#!/usr/bin/env python3
"""
SAI-Net H200 Fast Prototyping Training Script

For rapid hyperparameter testing and pattern identification.
Results can be extrapolated to full resolution training.

Usage:
  python scripts/train_h200_fast.py --test-config lr_sweep
  python scripts/train_h200_fast.py --test-config batch_test  
  python scripts/train_h200_fast.py --test-config augment_test
"""

import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

def get_fast_base_config():
    """Fast testing base configuration - 4x faster than production."""
    return {
        # FAST SETTINGS (4x speedup)
        'device': 0,                 # Single H200
        'imgsz': 512,                # 4x faster than 896
        'epochs': 25,                # 1/4 of production for pattern ID
        'patience': 6,               # Proportional to epochs
        
        # OPTIMIZED MEMORY/SPEED
        'batch': 256,                # 2x batch due to lower resolution
        'workers': 12,               # Keep optimized workers
        'cache': 'disk',             # Keep cache strategy
        
        # CORE TRAINING (keep identical to production)
        'single_cls': False,         # Multi-class (fire + smoke)
        'amp': True,                 # Mixed precision
        'save_period': 5,            # Save every 5 epochs
        
        # OPTIMIZER (identical to production)
        'optimizer': 'SGD',
        'lr0': 0.01,                 # TEST: baseline
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,          # Proportional to epochs
        'cos_lr': True,
        
        # LOSS WEIGHTS (identical)
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # AUGMENTATION (identical)
        'mosaic': 1.0,
        'mixup': 0.15,
        'copy_paste': 0.1,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        
        # PATHS
        'data': 'configs/yolo/fasdd_stage1.yaml',
        'project': 'runs_fast',
        'name': 'fast_test_baseline',
    }

def get_test_configs():
    """Pre-defined test configurations for common experiments."""
    base = get_fast_base_config()
    
    configs = {
        # Learning Rate Sweep
        'lr_sweep': [
            {**base, 'lr0': 0.005, 'name': 'fast_lr_005'},
            {**base, 'lr0': 0.01,  'name': 'fast_lr_010'}, 
            {**base, 'lr0': 0.02,  'name': 'fast_lr_020'},
        ],
        
        # Batch Size Test
        'batch_test': [
            {**base, 'batch': 128, 'name': 'fast_batch_128'},
            {**base, 'batch': 256, 'name': 'fast_batch_256'}, 
            {**base, 'batch': 384, 'name': 'fast_batch_384'},
        ],
        
        # Augmentation Test
        'augment_test': [
            {**base, 'mosaic': 0.5, 'mixup': 0.05, 'name': 'fast_aug_light'},
            {**base, 'mosaic': 1.0, 'mixup': 0.15, 'name': 'fast_aug_medium'},
            {**base, 'mosaic': 1.0, 'mixup': 0.25, 'name': 'fast_aug_heavy'},
        ],
        
        # Optimizer Test
        'optimizer_test': [
            {**base, 'optimizer': 'SGD', 'lr0': 0.01, 'name': 'fast_sgd'},
            {**base, 'optimizer': 'AdamW', 'lr0': 0.001, 'name': 'fast_adamw'},
        ],
        
        # Resolution Test (validate extrapolation)
        'resolution_test': [
            {**base, 'imgsz': 416, 'batch': 384, 'name': 'fast_res_416'},
            {**base, 'imgsz': 512, 'batch': 256, 'name': 'fast_res_512'},
            {**base, 'imgsz': 640, 'batch': 192, 'name': 'fast_res_640'},
        ],
    }
    
    return configs

def run_fast_test(config, test_name):
    """Run a single fast test configuration."""
    
    print(f"\\n🚀 Fast Test: {test_name}")
    print("=" * 50)
    print(f"📊 Configuration:")
    print(f"   Resolution: {config['imgsz']}")
    print(f"   Batch: {config['batch']}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   LR: {config['lr0']}")
    print(f"   Optimizer: {config['optimizer']}")
    
    # Estimate time
    batches_per_epoch = 37413 // config['batch']  # FASDD images
    estimated_time = (batches_per_epoch * config['epochs']) / (2.34 * 60)  # minutes
    
    print(f"   Estimated time: ~{estimated_time:.1f} minutes")
    print("=" * 50)
    
    # Initialize and train
    model = YOLO('yolov8s.pt')
    results = model.train(**config)
    
    # Extract key metrics
    metrics = {
        'final_map50': results.results_dict.get('metrics/mAP50(B)', 0),
        'final_precision': results.results_dict.get('metrics/precision(B)', 0),
        'final_recall': results.results_dict.get('metrics/recall(B)', 0),
        'box_loss': results.results_dict.get('train/box_loss', 0),
        'cls_loss': results.results_dict.get('train/cls_loss', 0),
    }
    
    print(f"\\n✅ {test_name} Results:")
    print(f"   mAP@0.5: {metrics['final_map50']:.3f}")
    print(f"   Precision: {metrics['final_precision']:.3f}") 
    print(f"   Recall: {metrics['final_recall']:.3f}")
    print(f"   Final box_loss: {metrics['box_loss']:.3f}")
    print(f"   Final cls_loss: {metrics['cls_loss']:.3f}")
    
    return results, metrics

def main():
    parser = argparse.ArgumentParser(
        description="SAI-Net H200 Fast Prototyping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=\"\"\"
Fast Testing Examples:
  # Test different learning rates (3 configs, ~15 min each)
  python scripts/train_h200_fast.py --test-config lr_sweep
  
  # Test batch sizes (3 configs, ~15 min each)  
  python scripts/train_h200_fast.py --test-config batch_test
  
  # Single custom test
  python scripts/train_h200_fast.py --single-test --lr0 0.005 --batch 128
        \"\"\"
    )
    
    parser.add_argument("--test-config", type=str, 
                       choices=['lr_sweep', 'batch_test', 'augment_test', 
                               'optimizer_test', 'resolution_test'],
                       help="Pre-defined test configuration to run")
    
    parser.add_argument("--single-test", action="store_true",
                       help="Run single custom test")
    
    # Custom test parameters
    parser.add_argument("--lr0", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch", type=int, default=256, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size")
    parser.add_argument("--epochs", type=int, default=25, help="Epochs")
    parser.add_argument("--optimizer", type=str, default='SGD', help="Optimizer")
    
    args = parser.parse_args()
    
    print("🔥 SAI-Net H200 Fast Prototyping")
    print("=" * 60)
    print("📊 Purpose: Rapid hyperparameter testing & pattern identification")
    print("⏱️  Speed: ~4x faster than production (512 vs 896 resolution)")
    print("🎯 Goal: Test configs in ~15-20 minutes vs ~6 hours")
    print("=" * 60)
    
    all_results = []
    
    if args.test_config:
        # Run pre-defined test suite
        configs = get_test_configs()[args.test_config]
        
        print(f"\\n🧪 Running test suite: {args.test_config}")
        print(f"📋 Number of configs: {len(configs)}")
        
        for i, config in enumerate(configs, 1):
            test_name = f"{args.test_config}_{i}_{config['name']}"
            try:
                results, metrics = run_fast_test(config, test_name)
                all_results.append((test_name, metrics))
            except Exception as e:
                print(f"❌ {test_name} failed: {str(e)}")
                continue
    
    elif args.single_test:
        # Run single custom test
        config = get_fast_base_config()
        config.update({
            'lr0': args.lr0,
            'batch': args.batch, 
            'imgsz': args.imgsz,
            'epochs': args.epochs,
            'optimizer': args.optimizer,
            'name': f'custom_lr{args.lr0}_b{args.batch}_s{args.imgsz}'
        })
        
        results, metrics = run_fast_test(config, "custom_test")
        all_results.append(("custom_test", metrics))
    
    else:
        print("❌ Specify --test-config or --single-test")
        parser.print_help()
        return
    
    # Summary report
    if all_results:
        print("\\n" + "=" * 80)
        print("📊 FAST TEST SUMMARY REPORT")
        print("=" * 80)
        print(f"{'Test Name':<25} {'mAP@0.5':<8} {'Precision':<10} {'Recall':<8} {'Box Loss':<10} {'Cls Loss':<10}")
        print("-" * 80)
        
        for test_name, metrics in all_results:
            print(f"{test_name:<25} {metrics['final_map50']:<8.3f} {metrics['final_precision']:<10.3f} {metrics['final_recall']:<8.3f} {metrics['box_loss']:<10.3f} {metrics['cls_loss']:<10.3f}")
        
        # Find best config
        best_test = max(all_results, key=lambda x: x[1]['final_map50'])
        print("\\n🏆 Best configuration (highest mAP@0.5):")
        print(f"   {best_test[0]}: mAP@0.5 = {best_test[1]['final_map50']:.3f}")
        
        print("\\n💡 Next steps:")
        print("   1. Apply best config to full resolution (896) training")
        print("   2. Scale epochs: fast_epochs × 4 for production")
        print("   3. Adjust batch if needed for memory constraints")

if __name__ == "__main__":
    main()