#!/usr/bin/env python3
"""
SAI-Net H200 Optimized Training Script

Implements two-stage training specifically configured for:
- 1× NVIDIA H200 (140GB VRAM)
- 258GB RAM system limit
- Disk cache (no RAM cache due to memory constraints)

Based on docs/1xH200_258RAM_train_optimal_settings.md
"""

import sys
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO

def get_h200_config():
    """Get H200-optimized configuration."""
    return {
        # Model settings
        'device': 0,                # Single GPU
        'imgsz': [1440, 808],       # Fixed resolution
        
        # Memory-safe batch/workers
        'batch': 96,                # Reduced for VRAM safety (was 128)
        'workers': 8,               # Safe for 258GB RAM
        'cache': 'disk',            # CRITICAL: No RAM cache
        
        # Training settings
        'patience': 10,             # Early stopping
        'amp': True,                # Mixed precision
        'half': True,               # FP16 inference
        'save_period': 3,           # Save every 3 epochs (per user request)
        
        # Optimizer
        'optimizer': 'SGD',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5,
        'cos_lr': True,
        
        # Loss weights
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
    }

def train_stage1_fasdd_h200(epochs=110, test_mode=False):
    """Stage 1: FASDD multi-class pre-training."""
    
    print("\n🔥 Stage 1: FASDD Pre-training (Multi-class)")
    print("=" * 60)
    
    # Get H200 config
    config = get_h200_config()
    
    # Stage 1 specific settings
    config.update({
        'data': 'configs/yolo/fasdd_stage1.yaml',
        'epochs': 3 if test_mode else epochs,
        'project': 'runs',
        'name': 'h200_stage1_fasdd',
        'single_cls': False,  # Multi-class (fire + smoke)
        
        # Aggressive augmentation for Stage 1
        'mosaic': 1.0,
        'mixup': 0.15,
        'copy_paste': 0.1,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
    })
    
    print(f"📊 Configuration:")
    print(f"   Dataset: FASDD (~95K images)")
    print(f"   Classes: fire + smoke (multi-class detection)")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch: {config['batch']} (single GPU)")
    print(f"   Workers: {config['workers']} (memory safe)")
    print(f"   Cache: {config['cache']} (no RAM cache)")
    print(f"   Resolution: {config['imgsz']}")
    
    # Initialize model
    model = YOLO('yolov8s.pt')
    
    # Train
    results = model.train(**config)
    
    print(f"✅ Stage 1 completed!")
    print(f"   Best checkpoint: runs/h200_stage1_fasdd/weights/best.pt")
    
    return results

def train_stage2_pyrosdis_h200(checkpoint=None, epochs=60, test_mode=False):
    """Stage 2: PyroSDIS single-class fine-tuning."""
    
    print("\n💨 Stage 2: PyroSDIS Fine-tuning (Single-class)")
    print("=" * 60)
    
    # Find checkpoint if not provided
    if checkpoint is None:
        checkpoint = Path("runs/h200_stage1_fasdd/weights/best.pt")
        if not checkpoint.exists():
            raise FileNotFoundError(
                "Stage 1 checkpoint not found. Run Stage 1 first or provide --checkpoint"
            )
    
    # Get H200 config
    config = get_h200_config()
    
    # Stage 2 specific settings
    config.update({
        'data': 'data/raw/pyro-sdis/data.yaml',
        'epochs': 3 if test_mode else epochs,
        'project': 'runs',
        'name': 'h200_stage2_pyrosdis',
        'single_cls': True,   # Single-class
        'lr0': 0.001,         # 10× reduced for fine-tuning
        
        # Moderate augmentation for Stage 2
        'mosaic': 0.5,
        'mixup': 0.1,
        'copy_paste': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.5,
        'hsv_v': 0.3,
    })
    
    print(f"📊 Configuration:")
    print(f"   Dataset: PyroSDIS (~33K images)")
    print(f"   Classes: smoke only (single-class)")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch: {config['batch']} (single GPU)")
    print(f"   Workers: {config['workers']} (memory safe)")
    print(f"   Cache: {config['cache']} (no RAM cache)")
    print(f"   Learning rate: {config['lr0']} (reduced)")
    print(f"   Checkpoint: {checkpoint}")
    
    # Load model from Stage 1
    model = YOLO(str(checkpoint))
    
    # Train
    results = model.train(**config)
    
    print(f"✅ Stage 2 completed!")
    print(f"   Final model: runs/h200_stage2_pyrosdis/weights/best.pt")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="SAI-Net H200 Optimized Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete two-stage workflow
  python scripts/train_h200.py --full-workflow
  
  # Run only Stage 1 (FASDD)
  python scripts/train_h200.py --stage 1 --epochs 110
  
  # Run only Stage 2 (PyroSDIS)
  python scripts/train_h200.py --stage 2 --epochs 60
  
  # Test with reduced epochs
  python scripts/train_h200.py --stage 1 --test-mode
        """
    )
    
    parser.add_argument("--full-workflow", action="store_true",
                       help="Execute complete two-stage workflow")
    parser.add_argument("--stage", type=int, choices=[1, 2],
                       help="Run specific stage (1: FASDD, 2: PyroSDIS)")
    parser.add_argument("--epochs", type=int,
                       help="Override epochs (Stage 1: 110, Stage 2: 60)")
    parser.add_argument("--checkpoint", type=str,
                       help="Path to Stage 1 checkpoint for Stage 2")
    parser.add_argument("--test-mode", action="store_true",
                       help="Test mode with 3 epochs")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.full_workflow, args.stage]):
        print("❌ Error: Specify either --full-workflow or --stage")
        parser.print_help()
        sys.exit(1)
    
    print("🖥️  SAI-Net H200 Optimized Training")
    print("=" * 60)
    print("📊 Hardware Configuration:")
    print(f"   GPU: 1× NVIDIA H200 (140GB VRAM)")
    print(f"   RAM: 258GB system limit")
    print(f"   Batch: 128 (optimal for H200)")
    print(f"   Workers: 8 (memory safe)")
    print(f"   Cache: Disk (no RAM cache)")
    print(f"   Resolution: 1440×808")
    print("=" * 60)
    
    try:
        if args.full_workflow:
            # Complete workflow
            print("\n🔄 Executing complete two-stage workflow...")
            
            # Stage 1
            results1 = train_stage1_fasdd_h200(
                epochs=args.epochs if args.epochs else 110,
                test_mode=args.test_mode
            )
            
            # Stage 2
            results2 = train_stage2_pyrosdis_h200(
                epochs=args.epochs if args.epochs else 60,
                test_mode=args.test_mode
            )
            
            print("\n✅ Complete workflow finished!")
            print(f"   Stage 1: runs/h200_stage1_fasdd/")
            print(f"   Stage 2: runs/h200_stage2_pyrosdis/")
            
        elif args.stage == 1:
            # Stage 1 only
            results = train_stage1_fasdd_h200(
                epochs=args.epochs if args.epochs else 110,
                test_mode=args.test_mode
            )
            
        elif args.stage == 2:
            # Stage 2 only
            results = train_stage2_pyrosdis_h200(
                checkpoint=args.checkpoint,
                epochs=args.epochs if args.epochs else 60,
                test_mode=args.test_mode
            )
            
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        sys.exit(1)
    
    print("\n🎉 Training completed successfully!")

if __name__ == "__main__":
    main()