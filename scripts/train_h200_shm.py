#!/usr/bin/env python3
"""
SAI-Net H200 Optimized Training Script with /dev/shm cache

Ultra-fast training using:
- 1× NVIDIA H200 (140GB VRAM)  
- 125GB /dev/shm tmpfs cache
- 258GB RAM system
- Images cached in RAM for maximum I/O performance

Based on docs/1xH200_258RAM_train_optimal_settings.md + /dev/shm optimization
"""

import sys
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO

# Import base functions from original script
sys.path.append(str(Path(__file__).parent))
from train_h200 import get_h200_config

def get_h200_shm_config():
    """Get H200-optimized configuration with /dev/shm enhancements."""
    config = get_h200_config()
    
    # /dev/shm optimizations
    config.update({
        # Ultra-fast I/O from RAM cache
        'cache': False,             # No additional cache - images already in /dev/shm
        'project': '/workspace/sai-net-detector/runs',  # Save outputs to repo
        
        # Optimized for RAM-cached data
        'workers': 12,              # Can increase workers (no disk I/O bottleneck)  
    })
    
    return config

def train_stage1_fasdd_shm(epochs=110, test_mode=False):
    """Stage 1: FASDD pre-training with /dev/shm cache."""
    
    print("\n🔥 Stage 1: FASDD Pre-training (/dev/shm cached)")
    print("=" * 60)
    
    # Verify /dev/shm cache exists
    cache_dir = Path("/dev/shm/sai_cache/images/train")
    if not cache_dir.exists():
        raise FileNotFoundError(
            "❌ /dev/shm cache not found. Run scripts/setup_shm_training.sh first"
        )
    
    # Count cached images
    cached_images = len(list(cache_dir.glob("*.jpg")))
    print(f"✅ Found {cached_images:,} images in /dev/shm cache")
    
    # Get optimized config
    config = get_h200_shm_config()
    
    # Stage 1 specific settings
    config.update({
        'data': '/workspace/sai-net-detector/configs/yolo/fasdd_stage1_shm.yaml',
        'epochs': 1 if test_mode else epochs,
        'name': 'h200_shm_stage1_fasdd',
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
    print(f"   Dataset: FASDD ({cached_images:,} images in /dev/shm)")
    print(f"   Classes: fire + smoke (multi-class detection)")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch: {config['batch']} (single GPU)")
    print(f"   Workers: {config['workers']} (optimized for RAM I/O)")
    print(f"   Cache: {config['cache']} (images pre-loaded in RAM)")
    print(f"   Resolution: {config['imgsz']}")
    print(f"   Expected speedup: 2-3× vs disk I/O")
    
    # Initialize model
    model = YOLO('yolov8s.pt')
    
    # Train with /dev/shm optimization
    results = model.train(**config)
    
    print(f"✅ Stage 1 completed with /dev/shm optimization!")
    print(f"   Model: runs/{config['name']}/weights/best.pt")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="SAI-Net H200 Training with /dev/shm Cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test 1-epoch training with /dev/shm cache
  python scripts/train_h200_shm.py --stage 1 --test-mode
  
  # Full Stage 1 training (110 epochs) 
  python scripts/train_h200_shm.py --stage 1 --epochs 110
        """
    )
    
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True,
                       help="Training stage (1: FASDD, 2: PyroSDIS)")
    parser.add_argument("--epochs", type=int,
                       help="Override epochs (Stage 1: 110, Stage 2: 60)")
    parser.add_argument("--test-mode", action="store_true",
                       help="Test mode with 3 epochs")
    
    args = parser.parse_args()
    
    print("🖥️  SAI-Net H200 + /dev/shm Training")
    print("=" * 60)
    print("📊 Hardware Configuration:")
    print(f"   GPU: 1× NVIDIA H200 (140GB VRAM)")
    print(f"   RAM: 258GB system limit")  
    print(f"   Cache: 125GB /dev/shm (tmpfs)")
    print(f"   I/O: Images cached in RAM (50-100× faster)")
    print("=" * 60)
    
    try:
        if args.stage == 1:
            results = train_stage1_fasdd_shm(
                epochs=args.epochs if args.epochs else 110,
                test_mode=args.test_mode
            )
        else:
            print("❌ Stage 2 with /dev/shm not implemented yet")
            return 1
            
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        return 1
    
    print("\n🎉 Training completed successfully with /dev/shm optimization!")
    return 0

if __name__ == "__main__":
    exit(main())
