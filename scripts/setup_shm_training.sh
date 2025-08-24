#!/bin/bash
"""
SAI-Net /dev/shm Training Setup Script
Optimiza el entrenamiento H200 usando 125GB /dev/shm como cache ultra-rápido
"""

set -e

echo "🚀 SAI-Net /dev/shm Training Optimization"
echo "=========================================="

# Check available space
SHM_AVAIL=$(df -BG /dev/shm | tail -1 | awk '{print $4}' | tr -d 'G')
echo "📊 /dev/shm disponible: ${SHM_AVAIL}GB"

if [ $SHM_AVAIL -lt 50 ]; then
    echo "❌ Insuficiente espacio en /dev/shm (necesitamos al menos 50GB)"
    exit 1
fi

# Create cache directories
echo "📁 Creando directorios de cache..."
mkdir -p /dev/shm/sai_cache/{train,val,labels}
mkdir -p /dev/shm/sai_runs  # Para outputs de entrenamiento

# Strategy 1: Copy training images to /dev/shm for ultra-fast I/O
echo "🔄 Copiando imágenes de entrenamiento a /dev/shm..."
echo "   Fuente: /workspace/sai-net-detector/data/yolo/images/train (48GB)"
echo "   Destino: /dev/shm/sai_cache/train"

# Use rsync for better progress and error handling  
rsync -ah --progress /workspace/sai-net-detector/data/yolo/images/train/ /dev/shm/sai_cache/train/

# Copy validation images
echo "🔄 Copiando imágenes de validación..."
rsync -ah --progress /workspace/sai-net-detector/data/yolo/images/val/ /dev/shm/sai_cache/val/

# Copy labels (small, but important for cache locality)
echo "🔄 Copiando labels..."
cp -r /workspace/sai-net-detector/data/yolo/labels/* /dev/shm/sai_cache/labels/

# Create optimized YOLO config pointing to /dev/shm
echo "⚙️  Creando configuración optimizada..."
cat > /workspace/sai-net-detector/configs/yolo/fasdd_stage1_shm.yaml << 'EOF'
# FASDD Stage 1: Pre-training con /dev/shm optimization
# SAI-Net Two-Stage Workflow - Stage 1 con cache en RAM

# FASDD Dataset - Cache ultra-rápido en /dev/shm (125GB tmpfs)
train: /dev/shm/sai_cache/train
val: /dev/shm/sai_cache/val

# Multi-class configuration (keep original fire + smoke classes)
nc: 2                    # Two classes: fire and smoke
names: ['fire', 'smoke'] # Original FASDD class names

# Optional test set (keep on disk, less frequent access)
test: /workspace/sai-net-detector/data/yolo/images/test

# Performance optimizations for /dev/shm
# Cache settings optimized for RAM-based storage
cache: true              # Enable image cache (will be ultra-fast from RAM)
rect: false              # Disable rectangular training (better for cache efficiency)

# Dataset statistics for Stage 1 (cached in RAM)
# FASDD: ~63,546 images cached in /dev/shm for maximum I/O performance
# Multi-class training for diverse pattern learning
# Expected I/O speedup: 50-100× vs SSD, ~2-3× overall training speedup
# Objective: Learn diverse fire/smoke detection with rich variety
# Expected output: Multi-class detector with good generalization
EOF

# Update H200 training script to use /dev/shm outputs
echo "🔧 Creando script de entrenamiento optimizado..."
cat > /workspace/sai-net-detector/scripts/train_h200_shm.py << 'EOF'
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
        'cache': True,              # Enable cache (images in RAM)
        'project': '/dev/shm/sai_runs',  # Fast output directory
        
        # Optimized for RAM-cached data
        'workers': 12,              # Can increase workers (no disk I/O bottleneck)  
        'prefetch_factor': 4,       # Higher prefetch (RAM is fast)
        
        # Enhanced performance settings
        'persistent_workers': True, # Keep workers alive
        'pin_memory': True,         # Fast GPU transfer
    })
    
    return config

def train_stage1_fasdd_shm(epochs=110, test_mode=False):
    """Stage 1: FASDD pre-training with /dev/shm cache."""
    
    print("\n🔥 Stage 1: FASDD Pre-training (/dev/shm cached)")
    print("=" * 60)
    
    # Verify /dev/shm cache exists
    cache_dir = Path("/dev/shm/sai_cache/train")
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
        'epochs': 3 if test_mode else epochs,
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
    print(f"   Model: {config['project']}/{config['name']}/weights/best.pt")
    
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
EOF

chmod +x /workspace/sai-net-detector/scripts/train_h200_shm.py

# Final status
echo ""
echo "✅ /dev/shm optimization setup complete!"
echo ""
echo "📊 Summary:"
df -h /dev/shm | tail -1
echo ""
echo "🚀 Ready for ultra-fast training:"
echo "   python scripts/train_h200_shm.py --stage 1 --test-mode"
echo ""
EOF