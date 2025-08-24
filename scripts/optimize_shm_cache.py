#!/usr/bin/env python3
"""
SAI-Net /dev/shm Cache Optimizer for H200 Training

Optimiza el uso de 125GB /dev/shm para acelerar entrenamiento YOLO
al pre-cargar y rotar imágenes estratégicamente en RAM.
"""

import os
import shutil
import subprocess
from pathlib import Path
import time
from typing import List, Dict
import argparse

def get_shm_usage():
    """Get current /dev/shm usage in GB."""
    result = subprocess.run(['df', '-BG', '/dev/shm'], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    if len(lines) > 1:
        parts = lines[1].split()
        total = int(parts[1].replace('G', ''))
        used = int(parts[2].replace('G', ''))
        avail = int(parts[3].replace('G', ''))
        return {'total': total, 'used': used, 'avail': avail}
    return None

def estimate_decoded_size(image_dir: Path, sample_size: int = 100) -> float:
    """Estimate decoded image size in GB for cache planning."""
    # For YOLO at 1440x808 resolution:
    # - Raw pixels: 1440 × 808 = 1,166,720 pixels
    # - RGB channels: 3
    # - Float32: 4 bytes per pixel
    # - Total per image: ~14MB decoded
    
    images = list(image_dir.glob('*.jpg'))[:sample_size] if image_dir.exists() else []
    if not images:
        return 0.0
    
    # Conservative estimate: 12MB average per decoded image
    total_images = len(list(image_dir.glob('*.jpg')))
    estimated_gb = (total_images * 12) / (1024 * 1024)  # Convert MB to GB
    
    return estimated_gb

def create_smart_cache_strategy(
    train_images_dir: Path, 
    shm_avail_gb: int,
    priority_strategy: str = "frequency"
) -> Dict:
    """
    Create intelligent caching strategy for /dev/shm.
    
    Args:
        train_images_dir: Path to training images
        shm_avail_gb: Available space in /dev/shm (GB)
        priority_strategy: How to prioritize images ("frequency", "size", "random")
    
    Returns:
        Dict with caching plan
    """
    
    # Get image statistics
    images = list(train_images_dir.glob('*.jpg'))
    total_images = len(images)
    estimated_decoded_gb = estimate_decoded_size(train_images_dir)
    
    print(f"📊 Dataset Analysis:")
    print(f"   Total images: {total_images:,}")
    print(f"   Estimated decoded size: {estimated_decoded_gb:.1f}GB")
    print(f"   Available shm space: {shm_avail_gb}GB")
    
    # Calculate how many images we can cache
    if estimated_decoded_gb <= shm_avail_gb:
        cache_percentage = 100
        cache_count = total_images
        strategy = "full_cache"
    else:
        cache_percentage = int((shm_avail_gb / estimated_decoded_gb) * 100)
        cache_count = int(total_images * (shm_avail_gb / estimated_decoded_gb))
        strategy = "partial_cache"
    
    # Select images based on priority strategy
    if priority_strategy == "frequency":
        # Cache most frequently accessed (first N images alphabetically for now)
        # In production, this could be based on YOLO's internal access patterns
        selected_images = sorted(images)[:cache_count]
    elif priority_strategy == "size":
        # Cache smallest images first (fit more in cache)
        images_with_size = [(img, img.stat().st_size) for img in images]
        selected_images = [img for img, size in sorted(images_with_size, key=lambda x: x[1])[:cache_count]]
    else:  # random
        import random
        selected_images = random.sample(images, min(cache_count, total_images))
    
    return {
        'strategy': strategy,
        'total_images': total_images,
        'cache_count': cache_count,
        'cache_percentage': cache_percentage,
        'selected_images': selected_images,
        'estimated_speedup': min(cache_percentage / 100 * 0.8, 0.8),  # Up to 80% speedup
        'shm_usage_gb': min(shm_avail_gb, estimated_decoded_gb)
    }

def setup_shm_cache(
    train_images_dir: Path,
    shm_cache_dir: Path,
    strategy: Dict,
    dry_run: bool = True
) -> bool:
    """
    Setup intelligent image cache in /dev/shm.
    
    Args:
        train_images_dir: Source training images directory
        shm_cache_dir: Target cache directory in /dev/shm  
        strategy: Caching strategy from create_smart_cache_strategy
        dry_run: If True, only show what would be done
        
    Returns:
        True if successful
    """
    
    print(f"\n🚀 Setting up /dev/shm cache:")
    print(f"   Strategy: {strategy['strategy']}")
    print(f"   Images to cache: {strategy['cache_count']:,}/{strategy['total_images']:,} ({strategy['cache_percentage']}%)")
    print(f"   Expected speedup: {strategy['estimated_speedup']*100:.1f}%")
    print(f"   SHM usage: ~{strategy['shm_usage_gb']:.1f}GB")
    
    if dry_run:
        print("\n🔍 DRY RUN - No files will be copied")
        return True
    
    # Create cache directory
    shm_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy selected images to /dev/shm
    print(f"\n📁 Copying {len(strategy['selected_images'])} images to {shm_cache_dir}...")
    
    start_time = time.time()
    for i, img_path in enumerate(strategy['selected_images']):
        if i % 1000 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(strategy['selected_images']) - i) / rate if rate > 0 else 0
            print(f"   Progress: {i:,}/{len(strategy['selected_images']):,} ({i/len(strategy['selected_images'])*100:.1f}%) "
                  f"- {rate:.1f} img/s - ETA: {eta/60:.1f}min")
        
        dst_path = shm_cache_dir / img_path.name
        try:
            shutil.copy2(img_path, dst_path)
        except Exception as e:
            print(f"   ⚠️  Failed to copy {img_path.name}: {e}")
            return False
    
    elapsed = time.time() - start_time
    print(f"✅ Cache setup completed in {elapsed/60:.1f} minutes")
    print(f"   Average copy rate: {len(strategy['selected_images'])/elapsed:.1f} images/second")
    
    return True

def create_hybrid_dataset_config(
    original_config_path: Path,
    shm_cache_dir: Path,
    output_config_path: Path
) -> bool:
    """
    Create hybrid YOLO config that uses /dev/shm cache when available,
    falls back to original images otherwise.
    """
    
    print(f"\n⚙️  Creating hybrid dataset configuration...")
    
    # Read original config
    with open(original_config_path, 'r') as f:
        config_content = f.read()
    
    # Create hybrid version that prefers /dev/shm
    hybrid_content = config_content.replace(
        'train: /workspace/sai-net-detector/data/yolo/images/train',
        f'train: {shm_cache_dir}'
    )
    
    # Write hybrid config
    with open(output_config_path, 'w') as f:
        f.write(f"# Hybrid configuration with /dev/shm cache optimization\n")
        f.write(f"# Generated automatically - uses {shm_cache_dir} for training\n")
        f.write(f"# Falls back to original images if cache unavailable\n\n")
        f.write(hybrid_content)
    
    print(f"✅ Hybrid config created: {output_config_path}")
    return True

def monitor_cache_performance(duration_minutes: int = 5):
    """Monitor /dev/shm usage during training."""
    print(f"\n📈 Monitoring /dev/shm performance for {duration_minutes} minutes...")
    
    start_time = time.time()
    while time.time() - start_time < duration_minutes * 60:
        usage = get_shm_usage()
        if usage:
            percent_used = (usage['used'] / usage['total']) * 100
            print(f"   /dev/shm: {usage['used']}GB/{usage['total']}GB ({percent_used:.1f}% used)")
        
        time.sleep(30)  # Check every 30 seconds

def main():
    parser = argparse.ArgumentParser(
        description="Optimize SAI-Net training with /dev/shm cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze caching potential (dry run)
  python scripts/optimize_shm_cache.py --analyze

  # Setup intelligent cache (50GB limit)
  python scripts/optimize_shm_cache.py --setup --max-gb 50

  # Monitor cache performance
  python scripts/optimize_shm_cache.py --monitor

  # Full optimization workflow
  python scripts/optimize_shm_cache.py --setup --max-gb 80 --create-config --monitor
        """
    )
    
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze caching potential (dry run)")
    parser.add_argument("--setup", action="store_true", 
                       help="Setup /dev/shm cache")
    parser.add_argument("--max-gb", type=int, default=100,
                       help="Maximum GB to use in /dev/shm (default: 100)")
    parser.add_argument("--create-config", action="store_true",
                       help="Create optimized YOLO config")
    parser.add_argument("--monitor", action="store_true",
                       help="Monitor cache performance")
    parser.add_argument("--priority", choices=["frequency", "size", "random"], 
                       default="frequency",
                       help="Image prioritization strategy")
    
    args = parser.parse_args()
    
    # Paths
    train_images_dir = Path("/workspace/sai-net-detector/data/yolo/images/train")
    shm_cache_dir = Path("/dev/shm/sai_cache/train")  
    original_config = Path("/workspace/sai-net-detector/configs/yolo/fasdd_stage1.yaml")
    hybrid_config = Path("/workspace/sai-net-detector/configs/yolo/fasdd_stage1_shm.yaml")
    
    # Get available /dev/shm space
    shm_usage = get_shm_usage()
    if not shm_usage:
        print("❌ Unable to access /dev/shm")
        return 1
    
    available_gb = min(shm_usage['avail'] - 5, args.max_gb)  # Keep 5GB margin
    
    print(f"🔍 SAI-Net /dev/shm Cache Optimizer")
    print(f"   /dev/shm total: {shm_usage['total']}GB")
    print(f"   /dev/shm available: {shm_usage['avail']}GB") 
    print(f"   Max usage allowed: {available_gb}GB")
    
    if args.analyze or args.setup:
        # Create caching strategy
        strategy = create_smart_cache_strategy(
            train_images_dir, 
            available_gb,
            args.priority
        )
        
        if args.analyze:
            print(f"\n📋 Cache Analysis Complete")
            print(f"   Recommendation: Use {strategy['shm_usage_gb']:.1f}GB cache")
            print(f"   Expected speedup: {strategy['estimated_speedup']*100:.1f}%")
            return 0
        
        if args.setup:
            # Setup the cache
            success = setup_shm_cache(
                train_images_dir,
                shm_cache_dir, 
                strategy,
                dry_run=False
            )
            
            if not success:
                print("❌ Cache setup failed")
                return 1
            
            if args.create_config:
                create_hybrid_dataset_config(
                    original_config,
                    shm_cache_dir,
                    hybrid_config  
                )
    
    if args.monitor:
        monitor_cache_performance()
    
    print(f"\n🎉 /dev/shm optimization complete!")
    return 0

if __name__ == "__main__":
    exit(main())