#!/usr/bin/env python3
"""
Parallel Benchmarking During Training
Run benchmarks on best.pt while training continues in background

Usage:
    python scripts/benchmark_parallel.py --model-path runs/h200_stage1_fasdd7/weights/best.pt
"""

import argparse
import time
from pathlib import Path
from ultralytics import YOLO
import torch
import pandas as pd
from datetime import datetime
import json

def benchmark_model(model_path: Path, data_config: Path, output_dir: Path):
    """Run comprehensive benchmark on a model checkpoint."""
    
    print(f"🔥 SAI-Net Parallel Benchmark")
    print(f"Model: {model_path}")
    print(f"Data: {data_config}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load model
    print("📥 Loading model...")
    model = YOLO(str(model_path))
    
    # Get model info
    model_info = {
        'model_path': str(model_path),
        'timestamp': datetime.now().isoformat(),
        'model_size_mb': model_path.stat().st_size / (1024*1024),
    }
    
    # Device info
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    model_info['device'] = device
    print(f"🔧 Device: {device}")
    
    # Validation metrics
    print("\n📊 Running validation...")
    start_time = time.time()
    
    # Run validation on test set (separate from training val set)
    results = model.val(
        data=str(data_config),
        split='test',  # Use test split for unbiased evaluation
        save=False,
        verbose=False,
        device=0
    )
    
    val_time = time.time() - start_time
    
    # Extract key metrics
    metrics = {
        'mAP50': float(results.box.map50),
        'mAP50_95': float(results.box.map),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr),
        'validation_time_sec': val_time,
        'images_per_sec': len(results.box.maps) / val_time if val_time > 0 else 0
    }
    
    # Speed benchmark
    print("\n⚡ Running speed benchmark...")
    
    # Warm-up
    dummy_input = torch.randn(1, 3, 896, 896).to('cuda' if torch.cuda.is_available() else 'cpu')
    for _ in range(10):
        _ = model(dummy_input, verbose=False)
    
    # Speed test
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    speed_start = time.time()
    
    num_iterations = 100
    for _ in range(num_iterations):
        _ = model(dummy_input, verbose=False)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    speed_time = time.time() - speed_start
    
    speed_metrics = {
        'inference_time_ms': (speed_time / num_iterations) * 1000,
        'fps': num_iterations / speed_time,
        'batch_size': 1,
        'image_size': [896, 896]
    }
    
    # Combine all metrics
    benchmark_results = {
        **model_info,
        **metrics,
        **speed_metrics
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("📈 BENCHMARK RESULTS")
    print("=" * 60)
    print(f"🎯 mAP@0.5: {metrics['mAP50']:.3f}")
    print(f"🎯 mAP@0.5-0.95: {metrics['mAP50_95']:.3f}")
    print(f"🎯 Precision: {metrics['precision']:.3f}")
    print(f"🎯 Recall: {metrics['recall']:.3f}")
    print(f"⚡ Inference: {speed_metrics['inference_time_ms']:.2f}ms")
    print(f"⚡ FPS: {speed_metrics['fps']:.1f}")
    print(f"📊 Validation time: {val_time:.1f}s")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON results
    results_file = output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    # CSV for tracking over time
    csv_file = output_dir / "benchmark_history.csv"
    df_new = pd.DataFrame([benchmark_results])
    
    if csv_file.exists():
        df_existing = pd.read_csv(csv_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(csv_file, index=False)
    
    print(f"\n💾 Results saved:")
    print(f"   JSON: {results_file}")
    print(f"   CSV: {csv_file}")
    
    return benchmark_results

def monitor_and_benchmark(model_dir: Path, data_config: Path, output_dir: Path, interval_minutes: int = 10):
    """Monitor for new best.pt and benchmark automatically."""
    
    print(f"👁️ Monitoring {model_dir}/best.pt for updates")
    print(f"🕐 Check interval: {interval_minutes} minutes")
    print(f"📊 Data config: {data_config}")
    print("Press Ctrl+C to stop monitoring\n")
    
    best_pt_path = model_dir / "best.pt"
    last_mtime = 0
    
    try:
        while True:
            if best_pt_path.exists():
                current_mtime = best_pt_path.stat().st_mtime
                
                if current_mtime > last_mtime:
                    print(f"\n🆕 Detected updated best.pt at {datetime.now().strftime('%H:%M:%S')}")
                    
                    try:
                        results = benchmark_model(best_pt_path, data_config, output_dir)
                        
                        # Quick comparison with SmokeyNet benchmark
                        smokey_map50 = 0.832  # SmokeyNet equivalent mAP@0.5
                        comparison = results['mAP50'] / smokey_map50 * 100
                        print(f"\n🏆 vs SmokeyNet: {comparison:.1f}% performance")
                        
                        last_mtime = current_mtime
                        
                    except Exception as e:
                        print(f"❌ Benchmark failed: {str(e)}")
                
            else:
                print(f"⚠️ {best_pt_path} not found, waiting...")
            
            print(f"💤 Sleeping {interval_minutes} minutes until next check...")
            time.sleep(interval_minutes * 60)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")

def main():
    parser = argparse.ArgumentParser(description="Parallel benchmarking during training")
    parser.add_argument('--model-path', type=Path, help='Path to model checkpoint (best.pt)')
    parser.add_argument('--model-dir', type=Path, help='Directory containing best.pt (for monitoring)')
    parser.add_argument('--data-config', type=Path, default='configs/yolo/fasdd_stage1.yaml', 
                       help='YOLO data configuration')
    parser.add_argument('--output-dir', type=Path, default='benchmarks', 
                       help='Output directory for results')
    parser.add_argument('--monitor', action='store_true', 
                       help='Monitor for model updates and benchmark automatically')
    parser.add_argument('--interval', type=int, default=10, 
                       help='Monitoring interval in minutes (default: 10)')
    
    args = parser.parse_args()
    
    if args.monitor:
        if not args.model_dir:
            print("❌ --model-dir required for monitoring mode")
            return
        monitor_and_benchmark(args.model_dir, args.data_config, args.output_dir, args.interval)
    
    elif args.model_path:
        benchmark_model(args.model_path, args.data_config, args.output_dir)
    
    else:
        print("❌ Specify either --model-path for single benchmark or --model-dir --monitor for continuous monitoring")

if __name__ == "__main__":
    main()