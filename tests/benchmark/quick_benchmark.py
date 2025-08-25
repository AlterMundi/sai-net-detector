#!/usr/bin/env python3
"""
Quick SAI-Net Detector Benchmark
Fast performance testing for the 4 key models
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

class QuickBenchmark:
    """Quick benchmark for SAI-Net models"""
    
    def __init__(self):
        self.models = {
            "production_final": "runs/h200_stage2_pyrosdis3/weights/best.pt",
            "stage1_baseline": "runs/h200_stage1_fasdd7/weights/best.pt", 
            "pretrained_yolov8s": "yolov8s.pt",
            "yolov11n": "yolo11n.pt"
        }
        
        self.test_datasets = {
            "fasdd_images": "/root/sai-benchmark.old/RNA/data/raw/fasdd/images/train",
            "figlib_images": "/root/sai-benchmark.old/RNA/data/raw/figlib/train",
            "dfire_images": "/root/sai-benchmark.old/RNA/data/raw/D-Fire/train/images"
        }
        
    def quick_test(self, model_name: str, dataset_path: str, max_images: int = 100):
        """Quick test on limited images"""
        print(f"\n🚀 Quick test: {model_name}")
        
        try:
            # Load model
            model_path = project_root / self.models[model_name]
            if not model_path.exists():
                print(f"❌ Model not found: {model_path}")
                return None
                
            model = YOLO(str(model_path))
            
            # Find test images
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png']:
                image_files.extend(list(Path(dataset_path).rglob(f"*{ext}")))
                
            if not image_files:
                print(f"❌ No images found in {dataset_path}")
                return None
                
            # Limit images for quick test
            test_images = image_files[:max_images]
            print(f"📊 Testing on {len(test_images)} images")
            
            # Benchmark
            start_time = time.time()
            initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            detections = 0
            for img_path in test_images:
                results = model(str(img_path), imgsz=896, verbose=False)
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        detections += len(result.boxes)
            
            end_time = time.time()
            peak_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Calculate metrics
            inference_time = end_time - start_time
            fps = len(test_images) / inference_time
            memory_mb = (peak_memory - initial_memory) / (1024**2)
            
            result = {
                'model': model_name,
                'images': len(test_images),
                'time': inference_time,
                'fps': fps,
                'memory_mb': memory_mb,
                'detections': detections
            }
            
            print(f"⚡ {fps:.2f} FPS | 🧠 {memory_mb:.1f}MB | 🎯 {detections} detections")
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def run_all_quick_tests(self):
        """Run quick tests on all models"""
        print("🎯 SAI-Net Quick Benchmark Suite")
        print("=" * 50)
        
        results = []
        
        for model_name in self.models.keys():
            # Test on FASDD dataset (more images available)
            result = self.quick_test(model_name, self.test_datasets['fasdd_images'])
            if result:
                results.append(result)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"tests/benchmark/results/quick_benchmark_{timestamp}.json"
        
        os.makedirs("tests/benchmark/results", exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({'timestamp': timestamp, 'results': results}, f, indent=2)
            
        print(f"\n💾 Results saved: {output_file}")
        
        # Print summary
        print(f"\n📊 QUICK BENCHMARK SUMMARY")
        print("-" * 50)
        for result in results:
            print(f"{result['model']:<20} | {result['fps']:>6.2f} FPS | {result['memory_mb']:>6.1f}MB | {result['detections']:>4d} det")

if __name__ == "__main__":
    benchmark = QuickBenchmark()
    benchmark.run_all_quick_tests()