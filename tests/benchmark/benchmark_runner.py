#!/usr/bin/env python3
"""
SAI-Net Detector Comprehensive Benchmark Runner
Intensive benchmarking for the 4 key production models
"""

import os
import sys
import yaml
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    model_path: str
    model_name: str
    dataset_path: str
    dataset_name: str
    input_size: int = 896
    batch_size: int = 32
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 300

@dataclass 
class BenchmarkResult:
    """Single benchmark result"""
    model_name: str
    dataset_name: str
    total_images: int
    inference_time: float
    fps: float
    memory_usage: float
    map50: Optional[float] = None
    map95: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    detections_count: int = 0
    errors: List[str] = None

class SAINetBenchmarkRunner:
    """Intensive benchmark runner for SAI-Net detector models"""
    
    def __init__(self, config_path: str = "tests/data/configs/test_datasets.yaml"):
        self.config_path = Path(config_path)
        self.results_dir = Path("tests/benchmark/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Setup logging
        self._setup_logging()
        
        # Results storage
        self.benchmark_results: List[BenchmarkResult] = []
        
    def _setup_logging(self):
        """Setup logging configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.results_dir / f"benchmark_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _get_image_files(self, dataset_path: str, extensions: List[str] = None) -> List[Path]:
        """Get all image files from dataset directory"""
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            
        image_files = []
        dataset_path = Path(dataset_path)
        
        for ext in extensions:
            image_files.extend(list(dataset_path.rglob(f"*{ext}")))
            image_files.extend(list(dataset_path.rglob(f"*{ext.upper()}")))
            
        return sorted(image_files)
        
    def _measure_memory_usage(self) -> float:
        """Measure current GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0.0
        
    def benchmark_model_on_dataset(self, model_config: Dict, dataset_config: Dict) -> BenchmarkResult:
        """Benchmark a single model on a single dataset"""
        
        model_name = model_config['description']
        dataset_name = list(dataset_config.keys())[0]
        dataset_info = dataset_config[dataset_name]
        
        self.logger.info(f"🚀 Starting benchmark: {model_name} on {dataset_name}")
        
        try:
            # Load model
            model_path = Path(project_root) / model_config['path']
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
                
            model = YOLO(str(model_path))
            
            # Get dataset images
            image_files = self._get_image_files(dataset_info['path'])
            if not image_files:
                raise ValueError(f"No images found in {dataset_info['path']}")
                
            self.logger.info(f"📊 Found {len(image_files)} images to process")
            
            # Benchmark configuration
            config = BenchmarkConfig(
                model_path=str(model_path),
                model_name=model_name,
                dataset_path=dataset_info['path'], 
                dataset_name=dataset_name,
                **self.config['test_config']
            )
            
            # Initialize metrics
            total_inference_time = 0.0
            detections_count = 0
            errors = []
            
            # Memory measurement
            initial_memory = self._measure_memory_usage()
            
            # Run inference on all images
            self.logger.info(f"⚡ Running inference with batch_size={config.batch_size}")
            
            # Process in batches for memory efficiency
            batch_size = min(config.batch_size, len(image_files))
            
            with tqdm(total=len(image_files), desc=f"Processing {dataset_name}") as pbar:
                for i in range(0, len(image_files), batch_size):
                    batch_files = image_files[i:i + batch_size]
                    
                    try:
                        # Measure inference time
                        start_time = time.time()
                        
                        results = model(
                            [str(f) for f in batch_files],
                            imgsz=config.input_size,
                            conf=config.confidence_threshold,
                            iou=config.iou_threshold,
                            max_det=config.max_detections,
                            verbose=False
                        )
                        
                        end_time = time.time()
                        batch_time = end_time - start_time
                        total_inference_time += batch_time
                        
                        # Count detections
                        for result in results:
                            if hasattr(result, 'boxes') and result.boxes is not None:
                                detections_count += len(result.boxes)
                                
                    except Exception as e:
                        error_msg = f"Batch {i//batch_size + 1} failed: {str(e)}"
                        errors.append(error_msg)
                        self.logger.warning(error_msg)
                        
                    pbar.update(len(batch_files))
            
            # Calculate metrics
            fps = len(image_files) / total_inference_time if total_inference_time > 0 else 0
            peak_memory = self._measure_memory_usage()
            memory_usage = peak_memory - initial_memory
            
            # Create result
            result = BenchmarkResult(
                model_name=model_name,
                dataset_name=dataset_name, 
                total_images=len(image_files),
                inference_time=total_inference_time,
                fps=fps,
                memory_usage=memory_usage,
                detections_count=detections_count,
                errors=errors
            )
            
            self.logger.info(f"✅ Completed: {fps:.2f} FPS, {detections_count} detections, {len(errors)} errors")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Benchmark failed: {str(e)}")
            return BenchmarkResult(
                model_name=model_name,
                dataset_name=dataset_name,
                total_images=0,
                inference_time=0.0,
                fps=0.0,
                memory_usage=0.0,
                errors=[str(e)]
            )
    
    def run_comprehensive_benchmark(self, models: Optional[List[str]] = None, 
                                  datasets: Optional[List[str]] = None) -> List[BenchmarkResult]:
        """Run comprehensive benchmark on all model/dataset combinations"""
        
        # Default to all models and compatible datasets only
        if models is None:
            models = list(self.config['test_models'].keys())
        if datasets is None:
            # Only use compatible datasets for testing
            datasets = []
            for name, config in self.config['test_datasets'].items():
                if name != 'base_path' and isinstance(config, dict) and config.get('compatible', False):
                    datasets.append(name)
            
        self.logger.info(f"🎯 Starting comprehensive benchmark")
        self.logger.info(f"📋 Models: {models}")
        self.logger.info(f"📋 Datasets: {datasets}")
        
        results = []
        total_combinations = len(models) * len(datasets)
        
        with tqdm(total=total_combinations, desc="Benchmark Progress") as pbar:
            for model_name in models:
                model_config = self.config['test_models'][model_name]
                
                for dataset_name in datasets:
                    dataset_config = {dataset_name: self.config['test_datasets'][dataset_name]}
                    
                    result = self.benchmark_model_on_dataset(model_config, dataset_config)
                    results.append(result)
                    self.benchmark_results.append(result)
                    
                    pbar.update(1)
                    
        self.logger.info(f"🏁 Comprehensive benchmark completed: {len(results)} results")
        return results
    
    def save_results(self, results: Optional[List[BenchmarkResult]] = None) -> str:
        """Save benchmark results to file"""
        
        if results is None:
            results = self.benchmark_results
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert results to dict for JSON serialization
        results_dict = {
            'timestamp': timestamp,
            'configuration': self.config,
            'results': []
        }
        
        for result in results:
            results_dict['results'].append({
                'model_name': result.model_name,
                'dataset_name': result.dataset_name,
                'total_images': result.total_images,
                'inference_time': result.inference_time,
                'fps': result.fps,
                'memory_usage': result.memory_usage,
                'detections_count': result.detections_count,
                'error_count': len(result.errors) if result.errors else 0,
                'errors': result.errors or []
            })
        
        # Save JSON results
        json_file = self.results_dir / f"comprehensive_benchmark_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
            
        # Save CSV summary
        csv_file = self.results_dir / f"benchmark_summary_{timestamp}.csv"
        with open(csv_file, 'w') as f:
            f.write("model_name,dataset_name,total_images,inference_time,fps,memory_mb,detections,errors\n")
            for result in results:
                f.write(f"{result.model_name},{result.dataset_name},{result.total_images},"
                       f"{result.inference_time:.3f},{result.fps:.2f},{result.memory_usage:.2f},"
                       f"{result.detections_count},{len(result.errors) if result.errors else 0}\n")
        
        self.logger.info(f"💾 Results saved: {json_file}")
        return str(json_file)

def main():
    """Main benchmark execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SAI-Net Detector Comprehensive Benchmark")
    parser.add_argument("--models", nargs="+", help="Models to benchmark", 
                       choices=["production_final", "stage1_baseline", "pretrained_yolov8s", "yolov11n"])
    parser.add_argument("--datasets", nargs="+", help="Datasets to test on",
                       choices=["fasdd", "pyronear", "dfire", "nemo", "figlib"])
    parser.add_argument("--config", default="tests/data/configs/test_datasets.yaml", 
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    # Initialize benchmark runner
    runner = SAINetBenchmarkRunner(args.config)
    
    # Run comprehensive benchmark
    results = runner.run_comprehensive_benchmark(args.models, args.datasets)
    
    # Save results
    output_file = runner.save_results(results)
    
    print(f"\n🎉 Benchmark completed successfully!")
    print(f"📊 Results saved to: {output_file}")
    print(f"📈 Total combinations tested: {len(results)}")

if __name__ == "__main__":
    main()