"""
YOLOv8 Training Module for SAI-Net Detector
Wildfire smoke detection using PyroSDIS + FASDD datasets
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from ultralytics import YOLO
import torch

def setup_logging():
    """Configure logging for training"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler()
        ]
    )

def train_detector(
    data_yaml: str = "configs/yolo/pyro_fasdd.yaml",
    model: str = "yolov8s.pt", 
    imgsz: int = 1440,
    epochs: int = 150,
    batch: int = 120,
    device: str = "0,1",
    workers: int = 79,
    amp: str = "bf16",
    cos_lr: bool = True,
    lr0: float = 0.01,
    lrf: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    warmup_epochs: int = 5,
    warmup_momentum: float = 0.8,
    warmup_bias_lr: float = 0.1,
    box: float = 7.5,
    cls: float = 0.5,
    dfl: float = 1.5,
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    degrees: float = 5.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 2.0,
    mosaic: float = 1.0,
    mixup: float = 0.1,
    copy_paste: float = 0.0,
    close_mosaic: int = 15,
    cache: str = "ram",
    project: str = "/dev/shm/rrn/sai-net-detector/runs",
    name: str = "sai_yolov8s_optimal",
    save_period: int = -1,
    single_cls: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Train YOLOv8 detector for wildfire smoke detection
    
    Args:
        data_yaml: Path to YOLO dataset configuration
        model: Model architecture (yolov8s.pt, yolov8m.pt)
        imgsz: Input image size 
        epochs: Number of training epochs
        batch: Batch size (distributed across GPUs)
        device: GPU devices (e.g., "0,1" for DDP)
        workers: Number of data loading workers
        amp: Mixed precision (bf16/fp16)
        cos_lr: Use cosine learning rate scheduler
        lr0: Initial learning rate
        lrf: Final learning rate factor
        momentum: SGD momentum
        weight_decay: Weight decay
        warmup_epochs: Warmup epochs
        warmup_momentum: Warmup momentum
        warmup_bias_lr: Warmup bias learning rate
        box: Box loss weight
        cls: Classification loss weight
        dfl: Distribution Focal Loss weight
        hsv_h: HSV hue augmentation
        hsv_s: HSV saturation augmentation
        hsv_v: HSV value augmentation
        degrees: Rotation degrees
        translate: Translation fraction
        scale: Scaling factor
        shear: Shearing degrees
        mosaic: Mosaic augmentation probability
        mixup: MixUp augmentation probability
        copy_paste: Copy-paste augmentation probability
        close_mosaic: Epochs to stop mosaic
        cache: Cache mode (ram/disk)
        project: Project directory
        name: Experiment name
        save_period: Save checkpoint every N epochs
        single_cls: Single class mode
        **kwargs: Additional arguments
        
    Returns:
        Training results dictionary
    """
    
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs(project, exist_ok=True)
    
    # Validate data configuration
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"Data configuration not found: {data_yaml}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
        batch = batch // 4  # Reduce batch for CPU
        workers = min(workers, 8)
        amp = False
    else:
        gpu_count = torch.cuda.device_count()
        logger.info(f"Available GPUs: {gpu_count}")
        if "," in device:
            device_list = [int(d.strip()) for d in device.split(",")]
            if max(device_list) >= gpu_count:
                logger.warning(f"Device {max(device_list)} not available, using available GPUs")
                device = ",".join([str(i) for i in range(min(len(device_list), gpu_count))])
    
    # Log hardware optimization
    logger.info("=== SAI-Net Detector Training Configuration ===")
    logger.info(f"Model: {model}")
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Image size: {imgsz}x{imgsz}")
    logger.info(f"Batch size: {batch}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Device(s): {device}")
    logger.info(f"Mixed precision: {amp}")
    logger.info(f"Cache mode: {cache}")
    logger.info("=" * 50)
    
    try:
        # Initialize model
        model = YOLO(model)
        
        # Configure training arguments
        train_args = {
            'data': data_yaml,
            'imgsz': imgsz,
            'epochs': epochs,
            'batch': batch,
            'device': device,
            'workers': workers,
            'amp': amp,
            'cos_lr': cos_lr,
            'lr0': lr0,
            'lrf': lrf,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'warmup_epochs': warmup_epochs,
            'warmup_momentum': warmup_momentum,
            'warmup_bias_lr': warmup_bias_lr,
            'box': box,
            'cls': cls,
            'dfl': dfl,
            'hsv_h': hsv_h,
            'hsv_s': hsv_s,
            'hsv_v': hsv_v,
            'degrees': degrees,
            'translate': translate,
            'scale': scale,
            'shear': shear,
            'mosaic': mosaic,
            'mixup': mixup,
            'copy_paste': copy_paste,
            'close_mosaic': close_mosaic,
            'cache': cache,
            'project': project,
            'name': name,
            'save_period': save_period,
            'single_cls': single_cls,
            'verbose': True,
            'plots': True,
            'val': True,
            **kwargs
        }
        
        logger.info("Starting training...")
        results = model.train(**train_args)
        
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {results.save_dir}")
        
        # Return results summary
        return {
            'success': True,
            'save_dir': str(results.save_dir),
            'best_fitness': float(results.best_fitness) if hasattr(results, 'best_fitness') else None,
            'metrics': results.results_dict if hasattr(results, 'results_dict') else None
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def train_optimal():
    """
    Train detector with optimal configuration for 2×A100 GPUs, 500GB RAM
    Resolution: 1440×808 (high-res), Batch: 120 (VRAM optimized)
    """
    return train_detector(
        data_yaml="configs/yolo/pyro_fasdd.yaml",
        model="yolov8s.pt",
        imgsz=1440,
        epochs=150,
        batch=120,
        device="0,1",
        workers=79,
        amp="bf16",
        cos_lr=True,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,
        close_mosaic=15,
        cache="ram",
        project="/dev/shm/rrn/sai-net-detector/runs",
        name="sai_yolov8s_optimal_1440x808",
        single_cls=True
    )

def train_conservative():
    """
    Train detector with conservative configuration (fallback)
    Resolution: 960×960 (standard), Batch: 64 (low VRAM)
    """
    return train_detector(
        data_yaml="configs/yolo/pyro_fasdd.yaml",
        model="yolov8s.pt",
        imgsz=960,
        batch=64,
        workers=16,
        cache="disk",
        name="sai_yolov8s_conservative_960x960"
    )

if __name__ == "__main__":
    # Run optimal configuration
    results = train_optimal()
    print(f"Training results: {results}")