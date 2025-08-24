"""
YOLOv8 Training Module with Forced DDP for SAI-Net Detector
Manual DDP configuration to avoid spawn explosion issues
"""

import os
import logging
import socket
from pathlib import Path
from typing import Optional, Dict, Any
from ultralytics import YOLO
import torch
import torch.distributed as dist

def find_free_port() -> int:
    """Find a free port for DDP communication"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def setup_forced_ddp(device_str: str, interactive: bool = True) -> bool:
    """
    Setup DDP with manual configuration to avoid ultralytics auto-spawn issues
    
    Args:
        device_str: Device string like "0,1"
        interactive: If True, ask user before fallback to single GPU
        
    Returns:
        True if DDP setup successful, False for single GPU fallback, None if user aborted
    """
    
    if "," not in device_str or device_str == "cpu":
        if interactive:
            print(f"⚠️  Single device detected: {device_str}")
            print("❓ Continue with single GPU mode? [y/N]: ", end="", flush=True)
            response = input().strip().lower()
            if response not in ['y', 'yes']:
                print("❌ Training aborted by user")
                return None  # None indicates user abort
        return False  # False indicates single GPU mode
        
    # Parse device list
    device_list = [int(d.strip()) for d in device_str.split(",")]
    world_size = len(device_list)
    
    if world_size < 2:
        if interactive:
            print(f"⚠️  Insufficient devices for DDP: {world_size}")
            print("❓ Continue with single GPU fallback? [y/N]: ", end="", flush=True)
            response = input().strip().lower()
            if response not in ['y', 'yes']:
                print("❌ Training aborted by user")
                return None  # None indicates user abort
        return False  # False indicates single GPU mode
    
    print(f"🔧 Setting up Forced DDP for {world_size} GPUs: {device_list}")
    
    # 1. Clean any existing DDP environment
    ddp_vars = [
        'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 
        'RANK', 'LOCAL_RANK', 'NODE_RANK'
    ]
    
    for var in ddp_vars:
        if var in os.environ:
            print(f"🧹 Clearing existing {var}={os.environ[var]}")
            del os.environ[var]
    
    # 2. Set controlled DDP environment
    master_port = find_free_port()
    
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # 3. NCCL stability settings
    os.environ['NCCL_P2P_DISABLE'] = '1'  # Disable peer-to-peer for stability
    os.environ['NCCL_IB_DISABLE'] = '1'   # Disable InfiniBand
    os.environ['NCCL_DEBUG'] = 'WARN'     # Moderate debugging
    
    # 4. PyTorch DDP settings
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    
    print(f"✅ DDP Environment configured:")
    print(f"   MASTER_ADDR: {os.environ['MASTER_ADDR']}")
    print(f"   MASTER_PORT: {os.environ['MASTER_PORT']}")
    print(f"   WORLD_SIZE: {os.environ['WORLD_SIZE']}")
    print(f"   NCCL_P2P_DISABLE: {os.environ['NCCL_P2P_DISABLE']}")
    
    return True

def setup_logging():
    """Configure logging for training"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training_forceddp.log'),
            logging.StreamHandler()
        ]
    )

def train_detector_forced_ddp(
    data_yaml: str = "configs/yolo/pyro_fasdd.yaml",
    model: str = "yolov8s.pt", 
    imgsz: int = 1440,
    epochs: int = 150,
    batch: int = 60,   # CORRECTED: Match exact successful test (60, not 120)
    device: str = "0,1",
    workers: int = 8,  # CORRECTED: Match exact successful test (8, not 16)
    amp: bool = True,
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
    name: str = "sai_forceddp_test",
    save_period: int = -1,
    single_cls: bool = True,
    interactive: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Train YOLOv8 detector with forced DDP configuration
    
    Args:
        Same as original train_detector but with forced DDP setup
        
    Returns:
        Training results dictionary
    """
    
    # Step 1: Setup forced DDP BEFORE any PyTorch operations
    ddp_enabled = setup_forced_ddp(device, interactive=interactive)
    
    if not ddp_enabled:
        if "," in device:
            print("🔄 Switching to single GPU fallback mode...")
            device = device.split(",")[0] if "," in device else device
            batch = batch * 2 if batch < 200 else batch  # Compensate for single GPU
            print(f"📊 Adjusted configuration:")
            print(f"   Device: {device} (from multi-GPU)")
            print(f"   Batch: {batch} (compensated for single GPU)")
        else:
            print("🔄 Single GPU mode")
            
    # Check if user aborted during DDP setup (setup_forced_ddp returns None for abort)
    if ddp_enabled is None:
        return {
            'success': False,
            'error': 'Training aborted by user during DDP setup',
            'ddp_mode': False,
            'user_aborted': True
        }
    
    # Step 2: Setup logging AFTER DDP environment is configured
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Step 3: Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs(project, exist_ok=True)
    
    # Step 4: Validate inputs
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"Data configuration not found: {data_yaml}")
    
    # Step 5: Hardware validation
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
        batch = batch // 4
        workers = min(workers, 8)
        amp = False
    else:
        gpu_count = torch.cuda.device_count()
        logger.info(f"Available GPUs: {gpu_count}")
        
        if ddp_enabled and "," in str(device):
            device_list = [int(d.strip()) for d in str(device).split(",")]
            if max(device_list) >= gpu_count:
                logger.warning(f"Device {max(device_list)} not available")
                device = ",".join([str(i) for i in range(min(len(device_list), gpu_count))])
    
    # Step 6: Log configuration
    logger.info("=== SAI-Net Detector ForcedDDP Training ===")
    logger.info(f"Model: {model}")
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"DDP Mode: {'Enabled' if ddp_enabled else 'Single GPU'}")
    logger.info(f"Image size: {imgsz}x{imgsz}")
    logger.info(f"Batch size: {batch}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Device(s): {device}")
    logger.info(f"Mixed precision: {amp}")
    logger.info(f"Cache mode: {cache}")
    logger.info("=" * 50)
    
    try:
        # Step 7: Initialize model
        model_instance = YOLO(model)
        
        # Step 8: Configure training arguments
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
        
        logger.info("🚀 Starting ForcedDDP training...")
        results = model_instance.train(**train_args)
        
        logger.info("✅ Training completed successfully!")
        if hasattr(results, 'save_dir') and results.save_dir:
            logger.info(f"Results saved to: {results.save_dir}")
        
        return {
            'success': True,
            'ddp_mode': ddp_enabled,
            'save_dir': str(results.save_dir) if hasattr(results, 'save_dir') and results.save_dir else None,
            'best_fitness': float(results.best_fitness) if hasattr(results, 'best_fitness') else None,
            'metrics': results.results_dict if hasattr(results, 'results_dict') else None
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        
        # Enhanced error reporting for DDP issues
        if "distributed" in str(e).lower() or "nccl" in str(e).lower():
            logger.error("🔴 DDP-specific error detected")
            
            if interactive and ddp_enabled:
                print("\n" + "="*60)
                print("⚠️  DDP TRAINING FAILED")
                print(f"Error: {str(e)}")
                print("="*60)
                print("❓ Do you want to retry with single GPU mode? [y/N]: ", end="", flush=True)
                
                retry_response = input().strip().lower()
                if retry_response in ['y', 'yes']:
                    print("🔄 Retrying with single GPU configuration...")
                    
                    # Recursive call with single GPU
                    single_gpu_device = device.split(",")[0] if "," in str(device) else device
                    return train_detector_forced_ddp(
                        data_yaml=data_yaml, model=model, imgsz=imgsz, epochs=epochs,
                        batch=batch*2 if batch < 200 else batch,  # Compensate batch
                        device=single_gpu_device, workers=workers, amp=amp, cos_lr=cos_lr,
                        lr0=lr0, lrf=lrf, momentum=momentum, weight_decay=weight_decay,
                        warmup_epochs=warmup_epochs, warmup_momentum=warmup_momentum,
                        warmup_bias_lr=warmup_bias_lr, box=box, cls=cls, dfl=dfl,
                        hsv_h=hsv_h, hsv_s=hsv_s, hsv_v=hsv_v, degrees=degrees,
                        translate=translate, scale=scale, shear=shear, mosaic=mosaic,
                        mixup=mixup, copy_paste=copy_paste, close_mosaic=close_mosaic,
                        cache=cache, project=project, name=f"{name}_singlegpu_retry",
                        save_period=save_period, single_cls=single_cls,
                        interactive=False,  # Disable interactive for retry
                        **kwargs
                    )
                else:
                    print("❌ Training aborted by user")
            
        return {
            'success': False,
            'error': str(e),
            'ddp_mode': ddp_enabled,
            'recommendation': 'single_gpu' if 'distributed' in str(e).lower() else 'investigate',
            'interactive_handled': interactive and ddp_enabled
        }

def train_optimal_forced_ddp():
    """
    Final production configuration optimized for 2×A100 GPUs, 500GB RAM
    Based on successful 1-epoch test with proven stable parameters
    EXACT MATCH to test configuration that achieved mAP50: 47.8%
    """
    return train_detector_forced_ddp(
        data_yaml="configs/yolo/pyro_fasdd.yaml",
        model="yolov8s.pt",
        imgsz=1440,  # High resolution for small smoke detection
        epochs=150,  # Full training cycle (test used 1 epoch)
        batch=60,    # EXACT MATCH: test used 60, not 120
        device="0,1",
        workers=8,   # EXACT MATCH: test used 8, not 16
        amp=True,    # Mixed precision for memory efficiency
        cos_lr=True, # Cosine learning rate scheduler
        lr0=0.01,    # Initial learning rate
        lrf=0.01,    # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,      # Gradual warmup
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Loss weights optimized for smoke detection
        box=7.5,     # Emphasize bounding box accuracy
        cls=0.5,     # Classification weight (single class)
        dfl=1.5,     # Distribution focal loss for small objects
        # Augmentation optimized for wildfire smoke
        hsv_h=0.015, # Minimal hue variation (preserve smoke color)
        hsv_s=0.7,   # Moderate saturation changes
        hsv_v=0.4,   # Value/brightness variations
        degrees=5,   # Minimal rotation (smoke is often vertical)
        translate=0.1, # Small translation
        scale=0.5,   # Scale augmentation for size variation
        shear=2.0,   # Geometric shearing
        mosaic=1.0,  # Full mosaic augmentation
        mixup=0.1,   # Light mixup for regularization
        copy_paste=0.0,  # No copy-paste (can confuse smoke detection)
        close_mosaic=15, # Stop mosaic in last 15 epochs
        cache="ram",     # RAM caching for performance
        project="/dev/shm/rrn/sai-net-detector/runs",
        name="sai_final_production_1440x808",
        single_cls=True  # Single smoke class detection
    )

if __name__ == "__main__":
    # Test forced DDP configuration
    results = train_optimal_forced_ddp()
    
    if results['success']:
        print(f"✅ ForcedDDP training successful!")
        print(f"Mode: {'DDP' if results['ddp_mode'] else 'Single GPU'}")
        print(f"Results: {results['save_dir']}")
        if results.get('best_fitness'):
            print(f"Best fitness: {results['best_fitness']:.4f}")
    else:
        print(f"❌ ForcedDDP training failed: {results['error']}")
        if results.get('recommendation'):
            print(f"💡 Recommendation: {results['recommendation']}")