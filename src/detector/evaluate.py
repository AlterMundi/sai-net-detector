"""
YOLOv8 Evaluation Module for SAI-Net Detector
Model validation and performance metrics for wildfire smoke detection
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from ultralytics import YOLO
import torch

def setup_logging():
    """Configure logging for evaluation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def evaluate_detector(
    model_path: Union[str, Path],
    data_yaml: str = "configs/yolo/pyro_fasdd.yaml",
    split: str = "val",
    imgsz: list = [1440, 808],
    batch: int = 32,
    device: str = "0",
    conf: float = 0.001,
    iou: float = 0.6,
    max_det: int = 300,
    half: bool = True,
    plots: bool = True,
    save_json: bool = True,
    save_hybrid: bool = False,
    verbose: bool = True,
    project: Optional[str] = None,
    name: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate YOLOv8 detector performance
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to YOLO dataset configuration
        split: Dataset split to evaluate (val/test)
        imgsz: Input image size
        batch: Batch size for evaluation
        device: GPU device
        conf: Confidence threshold for NMS
        iou: IoU threshold for NMS
        max_det: Maximum detections per image
        half: Use FP16 precision
        plots: Generate validation plots
        save_json: Save results in COCO JSON format
        save_hybrid: Save hybrid labels (for ensemble)
        verbose: Verbose output
        project: Project directory
        name: Experiment name
        **kwargs: Additional arguments
        
    Returns:
        Evaluation results dictionary
    """
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"Data configuration not found: {data_yaml}")
    
    # Check device availability
    if not torch.cuda.is_available() and device != "cpu":
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
        half = False
        batch = batch // 2
    
    logger.info("=== SAI-Net Detector Evaluation ===")
    logger.info(f"Model: {model_path}")
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Split: {split}")
    logger.info(f"Image size: {imgsz[0]}×{imgsz[1]}" if isinstance(imgsz, list) else f"Image size: {imgsz}×{imgsz}")
    logger.info(f"Batch size: {batch}")
    logger.info(f"Device: {device}")
    logger.info(f"Confidence threshold: {conf}")
    logger.info(f"IoU threshold: {iou}")
    logger.info("=" * 40)
    
    try:
        # Load model
        model = YOLO(str(model_path))
        
        # Configure validation arguments
        val_args = {
            'data': data_yaml,
            'split': split,
            'imgsz': imgsz,
            'batch': batch,
            'device': device,
            'conf': conf,
            'iou': iou,
            'max_det': max_det,
            'half': half,
            'plots': plots,
            'save_json': save_json,
            'save_hybrid': save_hybrid,
            'verbose': verbose,
            **kwargs
        }
        
        if project:
            val_args['project'] = project
        if name:
            val_args['name'] = name
        
        logger.info("Starting evaluation...")
        results = model.val(**val_args)
        
        # Extract key metrics
        metrics = {
            'mAP50': float(results.box.map50) if hasattr(results.box, 'map50') else None,
            'mAP50_95': float(results.box.map) if hasattr(results.box, 'map') else None,
            'precision': float(results.box.mp) if hasattr(results.box, 'mp') else None,
            'recall': float(results.box.mr) if hasattr(results.box, 'mr') else None,
            'f1': float(results.box.f1) if hasattr(results.box, 'f1') else None,
        }
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"mAP@0.5: {metrics['mAP50']:.4f}")
        logger.info(f"mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1: {metrics['f1']:.4f}")
        
        return {
            'success': True,
            'metrics': metrics,
            'results': results,
            'save_dir': str(results.save_dir) if hasattr(results, 'save_dir') else None
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def evaluate_best_model(
    experiment_name: str = "sai_yolov8s_optimal_1440x808",
    runs_dir: str = "/dev/shm/rrn/sai-net-detector/runs/detect"
) -> Dict[str, Any]:
    """
    Evaluate the best model from a training experiment
    
    Args:
        experiment_name: Name of the training experiment
        runs_dir: Directory containing training runs
        
    Returns:
        Evaluation results
    """
    
    # Find best model weights
    experiment_dir = Path(runs_dir) / experiment_name
    best_weights = experiment_dir / "weights" / "best.pt"
    
    if not best_weights.exists():
        raise FileNotFoundError(f"Best weights not found: {best_weights}")
    
    return evaluate_detector(
        model_path=best_weights,
        name=f"{experiment_name}_eval"
    )

def benchmark_detector(
    model_path: Union[str, Path],
    data_yaml: str = "configs/yolo/pyro_fasdd.yaml",
    conf_thresholds: list = [0.1, 0.25, 0.5, 0.75, 0.9]
) -> Dict[str, Any]:
    """
    Benchmark detector performance at different confidence thresholds
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to dataset configuration
        conf_thresholds: List of confidence thresholds to test
        
    Returns:
        Benchmark results
    """
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Benchmarking detector at {len(conf_thresholds)} confidence thresholds")
    
    results = {}
    for conf in conf_thresholds:
        logger.info(f"Evaluating at confidence threshold: {conf}")
        
        eval_result = evaluate_detector(
            model_path=model_path,
            data_yaml=data_yaml,
            conf=conf,
            plots=False,
            verbose=False,
            name=f"benchmark_conf_{conf}"
        )
        
        if eval_result['success']:
            results[f'conf_{conf}'] = eval_result['metrics']
        else:
            logger.error(f"Failed to evaluate at conf={conf}: {eval_result['error']}")
            results[f'conf_{conf}'] = None
    
    return {
        'benchmark_results': results,
        'thresholds': conf_thresholds
    }

if __name__ == "__main__":
    # Example usage
    try:
        # Evaluate best model from optimal training
        results = evaluate_best_model()
        print(f"Evaluation results: {results['metrics']}")
        
    except FileNotFoundError as e:
        print(f"Model not found: {e}")
        print("Train a model first using src.detector.train")