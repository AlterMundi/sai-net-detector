"""
YOLOv8 Export Module for SAI-Net Detector
Model export utilities for deployment (ONNX, TensorRT, etc.)
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from ultralytics import YOLO
import torch

def setup_logging():
    """Configure logging for export"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def export_detector(
    model_path: Union[str, Path],
    format: str = "onnx",
    imgsz: Union[int, List[int]] = [1440, 808],
    half: bool = False,
    dynamic: bool = False,
    simplify: bool = True,
    opset: Optional[int] = None,
    workspace: float = 4.0,
    nms: bool = False,
    device: str = "cpu",
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Export YOLOv8 detector to various formats
    
    Args:
        model_path: Path to trained model weights (.pt file)
        format: Export format (onnx, torchscript, engine, tflite, etc.)
        imgsz: Input image size(s)
        half: Use FP16 precision
        dynamic: Dynamic input shapes (ONNX)
        simplify: Simplify ONNX model
        opset: ONNX opset version
        workspace: TensorRT workspace size (GB)
        nms: Add NMS module to model
        device: Export device
        verbose: Verbose output
        **kwargs: Additional export arguments
        
    Returns:
        Export results dictionary
    """
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Supported formats
    supported_formats = [
        "torchscript", "onnx", "openvino", "engine", "coreml", 
        "tflite", "edgetpu", "tfjs", "paddle", "ncnn"
    ]
    
    if format not in supported_formats:
        raise ValueError(f"Unsupported format: {format}. Supported: {supported_formats}")
    
    # Check device
    if device != "cpu" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU for export")
        device = "cpu"
    
    logger.info("=== SAI-Net Detector Export ===")
    logger.info(f"Model: {model_path}")
    logger.info(f"Format: {format}")
    logger.info(f"Image size: {imgsz}")
    logger.info(f"Device: {device}")
    logger.info(f"Half precision: {half}")
    logger.info(f"Dynamic shapes: {dynamic}")
    logger.info("=" * 35)
    
    try:
        # Load model
        model = YOLO(str(model_path))
        
        # Configure export arguments
        export_args = {
            'format': format,
            'imgsz': imgsz,
            'half': half,
            'dynamic': dynamic,
            'simplify': simplify,
            'device': device,
            'verbose': verbose,
            **kwargs
        }
        
        # Format-specific arguments
        if format == "onnx" and opset:
            export_args['opset'] = opset
        elif format == "engine":
            export_args['workspace'] = workspace
        
        if nms:
            export_args['nms'] = nms
        
        logger.info(f"Exporting to {format.upper()}...")
        exported_path = model.export(**export_args)
        
        logger.info("Export completed successfully!")
        logger.info(f"Exported model saved to: {exported_path}")
        
        # Get file size
        exported_file = Path(exported_path)
        file_size_mb = exported_file.stat().st_size / (1024 * 1024)
        
        return {
            'success': True,
            'exported_path': str(exported_path),
            'format': format,
            'file_size_mb': round(file_size_mb, 2),
            'model_info': {
                'input_size': imgsz,
                'precision': 'FP16' if half else 'FP32',
                'dynamic': dynamic
            }
        }
        
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def export_for_deployment(
    model_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Export detector in multiple formats for different deployment scenarios
    
    Args:
        model_path: Path to trained model weights
        output_dir: Output directory for exported models
        
    Returns:
        Results for all export formats
    """
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export configurations for different deployment scenarios
    export_configs = {
        'onnx_cpu': {
            'format': 'onnx',
            'device': 'cpu',
            'half': False,
            'simplify': True,
            'dynamic': False,
            'opset': 11
        },
        'onnx_gpu': {
            'format': 'onnx',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'half': True,
            'simplify': True,
            'dynamic': True,
            'opset': 11
        },
        'torchscript': {
            'format': 'torchscript',
            'device': 'cpu',
            'half': False
        },
        'tensorrt': {
            'format': 'engine',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'half': True,
            'workspace': 4.0
        } if torch.cuda.is_available() else None
    }
    
    results = {}
    
    for config_name, config in export_configs.items():
        if config is None:
            logger.info(f"Skipping {config_name} (CUDA not available)")
            continue
            
        logger.info(f"Exporting {config_name}...")
        
        try:
            result = export_detector(model_path, **config)
            results[config_name] = result
            
            if result['success']:
                logger.info(f"✓ {config_name}: {result['file_size_mb']} MB")
            else:
                logger.error(f"✗ {config_name}: {result['error']}")
                
        except Exception as e:
            logger.error(f"✗ {config_name}: {str(e)}")
            results[config_name] = {'success': False, 'error': str(e)}
    
    return results

def export_best_model(
    experiment_name: str = "sai_yolov8s_optimal_1440x808",
    runs_dir: str = "runs/detect",
    formats: List[str] = ["onnx", "torchscript"]
) -> Dict[str, Any]:
    """
    Export the best model from a training experiment
    
    Args:
        experiment_name: Name of the training experiment
        runs_dir: Directory containing training runs
        formats: List of export formats
        
    Returns:
        Export results for all formats
    """
    
    # Find best model weights
    experiment_dir = Path(runs_dir) / experiment_name
    best_weights = experiment_dir / "weights" / "best.pt"
    
    if not best_weights.exists():
        raise FileNotFoundError(f"Best weights not found: {best_weights}")
    
    results = {}
    
    for fmt in formats:
        result = export_detector(
            model_path=best_weights,
            format=fmt,
            half=True if fmt in ["onnx", "engine"] else False,
            dynamic=True if fmt == "onnx" else False
        )
        results[fmt] = result
    
    return results

def validate_exported_model(
    exported_path: Union[str, Path],
    test_image: Optional[Union[str, Path]] = None,
    confidence: float = 0.25
) -> Dict[str, Any]:
    """
    Validate an exported model with test inference
    
    Args:
        exported_path: Path to exported model
        test_image: Path to test image (optional)
        confidence: Confidence threshold for detections
        
    Returns:
        Validation results
    """
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    exported_path = Path(exported_path)
    if not exported_path.exists():
        raise FileNotFoundError(f"Exported model not found: {exported_path}")
    
    try:
        # Load exported model
        model = YOLO(str(exported_path))
        
        # Test inference
        if test_image:
            if not Path(test_image).exists():
                logger.warning(f"Test image not found: {test_image}")
                test_image = None
        
        if test_image:
            logger.info(f"Running test inference on: {test_image}")
            results = model(test_image, conf=confidence)
            
            # Count detections
            detections = len(results[0].boxes) if results[0].boxes is not None else 0
            
            logger.info(f"Test inference successful: {detections} detections found")
            
            return {
                'success': True,
                'detections': detections,
                'model_loaded': True,
                'inference_working': True
            }
        else:
            logger.info("Model loaded successfully (no test image provided)")
            return {
                'success': True,
                'model_loaded': True,
                'inference_working': None
            }
            
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    # Example usage
    try:
        # Export best model from optimal training
        results = export_best_model()
        
        for fmt, result in results.items():
            if result['success']:
                print(f"✓ {fmt}: {result['exported_path']} ({result['file_size_mb']} MB)")
            else:
                print(f"✗ {fmt}: {result['error']}")
                
    except FileNotFoundError as e:
        print(f"Model not found: {e}")
        print("Train a model first using src.detector.train")