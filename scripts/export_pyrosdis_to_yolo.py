#!/usr/bin/env python3
"""
Export PyroSDIS dataset from Hugging Face to YOLO format.
Based on roadmap SAI-Net.md specifications for Stage 2 training.

Usage:
python scripts/export_pyrosdis_to_yolo.py \
  --hf_repo pyronear/pyro-sdis \
  --output data/raw/pyro-sdis \
  --splits train val \
  --single-cls
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from datasets import load_dataset
    import numpy as np
    from PIL import Image
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def setup_directories(output_dir: Path, splits: List[str]):
    """Create directory structure for YOLO format."""
    for split in splits:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    print(f"Created directory structure in {output_dir}")


def convert_bbox_to_yolo(bbox: Dict, img_width: int, img_height: int) -> List[float]:
    """
    Convert PyroSDIS bbox format to YOLO format.
    PyroSDIS uses normalized coordinates, YOLO expects [x_center, y_center, width, height].
    """
    # PyroSDIS bbox format: {'x': x_center, 'y': y_center, 'width': width, 'height': height}
    # Already normalized [0,1], just need to extract values
    x_center = bbox['x']
    y_center = bbox['y'] 
    width = bbox['width']
    height = bbox['height']
    
    return [x_center, y_center, width, height]


def process_split(
    dataset, 
    split: str, 
    output_dir: Path, 
    single_cls: bool = True
) -> Dict[str, int]:
    """Process one split of the PyroSDIS dataset."""
    
    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split
    
    print(f"\nProcessing {split} split...")
    print(f"  Total samples: {len(dataset)}")
    
    stats = {
        'images_processed': 0,
        'annotations_created': 0,
        'smoke_objects': 0,
        'skipped_images': 0
    }
    
    for idx, sample in enumerate(dataset):
        try:
            # Get image and annotations
            image = sample['image']
            annotations_str = sample.get('annotations', '')
            
            # Get image dimensions
            if hasattr(image, 'size'):
                img_width, img_height = image.size
            else:
                img_width, img_height = image.width, image.height
            
            # Create image filename
            image_filename = f"{split}_{idx:06d}.jpg"
            image_path = images_dir / image_filename
            
            # Save image
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(image_path, 'JPEG', quality=95)
            
            # Process annotations - PyroSDIS has annotations as YOLO string
            yolo_annotations = []
            
            if annotations_str and annotations_str.strip():
                # Parse YOLO format string: "class x_center y_center width height"
                annotation_lines = annotations_str.strip().split('\n')
                
                for line in annotation_lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # class + 4 coordinates
                        original_class = parts[0]
                        coordinates = parts[1:5]
                        
                        # PyroSDIS uses class "1" for smoke, convert to "0" if single_cls
                        class_id = 0 if single_cls else int(original_class)
                        
                        # Create YOLO annotation line
                        yolo_line = f"{class_id} {' '.join(coordinates)}"
                        yolo_annotations.append(yolo_line)
                        stats['smoke_objects'] += 1
            
            # Write label file
            label_filename = image_filename.replace('.jpg', '.txt')
            label_path = labels_dir / label_filename
            
            with open(label_path, 'w') as f:
                if yolo_annotations:
                    f.write('\n'.join(yolo_annotations) + '\n')
            
            stats['images_processed'] += 1
            stats['annotations_created'] += len(yolo_annotations)
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(dataset)} images...")
                
        except Exception as e:
            print(f"  Warning: Skipped image {idx} due to error: {str(e)}")
            stats['skipped_images'] += 1
            continue
    
    print(f"  Completed {split} split:")
    print(f"    Images processed: {stats['images_processed']}")
    print(f"    Annotations created: {stats['annotations_created']}")
    print(f"    Smoke objects: {stats['smoke_objects']}")
    print(f"    Skipped images: {stats['skipped_images']}")
    
    return stats


def create_data_yaml(output_dir: Path, splits: List[str]):
    """Create YOLO data.yaml configuration file."""
    
    yaml_content = [
        "# PyroSDIS dataset configuration for YOLO",
        "# SAI-Net Detector Stage 2: Single-class smoke detection",
        "",
    ]
    
    # Add dataset paths (relative to repository root)
    for split in splits:
        rel_path = f"data/raw/pyro-sdis/images/{split}"
        yaml_content.append(f"{split}: {rel_path}")
    
    yaml_content.extend([
        "",
        "# Single-class configuration (smoke detection specialization)",
        "nc: 1",
        "names: ['smoke']",
        "single_cls: true",
        "",
        "# Dataset statistics",
        "# PyroSDIS: ~33,637 images with smoke bounding boxes",
        "# Objective: Domain adaptation to fixed-camera smoke detection"
    ])
    
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write('\n'.join(yaml_content) + '\n')
    
    print(f"Created PyroSDIS data.yaml at {yaml_path}")


def main():
    if not HF_AVAILABLE:
        print("Error: Required packages not installed.")
        print("Please install: pip install datasets pillow numpy")
        return 1
    
    parser = argparse.ArgumentParser(
        description="Export PyroSDIS dataset from Hugging Face to YOLO format"
    )
    parser.add_argument(
        '--hf_repo', 
        default='pyronear/pyro-sdis',
        help='Hugging Face repository name'
    )
    parser.add_argument(
        '--output', 
        default='data/raw/pyro-sdis',
        help='Output directory for YOLO format data'
    )
    parser.add_argument(
        '--splits', 
        nargs='+',
        default=['train', 'val'],
        help='Dataset splits to process'
    )
    parser.add_argument(
        '--single-cls',
        action='store_true',
        default=True,
        help='Use single-class mode (smoke only)'
    )
    parser.add_argument(
        '--cache-dir',
        help='Cache directory for Hugging Face datasets'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    print("=== PyroSDIS to YOLO Conversion ===")
    print(f"Repository: {args.hf_repo}")
    print(f"Output directory: {output_dir}")
    print(f"Splits: {args.splits}")
    print(f"Single-class mode: {args.single_cls}")
    
    # Setup directories
    setup_directories(output_dir, args.splits)
    
    try:
        # Load dataset from Hugging Face
        print(f"\nLoading dataset from {args.hf_repo}...")
        dataset = load_dataset(
            args.hf_repo,
            cache_dir=args.cache_dir,
            trust_remote_code=True
        )
        
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Process each split
        total_stats = {}
        for split in args.splits:
            if split not in dataset:
                print(f"Warning: Split '{split}' not found in dataset, skipping")
                continue
                
            split_stats = process_split(
                dataset[split], 
                split, 
                output_dir, 
                args.single_cls
            )
            
            # Aggregate stats
            for key, value in split_stats.items():
                total_stats[key] = total_stats.get(key, 0) + value
        
        # Create data.yaml
        processed_splits = [s for s in args.splits if s in dataset]
        if processed_splits:
            create_data_yaml(output_dir, processed_splits)
        
        print(f"\n=== Conversion Complete ===")
        print(f"Total images processed: {total_stats.get('images_processed', 0)}")
        print(f"Total annotations: {total_stats.get('annotations_created', 0)}")
        print(f"Total smoke objects: {total_stats.get('smoke_objects', 0)}")
        print(f"Output directory: {output_dir}")
        print(f"\nReady for YOLOv8 Stage 2 training!")
        
        return 0
        
    except Exception as e:
        print(f"Error: Failed to process dataset: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())