#!/usr/bin/env python3
"""
Convert FASDD dataset from COCO format to YOLO format.
Based on roadmap SAI-Net.md specifications.

Usage:
python scripts/convert_fasdd_to_yolo.py \
  --src data/raw/fasdd \
  --dst data/yolo \
  --split-ratios 0.9 0.1 \
  --map-classes smoke
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_coco_annotations(annotation_file: str) -> Tuple[Dict, List, List]:
    """Load COCO format annotations."""
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    images = {img['id']: img for img in data['images']}
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    annotations = data['annotations']
    
    return images, categories, annotations


def coco_to_yolo_bbox(coco_bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """
    Convert COCO bbox format [x_min, y_min, width, height] to YOLO format
    [x_center, y_center, width, height] normalized by image dimensions.
    """
    x_min, y_min, width, height = coco_bbox
    
    # Calculate center coordinates
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    
    # Normalize width and height
    norm_width = width / img_width
    norm_height = height / img_height
    
    return [x_center, y_center, norm_width, norm_height]


def convert_split(
    src_dir: str,
    dst_dir: str,
    split: str,
    map_classes: Optional[str] = None
) -> Dict[str, int]:
    """Convert one split from COCO to YOLO format."""
    
    # Create output directories
    images_dir = Path(dst_dir) / "images" / split
    labels_dir = Path(dst_dir) / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Load COCO annotations
    annotation_file = Path(src_dir) / "annotations" / f"{split}.json"
    if not annotation_file.exists():
        print(f"Warning: {annotation_file} not found, skipping {split}")
        return {}
    
    images, categories, annotations = load_coco_annotations(annotation_file)
    
    print(f"Converting {split} split:")
    print(f"  Images: {len(images)}")
    print(f"  Categories: {categories}")
    print(f"  Annotations: {len(annotations)}")
    
    # Group annotations by image
    image_annotations = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    # Class mapping
    class_counts = {name: 0 for name in categories.values()}
    
    if map_classes == "smoke":
        # Map both fire and smoke to single class 0 (smoke)
        class_mapping = {cat_id: 0 for cat_id in categories.keys()}
        yolo_classes = ['smoke']
    else:
        # Keep original classes: fire=0, smoke=1
        class_mapping = {cat_id: cat_id for cat_id in categories.keys()}
        yolo_classes = [categories[i] for i in sorted(categories.keys())]
    
    print(f"  Class mapping: {class_mapping}")
    print(f"  YOLO classes: {yolo_classes}")
    
    converted_count = 0
    
    # Process each image
    for img_id, img_info in images.items():
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Copy image file
        src_img_path = Path(src_dir) / "images" / split / img_filename
        dst_img_path = images_dir / img_filename
        
        if not src_img_path.exists():
            print(f"Warning: Image {src_img_path} not found, skipping")
            continue
        
        shutil.copy2(src_img_path, dst_img_path)
        
        # Convert annotations to YOLO format
        yolo_annotations = []
        if img_id in image_annotations:
            for ann in image_annotations[img_id]:
                category_id = ann['category_id']
                bbox = ann['bbox']
                
                # Convert bbox from COCO to YOLO format
                yolo_bbox = coco_to_yolo_bbox(bbox, img_width, img_height)
                
                # Map class
                yolo_class = class_mapping[category_id]
                
                # Count classes
                original_class_name = categories[category_id]
                class_counts[original_class_name] += 1
                
                # Create YOLO annotation line
                yolo_line = f"{yolo_class} {' '.join(map(str, yolo_bbox))}"
                yolo_annotations.append(yolo_line)
        
        # Write YOLO label file
        label_filename = img_filename.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = labels_dir / label_filename
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
            if yolo_annotations:  # Add newline at end if file is not empty
                f.write('\n')
        
        converted_count += 1
    
    print(f"  Converted {converted_count} images")
    print(f"  Class distribution: {class_counts}")
    
    return class_counts


def create_data_yaml(dst_dir: str, yolo_classes: List[str], splits: List[str]):
    """Create YOLO data.yaml configuration file."""
    
    data_yaml = {
        'train': f"{dst_dir}/images/train",
        'val': f"{dst_dir}/images/val",
        'nc': len(yolo_classes),
        'names': yolo_classes
    }
    
    # Add test split if it exists
    if 'test' in splits:
        data_yaml['test'] = f"{dst_dir}/images/test"
    
    yaml_path = Path(dst_dir) / "data.yaml"
    
    # Write YAML manually (simple format)
    with open(yaml_path, 'w') as f:
        f.write(f"# FASDD dataset configuration for YOLO\n")
        for key, value in data_yaml.items():
            if isinstance(value, str):
                f.write(f"{key}: {value}\n")
            elif isinstance(value, int):
                f.write(f"{key}: {value}\n")
            elif isinstance(value, list):
                f.write(f"{key}: {value}\n")
    
    print(f"Created data.yaml at {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert FASDD dataset from COCO to YOLO format"
    )
    parser.add_argument(
        '--src', 
        required=True,
        help='Source directory containing COCO format data'
    )
    parser.add_argument(
        '--dst', 
        required=True,
        help='Destination directory for YOLO format data'
    )
    parser.add_argument(
        '--split-ratios', 
        nargs='+', 
        type=float,
        default=[0.9, 0.1],
        help='Split ratios (currently only used for reference, actual splits determined by existing files)'
    )
    parser.add_argument(
        '--map-classes', 
        choices=['smoke', 'original'],
        default='smoke',
        help='Class mapping strategy: "smoke" maps both classes to smoke, "original" keeps fire/smoke separate'
    )
    
    args = parser.parse_args()
    
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    
    if not src_dir.exists():
        print(f"Error: Source directory {src_dir} does not exist")
        return
    
    # Create destination directory
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect available splits
    annotation_dir = src_dir / "annotations"
    available_splits = []
    for split_file in annotation_dir.glob("*.json"):
        split_name = split_file.stem
        available_splits.append(split_name)
    
    print(f"Found splits: {available_splits}")
    
    # Convert each split
    all_class_counts = {}
    yolo_classes = None
    
    for split in available_splits:
        class_counts = convert_split(src_dir, dst_dir, split, args.map_classes)
        
        # Aggregate class counts
        for class_name, count in class_counts.items():
            all_class_counts[class_name] = all_class_counts.get(class_name, 0) + count
        
        # Set YOLO classes from first split
        if yolo_classes is None:
            if args.map_classes == "smoke":
                yolo_classes = ['smoke']
            else:
                # Determine from categories in first split
                annotation_file = src_dir / "annotations" / f"{split}.json"
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                categories = {cat['id']: cat['name'] for cat in data['categories']}
                yolo_classes = [categories[i] for i in sorted(categories.keys())]
    
    # Create data.yaml
    if yolo_classes:
        create_data_yaml(dst_dir, yolo_classes, available_splits)
    
    print(f"\nConversion complete!")
    print(f"Total class distribution: {all_class_counts}")
    print(f"Output directory: {dst_dir}")


if __name__ == "__main__":
    main()