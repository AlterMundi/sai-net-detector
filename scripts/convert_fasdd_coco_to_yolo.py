#!/usr/bin/env python3
"""
Convert FASDD COCO format to YOLO format following the sacred training plan.
Based on docs/planentrenamientoyolov8.md and Guia Descarga PyroSDIS y FASDD.md

This script converts the FASDD dataset from COCO JSON format to YOLO txt format,
maintaining the exact two-class structure (fire=0, smoke=1) as specified in the training plan.

Usage:
    python scripts/convert_fasdd_coco_to_yolo.py \
        --coco-dir /workspace/sai-net-detector/data/raw/fasdd \
        --output-dir /workspace/sai-net-detector/data/yolo \
        --verify
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os


def load_coco_annotation(coco_json_path: Path) -> Dict:
    """Load COCO annotation file."""
    print(f"Loading COCO annotations: {coco_json_path}")
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    print(f"  Images: {len(coco_data.get('images', []))}")
    print(f"  Annotations: {len(coco_data.get('annotations', []))}")
    print(f"  Categories: {len(coco_data.get('categories', []))}")
    
    return coco_data


def get_category_mapping(coco_data: Dict) -> Dict[int, int]:
    """
    Create category mapping from COCO to YOLO format.
    According to the training plan: fire=0, smoke=1
    """
    categories = coco_data.get('categories', [])
    
    # Create mapping based on FASDD format
    category_map = {}
    for cat in categories:
        if cat['name'].lower() == 'fire':
            category_map[cat['id']] = 0  # fire -> 0
        elif cat['name'].lower() == 'smoke':
            category_map[cat['id']] = 1  # smoke -> 1
        else:
            print(f"Warning: Unknown category '{cat['name']}' with id {cat['id']}")
    
    print(f"Category mapping: {category_map}")
    return category_map


def coco_bbox_to_yolo(bbox: List[float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert COCO bbox format to YOLO format.
    COCO: [x_min, y_min, width, height] (absolute coordinates)
    YOLO: [x_center, y_center, width, height] (normalized 0-1)
    """
    x_min, y_min, width, height = bbox
    
    # Convert to center coordinates
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    
    # Normalize to 0-1 range
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return x_center_norm, y_center_norm, width_norm, height_norm


def create_image_to_annotations_map(coco_data: Dict) -> Dict[int, List[Dict]]:
    """Create mapping from image ID to list of annotations."""
    image_annotations = {}
    
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    return image_annotations


def convert_coco_to_yolo(
    coco_dir: Path, 
    output_dir: Path, 
    splits: List[str] = ['train', 'val', 'test']
) -> Dict[str, int]:
    """
    Convert FASDD COCO format to YOLO format.
    
    Returns:
        Statistics about the conversion
    """
    print(f"Converting FASDD COCO to YOLO format")
    print(f"Source: {coco_dir}")
    print(f"Output: {output_dir}")
    
    stats = {}
    
    for split in splits:
        print(f"\n=== Processing {split} split ===")
        
        # Paths
        coco_json = coco_dir / 'annotations' / f'{split}.json'
        images_src = coco_dir / 'images' / split
        
        if not coco_json.exists():
            print(f"Skipping {split}: {coco_json} not found")
            continue
            
        if not images_src.exists():
            print(f"Skipping {split}: {images_src} not found")
            continue
        
        # Output directories
        images_dst = output_dir / 'images' / split
        labels_dst = output_dir / 'labels' / split
        
        images_dst.mkdir(parents=True, exist_ok=True)
        labels_dst.mkdir(parents=True, exist_ok=True)
        
        # Load COCO data
        coco_data = load_coco_annotation(coco_json)
        
        # Get category mapping (fire=0, smoke=1)
        category_map = get_category_mapping(coco_data)
        
        # Create image to annotations mapping
        image_annotations = create_image_to_annotations_map(coco_data)
        
        # Create image ID to image info mapping
        image_info = {img['id']: img for img in coco_data.get('images', [])}
        
        # Convert each image
        converted_images = 0
        converted_annotations = 0
        skipped_no_annotations = 0
        
        for img_info in coco_data.get('images', []):
            img_id = img_info['id']
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Source and destination paths
            img_src_path = images_src / img_filename
            img_dst_path = images_dst / img_filename
            
            label_filename = Path(img_filename).stem + '.txt'
            label_dst_path = labels_dst / label_filename
            
            # Check if image exists
            if not img_src_path.exists():
                print(f"Warning: Image not found: {img_src_path}")
                continue
            
            # Get annotations for this image
            annotations = image_annotations.get(img_id, [])
            
            if not annotations:
                print(f"Skipping {img_filename}: no annotations")
                skipped_no_annotations += 1
                continue
            
            # Copy image
            shutil.copy2(img_src_path, img_dst_path)
            
            # Convert annotations to YOLO format
            yolo_lines = []
            
            for ann in annotations:
                # Get category ID and map to YOLO class
                coco_category_id = ann['category_id']
                
                if coco_category_id not in category_map:
                    print(f"Warning: Unknown category ID {coco_category_id} in {img_filename}")
                    continue
                
                yolo_class_id = category_map[coco_category_id]
                
                # Convert bbox
                bbox = ann['bbox']
                x_center, y_center, width, height = coco_bbox_to_yolo(
                    bbox, img_width, img_height
                )
                
                # Validate bbox coordinates (must be 0-1)
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                       0 < width <= 1 and 0 < height <= 1):
                    print(f"Warning: Invalid bbox in {img_filename}: "
                          f"center=({x_center:.3f},{y_center:.3f}) size=({width:.3f},{height:.3f})")
                    continue
                
                # Create YOLO format line
                yolo_line = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                yolo_lines.append(yolo_line)
                converted_annotations += 1
            
            # Write label file
            if yolo_lines:
                with open(label_dst_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                converted_images += 1
            else:
                # Remove image if no valid annotations
                img_dst_path.unlink()
                print(f"Removed {img_filename}: no valid annotations after conversion")
        
        # Statistics
        stats[split] = {
            'converted_images': converted_images,
            'converted_annotations': converted_annotations,
            'skipped_no_annotations': skipped_no_annotations
        }
        
        print(f"{split} conversion complete:")
        print(f"  Converted images: {converted_images}")
        print(f"  Converted annotations: {converted_annotations}")
        print(f"  Skipped (no annotations): {skipped_no_annotations}")
    
    return stats


def create_data_yaml(output_dir: Path, stats: Dict[str, int]) -> None:
    """Create data.yaml file for YOLO training."""
    
    data_yaml_content = f"""# FASDD dataset configuration for YOLO (converted from COCO)
# Following planentrenamientoyolov8.md - Stage 1: Multi-class fire and smoke detection
# Converted: {', '.join(stats.keys())} splits

train: {output_dir / 'images' / 'train'}
val: {output_dir / 'images' / 'val'}
test: {output_dir / 'images' / 'test'}

# Multi-class configuration (Stage 1: fire and smoke)
nc: 2
names: ['fire', 'smoke']

# Dataset statistics
# Total images: {sum(s['converted_images'] for s in stats.values())}
# Total annotations: {sum(s['converted_annotations'] for s in stats.values())}
"""
    
    data_yaml_path = output_dir / 'data.yaml'
    
    with open(data_yaml_path, 'w') as f:
        f.write(data_yaml_content)
    
    print(f"\nCreated data.yaml: {data_yaml_path}")


def verify_conversion(output_dir: Path) -> Dict[str, int]:
    """Verify the conversion was successful."""
    print(f"\n=== Verifying conversion ===")
    
    verification_stats = {}
    
    for split in ['train', 'val', 'test']:
        images_dir = output_dir / 'images' / split
        labels_dir = output_dir / 'labels' / split
        
        if not images_dir.exists():
            continue
        
        image_files = list(images_dir.glob('*.jpg'))
        label_files = list(labels_dir.glob('*.txt'))
        
        # Check image-label pairs
        image_stems = {f.stem for f in image_files}
        label_stems = {f.stem for f in label_files}
        
        matched_pairs = len(image_stems & label_stems)
        missing_labels = len(image_stems - label_stems)
        missing_images = len(label_stems - image_stems)
        
        verification_stats[split] = {
            'images': len(image_files),
            'labels': len(label_files),
            'matched_pairs': matched_pairs,
            'missing_labels': missing_labels,
            'missing_images': missing_images
        }
        
        print(f"{split}:")
        print(f"  Images: {len(image_files)}")
        print(f"  Labels: {len(label_files)}")
        print(f"  Matched pairs: {matched_pairs}")
        
        if missing_labels > 0:
            print(f"  ⚠️  Missing labels: {missing_labels}")
        if missing_images > 0:
            print(f"  ⚠️  Missing images: {missing_images}")
    
    # Sample some label files to check format
    print(f"\n=== Sample label validation ===")
    
    for split in ['train', 'val']:
        labels_dir = output_dir / 'labels' / split
        if not labels_dir.exists():
            continue
        
        label_files = list(labels_dir.glob('*.txt'))[:3]  # Check first 3 files
        
        for label_file in label_files:
            print(f"Checking {label_file.name}:")
            
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines[:2]):  # Check first 2 lines
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x, y, w, h = parts
                    try:
                        class_id = int(class_id)
                        x, y, w, h = float(x), float(y), float(w), float(h)
                        
                        if class_id in [0, 1] and all(0 <= val <= 1 for val in [x, y, w, h]):
                            print(f"  ✅ Line {i+1}: class={class_id} bbox=({x:.3f},{y:.3f},{w:.3f},{h:.3f})")
                        else:
                            print(f"  ❌ Line {i+1}: Invalid values - class={class_id} bbox=({x:.3f},{y:.3f},{w:.3f},{h:.3f})")
                    except ValueError:
                        print(f"  ❌ Line {i+1}: Cannot parse numbers")
                else:
                    print(f"  ❌ Line {i+1}: Expected 5 values, got {len(parts)}")
    
    return verification_stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert FASDD COCO format to YOLO format following the sacred training plan"
    )
    parser.add_argument(
        '--coco-dir',
        type=Path,
        default='/workspace/sai-net-detector/data/raw/fasdd',
        help='Directory containing COCO format FASDD dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default='/workspace/sai-net-detector/data/yolo',
        help='Output directory for YOLO format dataset'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify conversion after completion'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Dataset splits to process'
    )
    
    args = parser.parse_args()
    
    print("=== FASDD COCO to YOLO Conversion ===")
    print("Following planentrenamientoyolov8.md and Guia Descarga PyroSDIS y FASDD.md")
    print(f"Source: {args.coco_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Splits: {args.splits}")
    
    # Convert
    stats = convert_coco_to_yolo(args.coco_dir, args.output_dir, args.splits)
    
    # Create data.yaml
    create_data_yaml(args.output_dir, stats)
    
    # Verify if requested
    if args.verify:
        verify_conversion(args.output_dir)
    
    print(f"\n=== Conversion Complete ===")
    print(f"YOLO dataset ready at: {args.output_dir}")
    print(f"Use with Stage 1 training: configs/yolo/fasdd_stage1.yaml")
    
    return 0


if __name__ == '__main__':
    exit(main())