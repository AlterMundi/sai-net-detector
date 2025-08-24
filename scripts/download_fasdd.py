#!/usr/bin/env python3
"""
Download FASDD dataset using Kaggle API.
Based on roadmap SAI-Net.md specifications for Stage 1 training.

Prerequisites:
1. Install Kaggle API: pip install kaggle
2. Setup Kaggle credentials: kaggle.json in ~/.kaggle/ or KAGGLE_USERNAME/KAGGLE_KEY env vars
3. Accept dataset terms on Kaggle website

Usage:
python scripts/download_fasdd.py \
  --output data/raw/fasdd \
  --extract \
  --verify
"""

import argparse
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Optional


def check_kaggle_setup() -> bool:
    """Check if Kaggle API is properly setup."""
    try:
        import kaggle
        # Test API access
        result = subprocess.run(
            ['kaggle', '--version'], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.returncode == 0:
            print(f"Kaggle API available: {result.stdout.strip()}")
            return True
        else:
            print(f"Kaggle API error: {result.stderr}")
            return False
    except ImportError:
        print("Error: Kaggle package not installed")
        print("Install with: pip install kaggle")
        return False
    except subprocess.TimeoutExpired:
        print("Error: Kaggle API timeout")
        return False
    except Exception as e:
        print(f"Error checking Kaggle setup: {str(e)}")
        return False


def find_fasdd_dataset() -> Optional[str]:
    """
    Find the correct FASDD dataset identifier on Kaggle.
    Returns the dataset identifier or None if not found.
    """
    
    # Common FASDD dataset identifiers to try
    potential_datasets = [
        'firebase/fasdd',
        'datafire/fasdd-dataset',
        'fasdd/firebase-smoke-detection-dataset',
        'firebase-ai/fasdd',
        'fasdd-dataset/firebase-smoke-detection'
    ]
    
    print("Searching for FASDD dataset on Kaggle...")
    
    for dataset_id in potential_datasets:
        try:
            result = subprocess.run(
                ['kaggle', 'datasets', 'list', '-s', dataset_id.split('/')[-1]],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and dataset_id.lower() in result.stdout.lower():
                print(f"Found FASDD dataset: {dataset_id}")
                return dataset_id
                
        except subprocess.TimeoutExpired:
            continue
        except Exception:
            continue
    
    print("Could not automatically find FASDD dataset.")
    print("Please manually search on Kaggle and provide the dataset identifier.")
    return None


def download_dataset(dataset_id: str, output_dir: Path, force: bool = False) -> bool:
    """Download dataset from Kaggle."""
    
    print(f"Downloading {dataset_id} to {output_dir}...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        cmd = ['kaggle', 'datasets', 'download', dataset_id, '-p', str(output_dir)]
        if force:
            cmd.append('--force')
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout for large downloads
        )
        
        if result.returncode == 0:
            print("Download completed successfully!")
            return True
        else:
            print(f"Download failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("Download timed out (1 hour limit)")
        return False
    except Exception as e:
        print(f"Download error: {str(e)}")
        return False


def extract_dataset(output_dir: Path) -> bool:
    """Extract downloaded ZIP files."""
    
    print("Extracting dataset files...")
    
    zip_files = list(output_dir.glob("*.zip"))
    if not zip_files:
        print("No ZIP files found to extract")
        return False
    
    extracted = False
    for zip_file in zip_files:
        try:
            print(f"Extracting {zip_file.name}...")
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Extract to same directory
                zip_ref.extractall(output_dir)
            
            print(f"Extracted {zip_file.name}")
            extracted = True
            
            # Optionally remove ZIP file after extraction
            # zip_file.unlink()  # Uncomment to delete ZIP after extraction
            
        except zipfile.BadZipFile:
            print(f"Error: {zip_file.name} is not a valid ZIP file")
            continue
        except Exception as e:
            print(f"Error extracting {zip_file.name}: {str(e)}")
            continue
    
    return extracted


def verify_dataset_structure(output_dir: Path) -> Dict[str, int]:
    """Verify the downloaded dataset has expected structure."""
    
    print("Verifying dataset structure...")
    
    stats = {
        'total_images': 0,
        'annotation_files': 0,
        'directories': 0
    }
    
    # Count files by extension
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    annotation_extensions = {'.json', '.txt', '.xml'}
    
    for item in output_dir.rglob("*"):
        if item.is_file():
            ext = item.suffix.lower()
            if ext in image_extensions:
                stats['total_images'] += 1
            elif ext in annotation_extensions:
                stats['annotation_files'] += 1
        elif item.is_dir():
            stats['directories'] += 1
    
    print(f"Dataset verification results:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Annotation files: {stats['annotation_files']}")
    print(f"  Directories: {stats['directories']}")
    
    # Check if this looks like a valid dataset
    if stats['total_images'] > 1000 and stats['annotation_files'] > 0:
        print("✅ Dataset structure looks valid")
        return stats
    else:
        print("⚠️  Dataset structure may be incomplete")
        return stats


def organize_fasdd_structure(output_dir: Path) -> bool:
    """
    Organize FASDD dataset into expected COCO format structure.
    Expected: images/ and annotations/ directories.
    """
    
    print("Organizing FASDD dataset structure...")
    
    # Check if already organized
    if (output_dir / "images").exists() and (output_dir / "annotations").exists():
        print("Dataset already organized in COCO format")
        return True
    
    # Find images and annotations
    all_images = []
    all_annotations = []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    annotation_extensions = {'.json'}
    
    for item in output_dir.rglob("*"):
        if item.is_file():
            ext = item.suffix.lower()
            if ext in image_extensions:
                all_images.append(item)
            elif ext in annotation_extensions:
                all_annotations.append(item)
    
    if not all_images:
        print("No images found to organize")
        return False
    
    # Create organized structure
    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"
    
    images_dir.mkdir(exist_ok=True)
    annotations_dir.mkdir(exist_ok=True)
    
    # Move images
    print(f"Moving {len(all_images)} images...")
    for img in all_images:
        dest = images_dir / img.name
        if not dest.exists():
            shutil.move(str(img), str(dest))
    
    # Move annotations
    print(f"Moving {len(all_annotations)} annotation files...")
    for ann in all_annotations:
        dest = annotations_dir / ann.name
        if not dest.exists():
            shutil.move(str(ann), str(dest))
    
    print("Dataset organized in COCO format structure")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download FASDD dataset using Kaggle API"
    )
    parser.add_argument(
        '--dataset-id',
        help='Kaggle dataset identifier (e.g., "username/dataset-name")'
    )
    parser.add_argument(
        '--output',
        default='data/raw/fasdd',
        help='Output directory for downloaded dataset'
    )
    parser.add_argument(
        '--extract',
        action='store_true',
        default=True,
        help='Extract ZIP files after download'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        default=True,
        help='Verify dataset structure after download'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if files exist'
    )
    parser.add_argument(
        '--organize',
        action='store_true',
        default=True,
        help='Organize dataset into COCO format structure'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    print("=== FASDD Dataset Download ===")
    print(f"Output directory: {output_dir}")
    
    # Check Kaggle setup
    if not check_kaggle_setup():
        print("\nKaggle API setup required:")
        print("1. Install: pip install kaggle")
        print("2. Get API key from kaggle.com/account")
        print("3. Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME/KAGGLE_KEY env vars")
        print("4. Accept dataset terms on Kaggle website")
        return 1
    
    # Determine dataset ID
    dataset_id = args.dataset_id
    if not dataset_id:
        dataset_id = find_fasdd_dataset()
        if not dataset_id:
            print("\nPlease provide the FASDD dataset identifier with --dataset-id")
            print("Example: python scripts/download_fasdd.py --dataset-id username/fasdd-dataset")
            return 1
    
    print(f"Using dataset: {dataset_id}")
    
    # Download dataset
    success = download_dataset(dataset_id, output_dir, args.force)
    if not success:
        print("Failed to download dataset")
        return 1
    
    # Extract if requested
    if args.extract:
        extract_success = extract_dataset(output_dir)
        if not extract_success:
            print("Failed to extract dataset")
            return 1
    
    # Organize structure
    if args.organize:
        organize_success = organize_fasdd_structure(output_dir)
        if not organize_success:
            print("Failed to organize dataset structure")
            return 1
    
    # Verify if requested
    if args.verify:
        stats = verify_dataset_structure(output_dir)
    
    print(f"\n=== Download Complete ===")
    print(f"Dataset location: {output_dir}")
    print(f"Next step: Convert to YOLO format with:")
    print(f"python scripts/convert_fasdd_to_yolo.py --src {output_dir} --dst data/yolo")
    print(f"\nReady for YOLOv8 Stage 1 training!")
    
    return 0


if __name__ == "__main__":
    exit(main())