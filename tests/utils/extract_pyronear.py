#!/usr/bin/env python3
"""
Extract PyroNear-2024 images using HuggingFace datasets
"""

import os
import sys
from pathlib import Path
from PIL import Image
import json

# Add virtual environment to path
sys.path.insert(0, '/tmp/pyronear_venv/lib/python3.13/site-packages')

try:
    from datasets import load_dataset
    print("✅ HuggingFace datasets library loaded")
except ImportError as e:
    print(f"❌ Failed to import datasets: {e}")
    sys.exit(1)

def extract_pyronear_images():
    """Extract PyroNear dataset images to origin folder"""
    
    # Load dataset from local arrow files
    dataset_path = "/root/sai-benchmark.old/RNA/data/raw/pyronear-2024"
    output_path = Path(dataset_path) / "extracted_images"
    
    print(f"🚀 Extracting PyroNear images to: {output_path}")
    
    # Create output directories
    (output_path / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "val").mkdir(parents=True, exist_ok=True)
    
    try:
        # Load dataset from local files
        print("📂 Loading dataset from local Arrow files...")
        dataset = load_dataset(dataset_path, cache_dir="/tmp/hf_cache")
        
        print(f"✅ Dataset loaded successfully")
        print(f"📊 Train samples: {len(dataset['train'])}")
        print(f"📊 Val samples: {len(dataset['val'])}")
        
        # Extract training images
        print("🔄 Extracting training images...")
        train_metadata = []
        for i, sample in enumerate(dataset['train']):
            try:
                image = sample['image']
                image_name = sample.get('image_name', f'train_{i:06d}.jpg')
                
                # Save image
                image_path = output_path / "train" / image_name
                image.save(str(image_path), 'JPEG')
                
                # Save metadata
                metadata = {
                    'image_name': image_name,
                    'annotations': sample.get('annotations', ''),
                    'partner': sample.get('partner', ''),
                    'camera': sample.get('camera', ''),
                    'date': sample.get('date', '')
                }
                train_metadata.append(metadata)
                
                if (i + 1) % 1000 == 0:
                    print(f"  ✅ Extracted {i + 1} training images...")
                    
            except Exception as e:
                print(f"  ⚠️ Error extracting train image {i}: {e}")
                continue
        
        # Extract validation images  
        print("🔄 Extracting validation images...")
        val_metadata = []
        for i, sample in enumerate(dataset['val']):
            try:
                image = sample['image']
                image_name = sample.get('image_name', f'val_{i:06d}.jpg')
                
                # Save image
                image_path = output_path / "val" / image_name
                image.save(str(image_path), 'JPEG')
                
                # Save metadata
                metadata = {
                    'image_name': image_name,
                    'annotations': sample.get('annotations', ''),
                    'partner': sample.get('partner', ''),
                    'camera': sample.get('camera', ''),
                    'date': sample.get('date', '')
                }
                val_metadata.append(metadata)
                
                if (i + 1) % 1000 == 0:
                    print(f"  ✅ Extracted {i + 1} validation images...")
                    
            except Exception as e:
                print(f"  ⚠️ Error extracting val image {i}: {e}")
                continue
        
        # Save metadata
        with open(output_path / "train_metadata.json", 'w') as f:
            json.dump(train_metadata, f, indent=2)
            
        with open(output_path / "val_metadata.json", 'w') as f:
            json.dump(val_metadata, f, indent=2)
        
        print(f"✅ Extraction complete!")
        print(f"📊 Train images: {len(train_metadata)}")
        print(f"📊 Val images: {len(val_metadata)}")
        print(f"📁 Output: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

if __name__ == "__main__":
    success = extract_pyronear_images()
    if not success:
        sys.exit(1)