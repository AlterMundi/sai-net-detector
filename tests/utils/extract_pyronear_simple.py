#!/usr/bin/env python3
"""
Simple PyroNear extraction - handle train/val separately
"""

import os
import sys
from pathlib import Path
from PIL import Image
import json

sys.path.insert(0, '/tmp/pyronear_venv/lib/python3.13/site-packages')

try:
    from datasets import Dataset
    import pyarrow as pa
    print("✅ Libraries loaded")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def extract_single_split(arrow_path: str, output_dir: Path, split_name: str):
    """Extract single split from arrow file"""
    
    print(f"🔄 Processing {split_name} from {arrow_path}")
    
    try:
        # Load arrow file directly
        table = pa.ipc.RecordBatchFileReader(open(arrow_path, 'rb')).read_all()
        dataset = Dataset(table)
        
        print(f"📊 Found {len(dataset)} samples in {split_name}")
        
        # Create output directory
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = []
        successful = 0
        
        for i, sample in enumerate(dataset):
            try:
                # Extract image
                image = sample['image']
                image_name = sample.get('image_name', f'{split_name}_{i:06d}.jpg')
                
                # Handle different image formats
                if hasattr(image, 'save'):
                    # PIL Image
                    pil_image = image
                elif isinstance(image, dict) and 'bytes' in image:
                    # Bytes format
                    from io import BytesIO
                    pil_image = Image.open(BytesIO(image['bytes']))
                else:
                    # Try direct conversion
                    pil_image = Image.fromarray(image)
                
                # Save image
                image_path = split_dir / image_name
                pil_image.save(str(image_path), 'JPEG', quality=95)
                
                # Save metadata
                metadata.append({
                    'image_name': image_name,
                    'annotations': sample.get('annotations', ''),
                    'partner': sample.get('partner', ''),
                    'camera': sample.get('camera', ''),
                    'date': sample.get('date', '')
                })
                
                successful += 1
                
                if successful % 500 == 0:
                    print(f"  ✅ Extracted {successful} {split_name} images...")
                    
            except Exception as e:
                print(f"  ⚠️ Error with {split_name} image {i}: {e}")
                continue
        
        # Save metadata
        metadata_path = output_dir / f"{split_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"✅ {split_name} complete: {successful} images extracted")
        return successful
        
    except Exception as e:
        print(f"❌ Error processing {split_name}: {e}")
        return 0

def main():
    """Main extraction function"""
    
    base_path = Path("/root/sai-benchmark.old/RNA/data/raw/pyronear-2024")
    output_path = base_path / "extracted_images"
    
    print("🚀 PyroNear Simple Extraction")
    print("=" * 50)
    
    # Extract train split
    train_arrow = base_path / "pyro_train" / "data-00000-of-00006.arrow"
    train_extracted = extract_single_split(str(train_arrow), output_path, "train")
    
    # Extract val split  
    val_arrow = base_path / "pyro_val" / "data-00000-of-00001.arrow"
    val_extracted = extract_single_split(str(val_arrow), output_path, "val")
    
    print("\n" + "="*50)
    print(f"🎉 Extraction Summary:")
    print(f"📊 Train images: {train_extracted}")
    print(f"📊 Val images: {val_extracted}")
    print(f"📁 Output: {output_path}")
    
    if train_extracted > 0 or val_extracted > 0:
        return True
    return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)