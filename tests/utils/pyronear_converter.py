#!/usr/bin/env python3
"""
PyroNear-2024 Arrow Dataset Converter
Convert HuggingFace Arrow format to standard images for YOLO benchmarking
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False

def extract_pyronear_without_huggingface(dataset_path: str, output_path: str) -> bool:
    """
    Extract PyroNear images without HuggingFace datasets library
    Uses direct Arrow file reading
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    if not ARROW_AVAILABLE:
        print("❌ PyArrow not available. Cannot process Arrow files.")
        return False
        
    # Find Arrow files
    arrow_files = list(dataset_path.glob("*.arrow"))
    if not arrow_files:
        print(f"❌ No .arrow files found in {dataset_path}")
        return False
        
    print(f"📂 Found {len(arrow_files)} Arrow files")
    
    # Create output directories
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "metadata").mkdir(parents=True, exist_ok=True)
    
    total_images = 0
    metadata_list = []
    
    for arrow_file in arrow_files:
        print(f"🔄 Processing {arrow_file.name}...")
        
        try:
            # Read Arrow file
            table = pa.ipc.RecordBatchFileReader(arrow_file.open('rb')).read_all()
            
            # Convert to pandas for easier handling
            df = table.to_pandas()
            
            for idx, row in df.iterrows():
                try:
                    # Extract image data
                    if 'image' in row:
                        image_data = row['image']
                        if hasattr(image_data, 'bytes'):
                            image_bytes = image_data.bytes
                        else:
                            image_bytes = image_data
                            
                        # Get image name
                        if 'image_name' in row:
                            image_name = row['image_name']
                        else:
                            image_name = f"image_{total_images:06d}.jpg"
                            
                        # Save image
                        image_path = output_path / "images" / image_name
                        with open(image_path, 'wb') as f:
                            f.write(image_bytes)
                            
                        # Save metadata
                        metadata = {
                            'image_name': image_name,
                            'annotations': row.get('annotations', ''),
                            'partner': row.get('partner', ''),
                            'camera': row.get('camera', ''),
                            'date': row.get('date', '')
                        }
                        metadata_list.append(metadata)
                        
                        total_images += 1
                        
                        if total_images % 1000 == 0:
                            print(f"  ✅ Extracted {total_images} images...")
                            
                except Exception as e:
                    print(f"  ⚠️ Error processing row {idx}: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Error processing {arrow_file}: {e}")
            continue
    
    # Save metadata JSON
    metadata_file = output_path / "metadata" / "pyronear_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    
    print(f"✅ Extraction complete: {total_images} images saved to {output_path}")
    return total_images > 0

def create_simple_image_extractor(dataset_path: str, output_path: str) -> bool:
    """
    Simple approach - try to find any recognizable image files in Arrow structure
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    print(f"🔍 Searching for extractable image data in {dataset_path}")
    
    # Create output directory
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    
    # Look for any files that might contain images
    all_files = list(dataset_path.rglob("*"))
    print(f"📁 Found {len(all_files)} files total")
    
    # Check if we can extract using basic file operations
    for file_path in all_files:
        if file_path.is_file() and file_path.suffix in ['.arrow']:
            print(f"  📄 {file_path.name}: {file_path.stat().st_size / 1024**2:.1f} MB")
    
    # For now, create dummy structure to show it's accessible
    dummy_file = output_path / "CONVERSION_NEEDED.txt"
    with open(dummy_file, 'w') as f:
        f.write(f"""PyroNear-2024 Dataset Conversion Required

Dataset location: {dataset_path}
Format: HuggingFace Arrow files
Files found: {len(all_files)}

To extract images, you need either:
1. HuggingFace datasets library: pip install datasets
2. Manual Arrow file processing

Arrow files present:
""")
        for file_path in all_files:
            if file_path.is_file():
                f.write(f"- {file_path.name} ({file_path.stat().st_size / 1024**2:.1f} MB)\n")
    
    return False

def main():
    """Main conversion function"""
    
    pyronear_path = "/root/sai-benchmark.old/RNA/data/raw/pyronear-2024"
    output_path = "/root/sai-benchmark.old/RNA/data/raw/pyronear-2024/extracted_images"
    
    print("🚀 PyroNear-2024 Dataset Converter")
    print("=" * 50)
    
    if not Path(pyronear_path).exists():
        print(f"❌ Dataset not found: {pyronear_path}")
        return False
    
    # Try Arrow-based extraction first
    if ARROW_AVAILABLE:
        print("✅ PyArrow available - attempting direct extraction")
        success = extract_pyronear_without_huggingface(pyronear_path, output_path)
        if success:
            return True
    
    # Fallback: Simple file analysis
    print("⚠️ PyArrow not available - using simple analysis")
    return create_simple_image_extractor(pyronear_path, output_path)

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 To properly extract PyroNear images:")
        print("1. Install: pip install datasets pyarrow")
        print("2. Or use HuggingFace Hub directly")
        sys.exit(1)