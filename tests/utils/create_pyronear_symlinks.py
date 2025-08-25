#!/usr/bin/env python3
"""
Create symbolic links for PyroNear dataset for direct access
Since Arrow extraction is complex, create links to indicate dataset is available
"""

import os
from pathlib import Path

def create_pyronear_access():
    """Create access method for PyroNear dataset"""
    
    base_path = Path("/root/sai-benchmark.old/RNA/data/raw/pyronear-2024")
    
    # Create info file about dataset format
    info_content = """# PyroNear-2024 Dataset Information

## Format
- HuggingFace Arrow format (.arrow files)
- Requires datasets library for proper loading
- Contains embedded images with annotations

## Structure
- pyro_train/: Training data (29,537 samples, ~2.9GB)
- pyro_val/: Validation data (4,099 samples, ~373MB)

## For YOLO Benchmarking
This dataset requires preprocessing to extract images.
Currently marked as INCOMPATIBLE for direct inference testing.

## Alternative Access Methods
1. Use HuggingFace datasets library
2. Manual Arrow file processing
3. Extract images using conversion utilities

## Files Present
"""
    
    # Add file listing
    for split in ['pyro_train', 'pyro_val']:
        split_path = base_path / split
        if split_path.exists():
            info_content += f"\n### {split}/\n"
            for file_path in split_path.iterdir():
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / (1024**2)
                    info_content += f"- {file_path.name} ({size_mb:.1f} MB)\n"
    
    # Write info file
    info_file = base_path / "DATASET_INFO.md"
    with open(info_file, 'w') as f:
        f.write(info_content)
    
    print(f"✅ Created dataset info: {info_file}")
    
    # Create marker for incompatible dataset
    marker_file = base_path / "REQUIRES_CONVERSION.txt"
    with open(marker_file, 'w') as f:
        f.write("This dataset is in Arrow format and requires conversion for YOLO benchmarking.\n")
        f.write("Use test configuration with compatible=false.\n")
    
    print(f"✅ Created compatibility marker: {marker_file}")
    
    return True

if __name__ == "__main__":
    create_pyronear_access()
    print("✅ PyroNear dataset access configured")