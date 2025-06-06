#!/usr/bin/env python3
"""
Script to explore patient information and prepare for aneurysm cropping
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("=== Aneurysm Cropping Data Exploration ===\n")
    
    # Check if Excel file exists at new location
    excel_file = Path.home() / "urp" / "data" / "segmentation" / "aneu" / "SNHU_TAnDB_DICOM.xlsx"
    print(f"Checking Excel file: {excel_file}")
    print(f"Excel file exists: {excel_file.exists()}")
    
    # Check segmentation structure
    seg_base = Path.home() / "urp" / "data" / "segmentation" / "aneu" / "UAN_processed" / "Input"
    print(f"\nChecking segmentation structure: {seg_base}")
    print(f"Segmentation path exists: {seg_base.exists()}")
    
    if seg_base.exists():
        patient_folders = [f for f in seg_base.iterdir() if f.is_dir()]
        print(f"Found {len(patient_folders)} patient folders")
        
        # Show first few patient folders
        for i, folder in enumerate(sorted(patient_folders)[:3]):
            print(f"\nPatient folder {i+1}: {folder.name}")
    
    # Create output directories
    base_dir = Path.home() / "urp" / "data"
    output_dirs = [
        base_dir / "aneurysm_cropping",
        base_dir / "smoothing", 
        base_dir / "processed"
    ]
    
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    print("\n=== Exploration Complete ===")

if __name__ == "__main__":
    main() 