#!/usr/bin/env python3
"""
Script to explore patient information and prepare for aneurysm cropping
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def explore_patient_data():
    """Explore the patient information from the Excel file"""
    
    # Load the Excel file
    excel_file = "SNHU_TAnDB_DICOM.xlsx"
    
    print(f"Loading patient data from {excel_file}...")
    
    try:
        # Try to read all sheets
        xlsx = pd.ExcelFile(excel_file)
        print(f"Available sheets: {xlsx.sheet_names}")
        
        # Read each sheet
        for sheet_name in xlsx.sheet_names:
            print(f"\n=== Sheet: {sheet_name} ===")
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print("\nFirst few rows:")
            print(df.head())
            
            # Show data types
            print("\nData types:")
            print(df.dtypes)
            
            # Show summary statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print("\nNumeric column statistics:")
                print(df[numeric_cols].describe())
            
            print("\n" + "="*50)
    
    except Exception as e:
        print(f"Error reading Excel file: {e}")

def explore_segmentation_structure():
    """Explore the segmentation folder structure"""
    
    seg_base = Path.home() / "Downloads" / "segmentation" / "aneu" / "UAN_processed" / "Input"
    
    print(f"\nExploring segmentation structure at: {seg_base}")
    
    if seg_base.exists():
        patient_folders = [f for f in seg_base.iterdir() if f.is_dir()]
        print(f"Found {len(patient_folders)} patient folders")
        
        # Show first few patient folders
        for i, folder in enumerate(sorted(patient_folders)[:5]):
            print(f"\nPatient folder {i+1}: {folder.name}")
            
            # Check subfolders
            subfolders = [f.name for f in folder.iterdir() if f.is_dir()]
            print(f"  Subfolders: {subfolders}")
            
            # Check files in key folders
            for subfolder in ['Raw', 'Output', 'Inference']:
                subfolder_path = folder / subfolder
                if subfolder_path.exists():
                    files = list(subfolder_path.glob("*.nii.gz"))
                    print(f"  {subfolder}: {len(files)} .nii.gz files")
                    if files:
                        for file in files:
                            print(f"    - {file.name} ({file.stat().st_size / (1024*1024):.1f} MB)")
    else:
        print(f"Segmentation path not found: {seg_base}")

def create_project_structure():
    """Create the project structure for aneurysm cropping"""
    
    folders = [
        "aneurysm_cropping/data",
        "aneurysm_cropping/output", 
        "aneurysm_cropping/scripts",
        "aneurysm_cropping/logs"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")

if __name__ == "__main__":
    print("=== Aneurysm Cropping Data Exploration ===\n")
    
    # Change to the aneurysm_cropping directory
    os.chdir("aneurysm_cropping")
    
    # Explore patient data
    explore_patient_data()
    
    # Go back to main directory
    os.chdir("..")
    
    # Explore segmentation structure
    explore_segmentation_structure()
    
    # Create project structure
    create_project_structure()
    
    print("\n=== Exploration Complete ===") 